import os
import time
from argparse import ArgumentParser, Namespace

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, PREPARED_IMAGE_SIZE, PREPARED_IMAGES_PATH, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Frame, Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D
from wormlab3d.dynamics.clusterer import DynamicsClusterer
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.plot_utils import tex_mode, make_3d_posture_plot_for_animation
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

tex_mode()
width = PREPARED_IMAGE_SIZE[0] * 3 * 2
height = PREPARED_IMAGE_SIZE[1] * 3
cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to generate a cluster video.')
    parser.add_argument('--checkpoint', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--distance-metric', type=str, help='Distance metric.')
    parser.add_argument('--linkage-method', type=str, help='Linkage method to use.')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters to plot.')
    parser.add_argument('--step', type=int, help='Window step size.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    assert args.step > 0, 'Step size must be > 0.'

    print_args(args)

    return args


def _make_traces_plot(
        args: Namespace,
        reconstruction: Reconstruction,
):
    """
    Build a traces plot.
    Returns an update function to call which updates the axis.
    """

    # Instantiate clusterer and calculate clusters.
    DC = DynamicsClusterer(
        checkpoint_id=args.checkpoint,
        reconstruction_id=args.reconstruction,
        step=args.step
    )
    L, distances = DC.cluster(
        distance_metric=args.distance_metric,
        linkage_method=args.linkage_method,
    )
    ws = DC.dataset_args.sample_duration

    # Load trajectory
    common_args = {
        'reconstruction_id': reconstruction.id,
        'smoothing_window': DC.dataset_args.smoothing_window
    }
    X, meta = get_trajectory(**common_args)
    N = len(X)
    if args.x_label == 'time':
        ts = np.linspace(0, N / reconstruction.trial.fps, N)
        t_range = 5
    else:
        ts = np.arange(N) + reconstruction.start_frame
        t_range = 5 * reconstruction.trial.fps

    # Speed
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True)

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    r = pcas.explained_variance_ratio.T
    nonp = r[2] / np.sqrt(r[1] * r[0])

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True)
    X_ew = DC.manager.ew.transform(np.array(Z))

    # Plot
    fig, axes = plt.subplots(3, figsize=(int(width / 2 / 100), int(height / 100)), sharex=True)

    # Speeds
    ax1 = axes[0]
    ax1.axhline(y=0, color='darkgrey')
    ax1.plot(ts, speeds)
    ax1.set_ylabel('Speed (mm/s)')
    ax1_marker = ax1.axvline(x=0, color='red')

    # Planarities
    ax2 = ax1.twinx()
    ax2.plot(ts, nonp, color='orange', alpha=0.6, linestyle='--')
    ax2.set_ylabel('Non-planarity', rotation=270, labelpad=15)
    ax2.axhline(y=0, color='darkgrey')

    # Eigenworms - absolute values
    ax3 = axes[1]
    for i in args.plot_components:
        ax3.plot(
            ts,
            np.abs(X_ew[:, i]),
            label=f'$\lambda_{i + 1}$',
            alpha=0.7,
            linewidth=1
        )
    ax3_marker = ax3.axvline(x=0, color='red')
    ax3.set_ylabel('$|\lambda|$')
    ax3.legend(loc=2)

    # Cluster plots
    logger.info(f'Clustering into {args.n_clusters} clusters.')
    clusters = fcluster(L, args.n_clusters, criterion='maxclust')
    unique, counts = np.unique(clusters, return_counts=True)
    sorted_idxs = np.argsort(counts)[::-1]
    y_locs = np.arange(1, args.n_clusters + 1)
    ax4 = axes[2]
    ax4.set_ylabel('Cluster index')
    for j in range(args.n_clusters):
        cluster_idx = sorted_idxs[j] + 1
        zs = np.argwhere(clusters == cluster_idx)[:, 0]
        xs = []
        for z in zs:
            xs.extend(np.arange(z * args.step, (z + 1) * args.step) + int((ws - args.step) / 2))
        ys = np.ones_like(xs) * y_locs[j]
        ax4.scatter(ts[xs], ys, label=cluster_idx)
    ax4_marker = ax4.axvline(x=0, color='red')
    ax4.set_yticks(y_locs)

    if args.x_label == 'time':
        ax4.set_xlabel('Time (s)')
    else:
        ax4.set_xlabel('Frame #')

    def update(frame_idx: int):
        # Update the axis limits
        ax1.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])

        # Move the markers
        ax1_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax3_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax4_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def _generate_annotated_images(image_triplet: np.ndarray, points_2d: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = []
    for i, img_array in enumerate(image_triplet):
        z = ((1 - img_array) * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)

        # Overlay 2d projection
        p2d = points_2d[:, i]
        for j, p in enumerate(p2d):
            z = cv2.drawMarker(
                z,
                p,
                color=colours[j].tolist(),
                markerType=cv2.MARKER_CROSS,
                markerSize=3,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(
                    z,
                    p2d[j - 1],
                    p2d[j],
                    color=colours[j].tolist(),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

        images.append(z)

    return np.concatenate(images).transpose(1, 0, 2)


def generate_cluster_video():
    """
    Generate a posture video showing a 3D midline reconstruction with camera images
    with overlaid 2D midline reprojections alongside eigenworm and cluster traces.
    """
    args = parse_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Build output path
    path = LOGS_PATH / f'{START_TIMESTAMP}' \
                       f'_r={reconstruction.id}' \
                       f'_cp={args.checkpoint}' \
                       f'_d={args.distance_metric}' \
                       f'_l={args.linkage_method}' \
                       f'_n={args.n_clusters}' \
                       f'_x={args.x_label}.mp4'

    if reconstruction.source == M3D_SOURCE_MF:
        D = reconstruction.mf_parameters.depth - 1  # todo - different depth videos
        ts = TrialState(reconstruction=reconstruction)
        from_idx = sum([2**d2 for d2 in range(D)])
        to_idx = from_idx + 2**D

        # Get 3D postures
        all_points = ts.get('points')
        X_full = all_points[:, from_idx:to_idx]

        # Get 2D projections
        all_projections = ts.get('points_2d')  # (M, N, 3, 2)
        points_2d = np.round(all_projections[:, from_idx:to_idx]).astype(np.int32)

        # Colour map
        colours = np.array([cmap(d) for d in np.linspace(0, 1, 2**D)])
        colours = np.round(colours * 255).astype(np.uint8)

    else:
        X_full, _ = get_trajectory(reconstruction_id=reconstruction.id)

    # Build plots
    # fig_clusters, update_plot_clusters = _make_cluster_block(args, reconstruction, ew)
    fig_traces, update_plot_traces = _make_traces_plot(args, reconstruction)
    fig_3d, update_plot_3d = make_3d_posture_plot_for_animation(X_full=X_full, width=width, height=height)

    # Fetch the images
    logger.info('Querying database.')
    pipeline = [
        {'$match': {
            'trial': reconstruction.trial.id,
            'frame_num': {
                '$gte': reconstruction.start_frame,
                '$lte': reconstruction.end_frame,
            }
        }},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
        }},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Initialise ffmpeg process
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': reconstruction.trial.fps,
        'metadata:g:0': f'title=Trial {reconstruction.trial.id}. Reconstruction {reconstruction.id}.',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(str(path), **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    frame_nums = []
    logger.info('Rendering frames.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i > 0 and i % 100 == 0:
            logger.info(f'Rendering frame {i}/{reconstruction.n_frames}.')

        # Check we don't miss any frames
        if i == 0:
            n0 = n
        assert n == n0 + i

        # Check images are present
        img_path = PREPARED_IMAGES_PATH / f'{reconstruction.trial.id:03d}' / f'{n:06d}.npz'
        try:
            image_triplet = np.load(img_path)['images']
        except Exception:
            logger.warning('Prepared images not available, stopping here.')
            break
        frame_nums.append(n)

        # Generate the annotated images
        if reconstruction.source == M3D_SOURCE_MF:
            images = _generate_annotated_images(image_triplet, points_2d[n], colours)

        else:
            m3d = Midline3D.objects.get(
                frame=res['_id'],
                source=reconstruction.source,
                source_file=reconstruction.source_file,
            )

            # Get 2D projections
            points_2d = np.round(m3d.get_prepared_2d_coordinates(regenerate=False)).astype(np.int32)
            points_2d = points_2d.transpose(1, 0, 2)

            # Colour map
            colours = np.array([cmap(i) for i in np.linspace(0, 1, points_2d.shape[0])])
            colours = np.round(colours * 255).astype(np.uint8)

            # Prepare images
            images = _generate_annotated_images(image_triplet, points_2d, colours)

        # Update the plots and extract rendered images
        update_plot_3d(i)
        update_plot_traces(i)
        plot_3d = np.asarray(fig_3d.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        # Join plots to images and write to stream
        frame_left = np.concatenate([images, plot_3d], axis=0)
        frame = np.concatenate([frame_left, plot_traces], axis=1)
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    logger.info(f'Generated video for frames {frame_nums[0]}-{frame_nums[-1]} ({frame_nums[-1] - frame_nums[0]}). '
                f'Total frames in reconstruction = {reconstruction.n_frames}.')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_cluster_video()
