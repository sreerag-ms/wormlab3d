import json
import os
import shutil
import time
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath
from typing import Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import logger, PREPARED_IMAGES_PATH, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Frame, Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import print_args, str2bool, normalise, to_dict
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import calculate_speeds, smooth_trajectory


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate a manoeuvre video.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')

    # Spec
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')

    # Trajectory
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--use-valid-range', type=str2bool, help='Use valid range if available.')
    parser.add_argument('--smoothing-window-postures', type=int, help='Smoothing window for the postures.')
    parser.add_argument('--smoothing-window-components', type=int, help='Smoothing window for the components.')

    # 3D plots
    parser.add_argument('--show-posture-axis', type=str2bool, default=True,
                        help='Show axis on the 3D posture plot.')
    parser.add_argument('--show-posture-grid', type=str2bool, default=True,
                        help='Show grid on the 3D posture plot.')
    parser.add_argument('--show-posture-ticks', type=str2bool, default=False,
                        help='Show ticks on the 3D posture plot.')
    parser.add_argument('--show-posture-tick-labels', type=str2bool, default=False,
                        help='Show tick labels on the 3D posture plot.')
    parser.add_argument('--revolution-rate', type=float, default=1 / 3,
                        help='Rate of 3D plot revolution in revolutions/minute.')

    # Traces
    parser.add_argument('--time-range-traces', type=float, default=5,
                        help='Time range to show on trace plots in seconds.')
    parser.add_argument('--eigenworms', type=str, help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20, help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')

    # Lambdas
    parser.add_argument('--time-range-lambdas', type=float, default=5,
                        help='Time range to show on lambdas plots in seconds.')

    args = parser.parse_args()
    assert args.reconstruction is not None or args.spec is not None, \
        'This script requires setting --reconstruction=id or --spec=path.'

    print_args(args)

    return args


def _make_3d_posture_plot(
        width: int,
        height: int,
        trial: Trial,
        X: np.ndarray,
        args: Namespace
) -> Tuple[Figure, Callable]:
    """
    Build a 3D posture plot.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D postures plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    gs = GridSpec(1, 1, top=0.99, bottom=0.01, left=0.01, right=0.99)
    ax = fig.add_subplot(gs[0, 0], projection='3d')

    # Add postures
    FS = FrameSequenceNumpy(x=X.transpose(0, 2, 1))
    fa = FrameArtist(F=FS[0])
    fa.add_midline(ax)

    # Setup axis
    equal_aspect_ratio(ax)
    if not args.show_posture_axis:
        ax.axis('off')
    if args.show_posture_grid:
        ax.grid()
    else:
        ax.grid(False)
    if not args.show_posture_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if not args.show_posture_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Aspects
    n_revolutions = len(X) / trial.fps / 60 * args.revolution_rate
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X))
    ax.view_init(azim=azims[0])

    def update(frame_idx: int):
        # Rotate the view.
        ax.view_init(azim=azims[frame_idx])

        # Update the worm
        F = FS[frame_idx]
        fa.update(F)
        bb = F.get_range()
        for i, axis in enumerate('xyz'):
            getattr(ax, f'set_{axis}lim')(bb[0][i], bb[1][i])
        equal_aspect_ratio(ax)

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _make_traces_plots(
        width: int,
        height: int,
        reconstruction: Reconstruction,
        X: np.ndarray,
        X_ew: np.ndarray,
        curvature: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a traces plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building traces plot.')
    N = len(X)
    trial = reconstruction.trial
    if args.x_label == 'time':
        ts = np.linspace(0, N / trial.fps, N)
        t_range = args.time_range_traces
    else:
        ts = np.arange(N) + reconstruction.start_frame
        t_range = args.time_range_traces * trial.fps

    # Plot
    fig, axes = plt.subplots(2, figsize=(width / 100, height / 100), gridspec_kw={
        'hspace': 0,
        'top': 0.98,
        'bottom': 0.07,
        'left': 0.15,
        'right': 0.86,
    })

    # Curvature
    ax_curvature = axes[0]
    im = ax_curvature.imshow(curvature.T, aspect='auto', cmap='Reds', origin='lower', extent=(0, ts[-1], 0, 1))
    cax = ax_curvature.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax_curvature.transAxes)
    fig.colorbar(im, ax=ax_curvature, cax=cax)
    ht_args = dict(transform=ax_curvature.transAxes, horizontalalignment='right', fontweight='bold')
    ax_curvature.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax_curvature.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax_curvature.set_ylabel('$\kappa$', fontsize=12, labelpad=10)
    ax_curvature.set_xticks([])
    ax_curvature.set_yticks([0, 1])
    ax_curvature.set_yticklabels([])
    ax_curvature.spines['top'].set_visible(False)
    ax_curvature.spines['bottom'].set_visible(False)
    ax_curvature_marker = ax_curvature.axvline(x=0, color='red')

    # Eigenworms - absolute values
    ax_ew = axes[1]
    for i in args.plot_components:
        ax_ew.plot(
            ts,
            np.abs(X_ew[:, i]),
            label=f'$\lambda_{i + 1}$',
            alpha=0.7,
            linewidth=1
        )
    ax_ew.set_ylabel('$|\lambda|$')
    ax_ew.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax_ew.spines['top'].set_visible(False)
    ax_ew_marker = ax_ew.axvline(x=0, color='red')

    if args.x_label == 'time':
        ax_ew.set_xlabel('Time (s)')
    else:
        ax_ew.set_xlabel('Frame #')

    def update(frame_idx: int):
        # Update the axis limits
        ax_curvature.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_ew.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])

        # Move the markers
        ax_curvature_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_ew_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _generate_annotated_images(
        width: int,
        height: int,
        image_triplet: np.ndarray,
        points_2d: np.ndarray
) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = generate_annotated_images(image_triplet, points_2d)
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    rh = height / images.shape[0]
    rw = width / images.shape[1]
    if images.shape[0] * rw > height:
        images = cv2.resize(images, None, fx=rh, fy=rh)
        new_width = images.shape[1]
        offset = int((width - new_width) / 2)
        panel[:, offset:offset + new_width] = images
    else:
        images = cv2.resize(images, None, fx=rw, fy=rw)
        new_height = images.shape[0]
        offset = int((height - new_height) / 2)
        panel[offset:offset + new_height] = images

    return panel


def generate_manoeuvre_video(
        args: Namespace,
        output_dir: PosixPath = None,
        clip_idx: int = 0
):
    """
    Generate a manoeuvre video showing a rotating 3D trajectory with reconstructed
    worm moving along it and camera images with overlaid 2D midline reprojections.
    """
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial

    # Set frame range - using valid range if set
    if args.use_valid_range and reconstruction.start_frame_valid is not None:
        r_start_frame = reconstruction.start_frame_valid
        r_end_frame = reconstruction.end_frame_valid
    else:
        r_start_frame = reconstruction.start_frame
        r_end_frame = reconstruction.end_frame
    if args.start_frame is None:
        start_frame = r_start_frame
    else:
        start_frame = max(args.start_frame, r_start_frame)
    if args.end_frame is None:
        end_frame = r_end_frame
    else:
        end_frame = min(args.end_frame, r_end_frame)

    # Fetch trajectory and postures
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': start_frame,
        'end_frame': end_frame,
    }
    X, _ = get_trajectory(**common_args, smoothing_window=args.smoothing_window_postures)
    Xc = X - X.mean(axis=0)

    # Eigenworm projections
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=reconstruction.id,
        n_components=args.n_components,
        regenerate=False
    )
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=args.smoothing_window_components)
    X_ew = ew.transform(Z)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        points_3d = ts.get('points')
        if args.smoothing_window_postures is not None and args.smoothing_window_postures > 1:
            points_3d = smooth_trajectory(points_3d, args.smoothing_window_postures)
        points_3d_base = ts.get('points_3d_base')
        points_2d_base = ts.get('points_2d_base')
        cam_coeffs = np.concatenate([
            ts.get(f'cam_{k}')
            for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
        ], axis=2)
        prs = ProjectRenderScoreModel(image_size=trial.crop_size)

    X_com = Xc.mean(axis=1)
    speeds = calculate_speeds(Xc, signed=True) * trial.fps
    e0 = normalise(np.gradient(X_com, axis=0))
    e0[speeds < 0] *= -1
    curvature_postures = np.abs(Z)

    # Build plots
    fig_posture, update_posture_plot = _make_3d_posture_plot(
        width=int(args.width / 2),
        height=int(args.height / 2),
        trial=trial,
        X=Xc - X_com[:, None],
        args=args
    )
    fig_traces, update_traces_plot = _make_traces_plots(
        width=int(args.width) / 2,
        height=int(args.height / 2),
        reconstruction=reconstruction,
        X=Xc,
        X_ew=X_ew,
        curvature=curvature_postures,
        args=args
    )

    # Fetch the frames
    logger.info('Querying database for frames.')
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'frame_num': {
                '$gte': start_frame,
                '$lte': end_frame,
            }
        }},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
        }},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Initialise ffmpeg process
    if output_dir is None:
        output_dir = LOGS_PATH / f'trial_{trial.id:03d}'
        output_path = output_dir / f'{START_TIMESTAMP}_trial={trial.id}_r={reconstruction.id}_f={start_frame}-{end_frame}'
    else:
        output_path = output_dir / f'{clip_idx:03d}_trial={trial.id}_r={reconstruction.id}_f={start_frame}-{end_frame}'
    os.makedirs(output_dir, exist_ok=True)

    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': args.fps,
        'metadata:g:0': f'title=Trial {trial.id}. Reconstruction {reconstruction.id}.',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{args.width}x{args.height}')
            .output(str(output_path) + '.mp4', **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    frame_nums = []
    logger.info('Rendering frames.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i > 0 and i % 100 == 0:
            logger.info(f'Rendering frame {n} - {i + 1}/{end_frame - start_frame + 1}.')

        # Check we don't miss any frames
        if i == 0:
            assert n == start_frame
            n0 = n
        assert n == n0 + i

        # Check images are present
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{n:06d}.npz'
        try:
            image_triplet = np.load(img_path)['images']
            if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
                logger.warning('Prepared images are the wrong size, regeneration needed!')
                raise RuntimeError()
        except Exception:
            logger.warning('Prepared images not available, stopping here.')
            break
        frame_nums.append(n)

        # Generate the annotated images
        if reconstruction.source == M3D_SOURCE_MF:
            points_2d = prs._project_to_2d(
                cam_coeffs=torch.from_numpy(cam_coeffs[n][None, ...]),
                points_3d=torch.from_numpy(points_3d[n][None, ...]),
                points_3d_base=torch.from_numpy(points_3d_base[n][None, ...].astype(np.float32)),
                points_2d_base=torch.from_numpy(points_2d_base[n][None, ...].astype(np.float32)),
            )
            points_2d = points_2d[0].numpy().transpose(1, 0, 2)
            points_2d = np.round(points_2d).astype(np.int32)

        else:
            m3d = Midline3D.objects.get(
                frame=res['_id'],
                source=reconstruction.source,
                source_file=reconstruction.source_file,
            )

            # Get 2D projections
            points_2d = np.round(m3d.prepare_2d_coordinates(X=X[i])).astype(np.int32)
            points_2d = points_2d.transpose(1, 0, 2)

        # Prepare images
        images = _generate_annotated_images(
            width=int(args.width),
            height=int(args.height / 2),
            image_triplet=image_triplet,
            points_2d=points_2d
        )

        # Update the plots and extract rendered image
        update_posture_plot(i)
        update_traces_plot(i)
        plot_posture = np.asarray(fig_posture.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        # Join plots and images and write to stream
        frame = np.concatenate([
            images,
            np.concatenate([plot_posture, plot_traces], axis=1),
        ], axis=0)
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    # Write meta data
    meta = to_dict(args)
    meta['created'] = START_TIMESTAMP
    with open(output_path.with_suffix('.meta'), 'w') as f:
        json.dump(meta, f, indent=2, separators=(',', ': '))

    logger.info(f'Generated video for frames {frame_nums[0]}-{frame_nums[-1]} ({frame_nums[-1] - frame_nums[0]}). '
                f'Total frames in reconstruction = {reconstruction.n_frames}.')


def generate_from_spec(args: Namespace):
    """
    Generate a set of clips from a specification file.
    """
    spec_dir = LOGS_PATH / args.spec
    with open(spec_dir / 'spec.yml') as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    # Copy the spec to the output dir
    output_dir = spec_dir / START_TIMESTAMP
    shutil.copy(spec_dir / 'spec.yml', output_dir / 'spec.yml')

    # Load arguments
    for k, v in spec['args'].items():
        assert hasattr(args, k)
        setattr(args, k, v)

    # Generate clips
    clips = spec['clips']
    for i, clip in enumerate(clips):
        logger.info(f'Generating clip {i + 1}/{len(clips)}.')
        args.reconstruction = clip['reconstruction']
        args.start_frame = clip['start']
        args.end_frame = clip['end']
        generate_manoeuvre_video(
            args,
            output_dir=output_dir / 'clips',
            clip_idx=i
        )


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    args_ = get_args()

    if args_.spec is None:
        generate_manoeuvre_video(args_)
    else:
        generate_from_spec(args_)
