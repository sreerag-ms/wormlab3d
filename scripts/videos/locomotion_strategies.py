import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath
from typing import Dict, Union
from typing import Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from mayavi import mlab

from wormlab3d import PREPARED_IMAGES_PATH
from wormlab3d import logger, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Frame, Reconstruction, Trial, Midline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.particles.tumble_run import calculate_curvature
from wormlab3d.postures.helicities import calculate_helicities, plot_helicities
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.plot_utils import overlay_image
from wormlab3d.toolkit.util import normalise, print_args, str2bool, to_dict
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate a locomotion strategies video.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')
    parser.add_argument('--buffer-window', type=int, help='Number of frames to load either side of the clip.')

    # Spec
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')

    # Trajectory
    parser.add_argument('--trajectory-point', type=int, default=-1, help='Trajectory point.')
    parser.add_argument('--smoothing-window-trajectories', type=int, default=0,
                        help='Smoothing window for the trajectories.')
    parser.add_argument('--smoothing-window-postures', type=int, default=0, help='Smoothing window for the postures.')
    parser.add_argument('--smoothing-window-components', type=int, default=0,
                        help='Smoothing window for the components.')
    parser.add_argument('--smoothing-window-speed', type=int, default=25,
                        help='Smoothing window for the speed calculation.')

    # 3D plot
    parser.add_argument('--trajectory-colouring', type=str, choices=['time', 'speed', 'curvature'], default='time',
                        help='Colour the 3D trajectory by time, speed or curvature.')
    parser.add_argument('--show-trajectory-colourbar', type=str2bool, default=False,
                        help='Show colourbar on the 3D plot.')
    parser.add_argument('--show-3d-axis', type=str2bool, default=True,
                        help='Show axis on the 3D plot.')
    parser.add_argument('--revolution-rate', type=float, default=1 / 3,
                        help='Rate of 3D plot revolution in revolutions/minute.')

    # Traces
    parser.add_argument('--time-range-traces', type=float, default=5,
                        help='Time range to show on trace plots in seconds.')
    parser.add_argument('--rebuild-planarity-cache', type=str2bool, default=False, help='Rebuild the planarity caches.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')
    parser.add_argument('--n-helicity-fade-lines', type=int, default=100,
                        help='Filled region fade resolution in helicity plot.')
    parser.add_argument('--speed-max', type=float, help='Maximum value for the speed axis.')
    parser.add_argument('--speed-min', type=float, help='Minimum value for the speed axis.')
    parser.add_argument('--nonp-max', type=float, help='Maximum value for the non-planarity axis.')
    parser.add_argument('--nonp-min', type=float, help='Minimum value for the non-planarity axis.')
    parser.add_argument('--hel-max', type=float, help='Maximum value for the helicity axis.')
    parser.add_argument('--hel-min', type=float, help='Minimum value for the helicity axis.')

    args = parser.parse_args()
    assert args.spec is not None, 'This script requires setting --spec=path.'

    return args


def _make_info_panel(
        width: int,
        height: int,
        trial: Trial,
) -> Tuple[Figure, Callable]:
    """
    Info panel.
    """
    logger.info('Building infos plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')

    def get_details(frame_idx) -> str:
        return f'Trial #{trial.id}\n' \
               f'{trial.experiment.concentration}% Gelatine'

    # Details
    text = fig.text(0.05, 0.95, get_details(0), ha='left', va='top',
                    fontsize=16, linespacing=1.5, fontweight='bold')

    def update(frame_idx: int):
        # Update the text
        text.set_text(get_details(frame_idx))

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def _make_3d_plot(
        width: int,
        height: int,
        trial: Trial,
        X_postures: np.ndarray,
        X_trajectory: np.ndarray,
        speeds: np.ndarray,
        curvature: np.ndarray,
        curvature_postures: np.ndarray,
        lengths: np.ndarray,
        azim_offset: int,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D trajectory plot with worm using mayavi.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D plot.')
    distance = lengths.max() * 2
    T = len(X_postures)

    # Construct colours
    if args.trajectory_colouring == 'time':
        s = np.linspace(0, 1, T)
        cmap = plt.get_cmap('viridis_r')
    elif args.trajectory_colouring == 'speed':
        s = speeds
        cmap = plt.get_cmap('PRGn')
    elif args.trajectory_colouring == 'curvature':
        s = curvature
        cmap = plt.get_cmap('Reds')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Set up mlab figure
    fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))

    # Doesn't really seem to make any difference
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Render the trajectory with simple lines
    path = mlab.plot3d(*X_trajectory.T, s, opacity=0.4, tube_radius=None, line_width=8)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Smooth the midpoints for nicer camera tracking
    mps = np.zeros((T, 3))
    for i, X in enumerate(X_postures):
        mps[i] = X.min(axis=0) + np.ptp(X, axis=0) / 2
    mps = smooth_trajectory(mps, window_len=51)

    # Set up the artist and add the pieces
    NF = NaturalFrame(X_postures[0])
    fa = FrameArtistMLab(
        NF,
        use_centred_midline=False,
        midline_opts={'opacity': 1, 'line_width': 8},
        surface_opts={'radius': 0.024 * lengths.mean()}
    )
    fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvature_postures.max(), v_max=curvature_postures.max())
    fa.add_outline(fig)

    # Aspects
    n_revolutions = T / trial.fps / 60 * args.revolution_rate
    azims = azim_offset + np.linspace(start=0, stop=360 * n_revolutions, num=T)
    mlab.view(figure=fig, azimuth=azims[0], distance=distance, focalpoint=mps[0])

    def update(frame_idx: int):
        fig.scene.disable_render = True
        NF = NaturalFrame(X_postures[frame_idx])
        fa.update(NF)
        fig.scene.disable_render = False
        mlab.view(figure=fig, azimuth=azims[frame_idx], distance=distance, focalpoint=mps[frame_idx])
        fig.scene.render()

    return fig, update


def _make_traces_plots(
        width: int,
        height: int,
        reconstruction: Reconstruction,
        start_frame: int,
        speeds: np.ndarray,
        nonp_postures: np.ndarray,
        helicities: np.ndarray,
        curvature: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a traces plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building traces plot.')
    N = len(speeds)
    trial = reconstruction.trial
    if args.x_label == 'time':
        ts = (start_frame + np.linspace(0, N, N)) / trial.fps
        t_range = args.time_range_traces
    else:
        ts = start_frame + np.arange(N)
        t_range = args.time_range_traces * trial.fps

    # Plot
    fig, axes = plt.subplots(2, figsize=(width / 100, height / 100), gridspec_kw={
        'hspace': 0,
        'top': 0.98,
        'bottom': 0.15,
        'left': 0.11,
        'right': 0.88,
    })

    # Speeds
    ax_speed = axes[0]
    ax_speed.axhline(y=0, color='darkgrey')
    ax_speed.plot(ts, speeds)
    ax_speed.set_ylabel('Speed (mm/s)')
    if args.speed_min is not None:
        ax_speed.set_ylim(bottom=args.speed_min)
    if args.speed_max is not None:
        ax_speed.set_ylim(top=args.speed_max)
    ax_speed.set_xticklabels([])
    ax_speed.spines['bottom'].set_visible(False)
    ax_speed_marker = ax_speed.axvline(x=0, color='red')

    # Non-planarity of postures
    ax_nonp = ax_speed.twinx()
    ax_nonp.plot(ts, nonp_postures, color='orange', alpha=0.6, linestyle='--')
    ax_nonp.set_ylabel('Non-planarity', rotation=270, labelpad=10)
    if args.nonp_min is not None:
        ax_nonp.set_ylim(bottom=args.nonp_min)
    if args.speed_max is not None:
        ax_nonp.set_ylim(top=args.nonp_max)
    ax_nonp.set_xticklabels([])
    ax_nonp.axhline(y=0, color='darkgrey')

    # Helicity
    ax_hel = ax_speed.twinx()
    plot_helicities(
        ax=ax_hel,
        helicities=helicities,
        xs=ts,
        n_fade_lines=args.n_helicity_fade_lines
    )
    if args.hel_min is not None:
        ax_hel.set_ylim(bottom=args.hel_min)
    if args.hel_max is not None:
        ax_hel.set_ylim(top=args.hel_max)
    label_args = dict(transform=ax_hel.transAxes, horizontalalignment='right', fontweight='bold', fontsize=16,
                      fontfamily='Symbol')
    ax_hel.text(0.98, 0.94, '↻', verticalalignment='top', color='purple', **label_args)
    ax_hel.text(0.98, 0.05, '↺', verticalalignment='bottom', color='green', **label_args)
    ax_hel.set_yticks([])
    ax_hel.set_yticklabels([])
    ax_hel.set_xticklabels([])
    ax_hel.spines['top'].set_visible(False)
    ax_hel.spines['bottom'].set_visible(False)

    # Curvature
    ax_curvature = axes[1]
    im = ax_curvature.imshow(curvature.T, aspect='auto', cmap='Reds', origin='lower', extent=(ts[0], ts[-1], 0, 1))
    cax = ax_curvature.inset_axes([1.03, 0.1, 0.04, 0.8], transform=ax_curvature.transAxes)
    fig.colorbar(im, ax=ax_curvature, cax=cax)
    ht_args = dict(transform=ax_curvature.transAxes, horizontalalignment='right', fontweight='bold')
    ax_curvature.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax_curvature.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax_curvature.set_ylabel('Curvature', labelpad=10)
    ax_curvature.set_yticks([0, 1])
    ax_curvature.set_yticklabels([])
    ax_curvature.xaxis.get_major_locator().set_params(integer=True)
    ax_curvature.spines['top'].set_visible(False)
    ax_curvature_marker = ax_curvature.axvline(x=0, color='red')

    if args.x_label == 'time':
        ax_curvature.set_xlabel('Time (s)')
    else:
        ax_curvature.set_xlabel('Frame #')

    def update(frame_idx: int):
        # Update the axis limits
        ax_speed.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_curvature.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])

        # Move the markers
        ax_speed_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_curvature_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])

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
    panel = np.ones((height, width, 3), dtype=np.uint8) * 155
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


def prepare_reconstruction_panel(
        args: Namespace,
        reconstruction: Reconstruction,
        start_frame: int,
        end_frame: int,
        width: int,
        height: int,
        azim_offset: int,
):
    """
    Prepare the panel of plots for the given reconstruction.
    """
    logger.info(f'Preparing panel for reconstruction {reconstruction.id}.')
    trial = reconstruction.trial
    if args.buffer_window is not None:
        r_start_frame = max(reconstruction.start_frame_valid, start_frame - args.buffer_window)
        r_end_frame = min(reconstruction.end_frame_valid, end_frame + args.buffer_window)
    else:
        r_start_frame = reconstruction.start_frame_valid
        r_end_frame = reconstruction.end_frame_valid

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)
    Xrc = Xr - Xr.mean(axis=0)
    Xr_com = Xr.mean(axis=1)

    # Pick trajectory point and smooth
    if args.trajectory_point == -1:
        Xt = Xr.mean(axis=1)
    else:
        N = Xr.shape[1]
        u = round(args.trajectory_point * N)
        if u == N:
            u -= 1
        assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
        logger.info(f'Using trajectory point {u}/{N}.')
        Xt = Xr[:, u]
    if args.smoothing_window_trajectories > 1:
        Xt = smooth_trajectory(Xt, window_len=args.smoothing_window_trajectories)

    # Smooth the postures and centre
    if args.smoothing_window_postures > 1:
        Xp = smooth_trajectory(Xr, window_len=args.smoothing_window_postures)
    else:
        Xp = Xr
    Xpc = Xp - Xp.mean(axis=1, keepdims=True)

    # Get natural frame representation
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=args.smoothing_window_components)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        points_3d = ts.get('points')
        if args.smoothing_window_postures > 1:
            points_3d = smooth_trajectory(points_3d, args.smoothing_window_postures)
        points_3d_base = ts.get('points_3d_base')
        points_2d_base = ts.get('points_2d_base')
        lengths = ts.get('length', r_start_frame, r_end_frame + 1)[:, 0]
        cam_coeffs = np.concatenate([
            ts.get(f'cam_{k}')
            for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
        ], axis=2)
        prs = ProjectRenderScoreModel(image_size=trial.crop_size)
    else:
        lengths = np.linalg.norm(Xpc[:, 1:] - Xpc[:, :-1], axis=-1).sum(axis=-1)

    # Calculate speed
    if args.smoothing_window_speed > 1:
        Xc_smoothed = smooth_trajectory(Xrc, window_len=args.smoothing_window_speed)
        speeds = calculate_speeds(Xc_smoothed, signed=True) * trial.fps
    else:
        speeds = calculate_speeds(Xrc, signed=True) * trial.fps

    # Calculate non-planarity of the postures
    logger.info('Fetching posture planarities.')
    pcas, meta = generate_or_load_pca_cache(
        **common_args,
        window_size=1,
        smoothing_window=args.smoothing_window_postures,
        rebuild_cache=args.rebuild_planarity_cache
    )
    nonp_postures = pcas.nonp

    # Calculate curvatures of postures and trajectory
    e0 = normalise(np.gradient(Xr_com, axis=0))
    e0[speeds < 0] *= -1
    curvature_traj = calculate_curvature(e0)
    curvature_postures = np.abs(Z)

    # Calculate posture helicities
    helicities = calculate_helicities(Xpc)

    # Build plots
    fig_info, update_info_plot = _make_info_panel(
        width=width,
        height=int(height / 2),
        trial=trial,
    )
    fig_3d, update_3d_plot = _make_3d_plot(
        width=width,
        height=int(height / 2),
        trial=trial,
        X_trajectory=Xt,
        X_postures=Xp,
        speeds=speeds,
        curvature=curvature_traj,
        curvature_postures=curvature_postures,
        lengths=lengths,
        azim_offset=azim_offset,
        args=args,
    )
    fig_traces, update_traces_plot = _make_traces_plots(
        width=width,
        height=int(height / 10 * 3),
        reconstruction=reconstruction,
        start_frame=r_start_frame,
        speeds=speeds,
        nonp_postures=nonp_postures,
        helicities=helicities,
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

    frame_nums = []
    for i, res in enumerate(cursor):
        n = res['frame_num']

        # Check we don't miss any frames
        if i == 0:
            assert n == start_frame
            n0 = n
        assert n == n0 + i

        frame_nums.append(n)

    def prepare_frame(i: int) -> np.ndarray:
        n = frame_nums[i]
        # Check images are present
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{n:06d}.npz'
        image_triplet = np.load(img_path)['images']
        if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
            raise RuntimeError('Prepared images are the wrong size, regeneration needed!')

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
            points_2d = np.round(m3d.prepare_2d_coordinates(X=Xr[i])).astype(np.int32)
            points_2d = points_2d.transpose(1, 0, 2)

        # Prepare images
        images = _generate_annotated_images(
            width=width,
            height=int(height / 5),
            image_triplet=image_triplet,
            points_2d=points_2d
        )

        # Update the plots and extract renders
        idx = start_frame - r_start_frame + i
        update_info_plot(idx)
        update_3d_plot(idx)
        update_traces_plot(idx)

        plot_info = np.asarray(fig_info.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_3d = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_3d)
        plot_3d = cv2.resize(plot_3d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Overlay plot info on top of 3d panel
        plot_3d = overlay_image(plot_3d, plot_info, x_offset=0, y_offset=0)

        # Join plots and images and write to stream
        frame = np.concatenate([plot_3d, images, plot_traces], axis=0)

        return frame

    return frame_nums, prepare_frame


def generate_locomotion_strategies_video(
        args: Namespace,
        panels: Dict[str, Union[str, int]],
        output_dir: PosixPath,
        clip_idx: int
):
    """
    Generate a locomotion strategies video showing a rotating 3D trajectory with reconstructed
    worm moving along it and camera images with overlaid 2D midline reprojections.
    """

    # Prepare each panel to be displayed in columns
    reconstructions = []
    frames = []
    fns = []
    for panel in panels:
        reconstruction = Reconstruction.objects.get(id=panel['reconstruction'])
        frame_nums, fn = prepare_reconstruction_panel(
            args=args,
            reconstruction=reconstruction,
            start_frame=panel['start'],
            end_frame=panel['end'],
            width=int(args.width / len(panels)),
            height=args.height,
            azim_offset=panel['azim_offset'] if 'azim_offset' in panel else 0
        )
        reconstructions.append(reconstruction)
        frames.append(frame_nums)
        fns.append(fn)

    # Each panel must run for the same number of frames
    n_frames = len(frames[0])
    assert all([len(frame_nums) == n_frames for frame_nums in frames]), \
        'Number of frames must be the same for all clips!'

    # Initialise ffmpeg process
    trials_and_ranges_str = ','.join([f'{r.trial.id:03d}_({f[0]}-{f[-1]})' for r, f in zip(reconstructions, frames)])
    reconstructions_str = ','.join([str(r.id) for r in reconstructions])
    output_path = output_dir / f'{clip_idx:03d}_trials={trials_and_ranges_str}_r={reconstructions_str}'
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': args.fps,
        'metadata:g:0': f'title=Trials {trials_and_ranges_str}. Reconstructions {reconstructions_str}.',
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

    logger.info('Rendering frames.')
    for i in range(n_frames):
        if i > 0 and i % 50 == 0:
            logger.info(f'Rendering frame {i + 1}/{n_frames}.')

        # Build panels for each reconstruction
        panels = []
        for j in range(len(reconstructions)):
            panels.append(fns[j](i))

        # Join panels with 1px dividing lines and write to stream
        frame = np.concatenate(panels, axis=1)
        frame[:, [int(j * (args.width / len(panels))) for j in range(1, len(panels))]] = 0
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    logger.info(f'Generated video.')


def generate_from_spec():
    """
    Generate a set of clips from a specification file.
    """
    args = get_args()
    spec_dir = LOGS_PATH / args.spec
    with open(spec_dir / 'spec.yml') as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    # Load arguments
    for k, v in spec['args'].items():
        assert hasattr(args, k), f'{k} is not a valid argument!'
        setattr(args, k, v)
    print_args(args)

    # Copy the spec with final args to the output dir
    output_dir = spec_dir / START_TIMESTAMP
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / 'spec.yml', 'w') as f:
        spec['created'] = START_TIMESTAMP
        spec['args'] = to_dict(args)
        yaml.dump(spec, f)

    # Generate clips
    clips = spec['clips']
    for i, clip in enumerate(clips):
        logger.info(f'Generating clip {i + 1}/{len(clips)}.')
        generate_locomotion_strategies_video(
            args,
            panels=clip['panels'],
            output_dir=output_dir,
            clip_idx=i
        )
        plt.close('all')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_from_spec()
