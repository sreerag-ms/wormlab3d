import os
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import PosixPath
from typing import Dict, Any, Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from mayavi import mlab

from wormlab3d import logger, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.plot_utils import overlay_image, make_box_outline, to_rgb
from wormlab3d.toolkit.util import print_args, to_dict
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import smooth_trajectory

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate trajectory videos.')

    # Target
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--fps', type=int, default=25, help='Video output framerate.')
    parser.add_argument('--clip-duration', type=int, default=30, help='Duration of each clip.')
    parser.add_argument('--speed', type=int, default=2, help='Speedup factor.')

    # Trajectory
    parser.add_argument('--trajectory-point', type=int, default=-1, help='Trajectory point.')
    parser.add_argument('--smoothing-window-trajectories', type=int, default=0,
                        help='Smoothing window for the trajectories.')
    parser.add_argument('--smoothing-window-postures', type=int, default=0, help='Smoothing window for the postures.')
    parser.add_argument('--smoothing-window-components', type=int, default=0,
                        help='Smoothing window for the components.')

    # 3D plot
    parser.add_argument('--revolution-rate', type=float, default=1 / 3,
                        help='Rate of 3D plot revolution in revolutions/minute.')
    parser.add_argument('--distance', type=float, default=10.,
                        help='Camera distance in worm lengths.')
    parser.add_argument('--outline-mode', type=str, default='xyz', choices=['xyz', 'pca', 'none'],
                        help='Add outline box in the main axis mode (xyz, default), by PCA components (pca) or none.')

    args = parser.parse_args()
    assert args.spec is not None, 'This script requires setting --spec=path.'

    return args


def _make_info_panel(
        width: int,
        height: int,
        caption: str,
        speed: int,
        reconstruction: Reconstruction,
) -> Tuple[Figure, Callable]:
    """
    Info panel.
    """
    logger.info('Building infos plot.')
    trial = reconstruction.trial
    if caption != '':
        caption += '\n'
    r_end_frame = reconstruction.end_frame_valid

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')

    def get_details(frame_num: int) -> str:
        curr_time = datetime.fromtimestamp(np.floor(frame_num / trial.fps))
        total_time = datetime.fromtimestamp(np.floor(r_end_frame / trial.fps))
        # return f'{curr_time:%M:%S}/{total_time:%M:%S}'
        return caption + \
               f'Concentration: {trial.experiment.concentration}%\n' \
               f'Time: {curr_time:%M:%S}/{total_time:%M:%S}\n' \
               f'Speed: {speed}x'

    # Details
    text = fig.text(0.05, 0.95, get_details(0), ha='left', va='top', fontsize=20, linespacing=1.5)

    def update(frame_num: int):
        # Update the text
        text.set_text(get_details(frame_num))

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
        curvatures: np.ndarray,
        lengths: np.ndarray,
        distance: float,
        azim_offset: int,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D worm plot.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D plot.')
    distance = lengths.mean() * distance
    T = len(X_postures)

    # Construct colours
    s = np.linspace(0, 1, T)
    cmap = plt.get_cmap('viridis_r')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Set up mlab figure
    fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Render the trajectory with simple lines
    path = mlab.plot3d(*X_trajectory.T, s, opacity=0.2, tube_radius=0.015)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    if args.outline_mode == 'xyz':
        mlab.outline(path, color=(0, 0, 0), figure=fig)
    elif args.outline_mode == 'pca':
        # Add outline box aligned with PCA components
        lines = make_box_outline(X=X_trajectory, use_extents=True)
        for l in lines:
            mlab.plot3d(
                *l.T,
                figure=fig,
                color=to_rgb('darkgrey'),
                tube_radius=0.001,
            )

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
        # midline_opts={'opacity': 1, 'line_width': 8},
        mesh_opts={'opacity': 1},
        surface_opts={'radius': 0.024 * lengths.mean()}
    )
    # fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvatures.max(), v_max=curvatures.max())

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


def prepare_panels(
        args: Namespace,
        reconstruction: Reconstruction,
        frame_num: int,
        caption: str,
        distance: float,
        azim_offset: int,
        speed: int,
):
    """
    Prepare the panel of plots for the given reconstruction.
    """
    logger.info(f'Preparing panels for reconstruction {reconstruction.id}.')
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions supported!'
    trial = reconstruction.trial
    r_start_frame = reconstruction.start_frame_valid
    r_end_frame = reconstruction.end_frame_valid

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)

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

    # Get natural frame representation for curvatures
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=args.smoothing_window_components)
    curvatures = np.abs(Z)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    ts = TrialState(reconstruction)
    lengths = ts.get('length', r_start_frame, r_end_frame + 1)[:, 0]

    # Build plots
    fig_info, update_infos = _make_info_panel(
        width=args.width,
        height=args.height,
        caption=caption,
        speed=speed,
        reconstruction=reconstruction,
    )

    fig_3d, update_3d_plot = _make_3d_plot(
        width=args.width,
        height=args.height,
        trial=trial,
        X_postures=Xp,
        X_trajectory=Xt,
        curvatures=curvatures,
        lengths=lengths,
        distance=distance,
        azim_offset=azim_offset,
        args=args,
    )

    # Figure out which frames to display
    n_frames = int(args.clip_duration * speed * trial.fps)
    start_frame = max(r_start_frame, int(frame_num - n_frames / 2))
    end_frame = min(r_end_frame, start_frame + n_frames)
    frame_nums = np.arange(start_frame, end_frame + 1)

    def prepare_frame(i: int) -> np.ndarray:
        # Update the plots and extract renders
        n = start_frame - r_start_frame + i
        update_3d_plot(n)
        update_infos(n)
        plot_3d = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_3d)
        plot_3d = cv2.resize(plot_3d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        plot_info = np.asarray(fig_info.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        # Overlay plot info on top of 3d panel
        plot_3d = overlay_image(plot_3d, plot_info, x_offset=0, y_offset=0)
        frame = plot_3d

        return frame

    return prepare_frame, frame_nums


def generate_clip(
        args: Namespace,
        clip: Dict[str, Any],
        output_dir: PosixPath,
        clip_idx: int
):
    """
    Generate a trajectory video showing a 3D worm moving along a trajectory.
    """
    reconstruction = Reconstruction.objects.get(id=clip['reconstruction'])
    update_fn, frame_nums = prepare_panels(
        args=args,
        reconstruction=reconstruction,
        frame_num=clip['frame_num'],
        caption=clip['caption'] if 'caption' in clip else '',
        azim_offset=clip['azim_offset'] if 'azim_offset' in clip else 0,
        distance=clip['distance'] if 'distance' in clip else args.distance,
        speed=clip['speed'] if 'speed' in clip else args.speed,
    )

    # Initialise ffmpeg process
    output_path = output_dir / f'{clip_idx:03d}_trial={reconstruction.trial.id}_r={reconstruction.id}_f={frame_nums[0]}-{frame_nums[-1]}_d={args.distance}_s={args.speed}'
    input_args = {
        'format': 'rawvideo',
        'pix_fmt': 'rgb24',
        's': f'{args.width}x{args.height}',
        'r': len(frame_nums) / args.clip_duration,
    }
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': args.fps,
        'metadata:g:0': f'title=Trial {reconstruction.trial.id}. Reconstruction {reconstruction.id}. Frames {frame_nums[0]}-{frame_nums[-1]}',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }
    process = (
        ffmpeg
            .input('pipe:', **input_args)
            .output(str(output_path) + '.mp4', **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    logger.info('Rendering frames.')
    n_frames = len(frame_nums)
    for i in range(n_frames):
        if i > 0 and i % 50 == 0:
            logger.info(f'Rendering frame {i + 1}/{n_frames}.')

        # Update the frame and write to stream
        frame = update_fn(i)
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
        assert 'reconstruction' in clip
        assert 'frame_num' in clip

        logger.info(f'Generating clip {i + 1}/{len(clips)}.')
        generate_clip(
            args,
            clip,
            output_dir=output_dir,
            clip_idx=i
        )
        plt.close('all')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_from_spec()
