import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath
from typing import Dict, Any, Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from mayavi import mlab
from scipy.signal import find_peaks

from wormlab3d import PREPARED_IMAGES_PATH
from wormlab3d import logger, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Reconstruction, Frame, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.mf_render_wrapper import RenderWrapper
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.plot_utils import overlay_image, make_box_from_pca_mlab
from wormlab3d.toolkit.util import print_args, str2bool, to_dict
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

# Off-screen rendering
mlab.options.offscreen = True

plt.rcParams['font.family'] = 'Helvetica'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate a basic exemplar video.')

    # Target
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')
    parser.add_argument('--buffer-window', type=int, help='Number of frames to load either side of the clip.')

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
    parser.add_argument('--manoeuvres-mode', type=str2bool, default=False,
                        help='Render in manoeuvres mode.')
    parser.add_argument('--revolution-rate', type=float, default=1 / 3,
                        help='Rate of 3D plot revolution in revolutions/minute.')
    parser.add_argument('--distance', type=float, default=2.,
                        help='Camera distance in worm lengths.')

    # Renders/midline
    parser.add_argument('--overlay-midlines', type=str2bool, default=True,
                        help='Add midlines to the rendered images.')
    parser.add_argument('--posture-outline', type=str2bool, default=False,
                        help='Add outline box around the postures.')
    parser.add_argument('--show-renders', type=str2bool, default=False,
                        help='Show renders alongside the images.')
    parser.add_argument('--renders-white-at', type=float, default=0.1,
                        help='What pixel intensity below which to set to zero for the renders.')

    args = parser.parse_args()
    assert args.spec is not None, 'This script requires setting --spec=path.'

    return args


def _make_info_panel(
        width: int,
        height: int,
        caption: str
) -> Figure:
    """
    Info panel.
    """
    logger.info('Building infos plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.text(0.05, 0.95, caption, ha='left', va='top',
             fontsize=18, linespacing=1.5, fontweight='bold')
    fig.canvas.draw()
    fig.tight_layout()

    return fig


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
    fa.add_surface(fig, v_min=-curvatures.max(), v_max=curvatures.max())
    if args.posture_outline:
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


def _make_3d_manoeuvres_plot(
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
        manoeuvre_spec: dict,
        speeds: np.ndarray,
        r_start_frame: int,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D trajectory plot with worm using mayavi.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D plot.')
    distance = lengths.max() * distance
    T = len(X_postures)

    inc_start_frame = manoeuvre_spec['inc_start']
    inc_end_frame = manoeuvre_spec['inc_end']
    man_start_frame = manoeuvre_spec['man_start']
    man_end_frame = manoeuvre_spec['man_end']
    out_start_frame = manoeuvre_spec['out_start']
    out_end_frame = manoeuvre_spec['out_end']

    # Guess missing ranges
    if man_start_frame is None:
        rev_peaks, rev_props = find_peaks(speeds < 0, width=5)
        assert len(rev_peaks) > 0, 'No reversals found in clip!'
        rev_idx = rev_props['widths'].argmax()
        man_start_idx = rev_props['left_bases'][rev_idx] + 10
        man_end_idx = rev_props['right_bases'][rev_idx] - 10
    else:
        man_start_idx = man_start_frame - r_start_frame
        man_end_idx = man_end_frame - r_start_frame
    if inc_start_frame is None:
        inc_end_idx = man_start_idx - 20
        inc_start_idx = max(0, inc_end_idx - 250)
    else:
        inc_start_idx = max(0, inc_start_frame - r_start_frame)
        inc_end_idx = inc_end_frame - r_start_frame
    if out_start_frame is None:
        out_start_idx = man_end_idx + 20
        out_end_idx = min(len(X_trajectory), out_start_idx + 250)
    else:
        out_start_idx = out_start_frame - r_start_frame
        out_end_idx = min(len(X_trajectory), out_end_frame - r_start_frame)

    # Construct colours
    s = speeds
    cmap = plt.get_cmap('PRGn')
    vmax = np.abs(s).max()
    vmin = -vmax
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
    path = mlab.plot3d(*X_trajectory.T, s, vmax=vmax, vmin=vmin, opacity=0.8, tube_radius=None, line_width=9)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Add the cuboids
    cuboid_args = dict(
        dimensions='extents',
        opacity=0.2,
        draw_outline=True,
        outline_opacity=0.8,
        outline_tube_radius=0.001,
        fig=fig
    )
    make_box_from_pca_mlab(
        X=X_trajectory[inc_start_idx:inc_end_idx],
        colour='aqua',
        outline_colour='teal',
        **cuboid_args
    )
    make_box_from_pca_mlab(
        X=X_trajectory[man_start_idx:man_end_idx],
        colour='plum',
        outline_colour='hotpink',
        **cuboid_args
    )
    make_box_from_pca_mlab(
        X=X_trajectory[out_start_idx:out_end_idx],
        colour='aqua',
        outline_colour='teal',
        **cuboid_args
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
        midline_opts={'opacity': 1, 'line_width': 8},
        surface_opts={'radius': 0.024 * lengths.mean()}
    )
    fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvatures.max(), v_max=curvatures.max())
    # fa.add_outline(fig)

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


def _resize_image_panel(
        width: int,
        height: int,
        images: np.ndarray
) -> np.ndarray:
    """
    Resize a set of images or renders.
    """
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


def _generate_annotated_images(
        width: int,
        height: int,
        image_triplet: np.ndarray,
        points_2d: np.ndarray,
        overlay_midlines: bool
) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    if overlay_midlines:
        images = generate_annotated_images(image_triplet, points_2d)
        images = images.transpose(1, 0, 2)
    else:
        images = []
        for img in image_triplet:
            z = ((1 - img) * 255).astype(np.uint8)
            z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
            images.append(z)
        images = np.fliplr(np.concatenate(images))

    panel = _resize_image_panel(width, height, images)
    return panel


def _get_reds(
        white_at: float = 0.1,
) -> np.ndarray:
    """
    Get the alpha-blended red colours.
    """
    cm = plt.get_cmap('Reds')
    reds = cm(np.linspace(0, 1, 256))[..., :3]
    reds[:int(255 * white_at)] = (1, 1, 1)
    reds = (reds * 255).astype(np.uint8)
    return reds


def _make_renders(
        width: int,
        height: int,
        renderer: RenderWrapper,
        white_at: float = 0.1,
) -> Callable:
    """
    Prepare rendered images.
    """
    reds = _get_reds(white_at)

    def update(frame_num: int):
        renderer.frame = renderer.ts.trial.get_frame(frame_num)
        renderer.points_2d = torch.from_numpy(renderer.ts.get('points_2d', frame_num, frame_num + 1).copy())
        renderer.init_params()
        masks, blobs = renderer.get_masks_and_blobs()
        masks = (masks * 255).astype(np.uint8)
        renders = np.take(reds, masks, axis=0)
        renders = np.fliplr(np.concatenate(renders, axis=0))
        renders = _resize_image_panel(width, height, renders)

        return renders

    return update


def prepare_panels(
        args: Namespace,
        reconstruction: Reconstruction,
        start_frame: int,
        end_frame: int,
        caption: str,
        distance: float,
        azim_offset: int,
        manoeuvre_spec: dict,
):
    """
    Prepare the panel of plots for the given reconstruction.
    """
    logger.info(f'Preparing panels for reconstruction {reconstruction.id}.')
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions supported!'
    trial = reconstruction.trial

    if args.manoeuvres_mode:
        start_frame = manoeuvre_spec['anim_start']
        end_frame = manoeuvre_spec['anim_end']
        if manoeuvre_spec['traj_start'] is None:
            manoeuvre_spec['traj_start'] = start_frame - 500
        if manoeuvre_spec['traj_end'] is None:
            manoeuvre_spec['traj_end'] = end_frame + 500
        r_start_frame = max(reconstruction.start_frame_valid, manoeuvre_spec['traj_start'])
        r_end_frame = min(reconstruction.end_frame_valid, manoeuvre_spec['traj_end'])
    else:
        if args.buffer_window is not None:
            r_start_frame = max(reconstruction.start_frame_valid, start_frame - args.buffer_window)
            r_end_frame = min(reconstruction.end_frame_valid, end_frame + args.buffer_window)
        else:
            r_start_frame = reconstruction.start_frame_valid
            r_end_frame = reconstruction.end_frame_valid

    # Instantiate renderer
    renderer = RenderWrapper(reconstruction, trial.get_frame(r_start_frame))

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)

    # Calculate speed
    if args.smoothing_window_speed > 1:
        Xc_smoothed = smooth_trajectory(Xr, window_len=args.smoothing_window_speed)
        speeds = calculate_speeds(Xc_smoothed, signed=True) * trial.fps
    else:
        speeds = calculate_speeds(Xr, signed=True) * trial.fps

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

    # Build plots
    imgs_width = int(args.height / 3)
    if args.show_renders:
        imgs_width *= 2
        update_renders = _make_renders(
            width=int(args.height / 3),
            height=int(args.height),
            renderer=renderer,
            white_at=args.renders_white_at
        )

    fig_info = _make_info_panel(
        width=int(args.width - imgs_width),
        height=args.height,
        caption=caption
    )
    plot_info = np.asarray(fig_info.canvas.renderer._renderer).take([0, 1, 2], axis=2)

    args_3d = dict(
        width=int(args.width - imgs_width),
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
    if args.manoeuvres_mode:
        fig_3d, update_3d_plot = _make_3d_manoeuvres_plot(
            **args_3d,
            manoeuvre_spec=manoeuvre_spec,
            speeds=speeds,
            r_start_frame=r_start_frame
        )
    else:
        fig_3d, update_3d_plot = _make_3d_plot(**args_3d)

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
        points_2d = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs[n][None, ...]),
            points_3d=torch.from_numpy(points_3d[n][None, ...]),
            points_3d_base=torch.from_numpy(points_3d_base[n][None, ...].astype(np.float32)),
            points_2d_base=torch.from_numpy(points_2d_base[n][None, ...].astype(np.float32)),
        )
        points_2d = points_2d[0].numpy().transpose(1, 0, 2)
        points_2d = np.round(points_2d).astype(np.int32)

        # Prepare images
        images = _generate_annotated_images(
            width=int(args.height / 3),
            height=int(args.height),
            image_triplet=image_triplet,
            points_2d=points_2d,
            overlay_midlines=args.overlay_midlines
        )

        if args.show_renders:
            renders = update_renders(n)
            images[:, -1] = 0
            images = np.concatenate([images, renders], axis=1)

        # Update the plots and extract renders
        update_3d_plot(start_frame - r_start_frame + i)
        plot_3d = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_3d)
        plot_3d = cv2.resize(plot_3d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Overlay plot info on top of 3d panel
        plot_3d = overlay_image(plot_3d, plot_info, x_offset=0, y_offset=0)

        # Join plots and images and write to stream
        images[int(args.height / 3)] = 0
        images[int(args.height / 3 * 2)] = 0
        images[:, -1] = 0
        frame = np.concatenate([images, plot_3d], axis=1)

        return frame

    return prepare_frame, frame_nums


def generate_clip(
        args: Namespace,
        clip: Dict[str, Any],
        output_dir: PosixPath,
        clip_idx: int
):
    """
    Generate a basic exemplar video showing a rotating 3D worm along a trajectory
    and camera images with overlaid 2D midline reprojections.
    """
    reconstruction = Reconstruction.objects.get(id=clip['reconstruction'])

    if args.manoeuvres_mode:
        manoeuvre_spec = clip['manoeuvre_spec']
        for k in ['traj_start', 'traj_end', 'inc_start', 'inc_end',
                  'man_start', 'man_end', 'out_start', 'out_end']:
            if k not in manoeuvre_spec:
                manoeuvre_spec[k] = None
        clip['start'] = manoeuvre_spec['anim_start']
        clip['end'] = manoeuvre_spec['anim_end']
    else:
        manoeuvre_spec = None

    update_fn, frame_nums = prepare_panels(
        args=args,
        reconstruction=reconstruction,
        start_frame=clip['start'] if 'start' in clip else 0,
        end_frame=clip['end'] if 'end' in clip else 0,
        caption=clip['caption'] if 'caption' in clip else '',
        azim_offset=clip['azim_offset'] if 'azim_offset' in clip else 0,
        distance=clip['distance'] if 'distance' in clip else args.distance,
        manoeuvre_spec=manoeuvre_spec,
    )

    # Initialise ffmpeg process
    output_path = output_dir / f'{clip_idx:03d}_trial={reconstruction.trial.id}_r={reconstruction.id}_f={clip["start"]}-{clip["end"]}'
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': args.fps,
        # 'metadata:g:0': f'title=Trial {reconstruction.trial.id}. Reconstruction {reconstruction.id}. Frame #{clip["frame_num"]}',
        # 'metadata:g:1': 'artist=Leeds Wormlab',
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
        if args.manoeuvres_mode:
            assert 'manoeuvre_spec' in clip
        else:
            assert 'manoeuvre_spec' not in clip
            assert 'start' in clip
            assert 'end' in clip
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
