import itertools
import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath
from typing import Dict, Any
from typing import Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from mayavi import mlab
from mayavi.core.scene import Scene
from tvtk.tools import visual

from simple_worm.frame import FRAME_COMPONENT_KEYS
from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import PREPARED_IMAGES_PATH
from wormlab3d import logger, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.f0_state import F0State
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab, plot_arrow
from wormlab3d.toolkit.plot_utils import overlay_image, to_rgb
from wormlab3d.toolkit.util import print_args, str2bool, to_dict

# Off-screen rendering
mlab.options.offscreen = True

m1_colour = 'deepskyblue'
m2_colour = 'mediumseagreen'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate an f0-optimisation video.')

    # Target
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--frame-num', type=int, help='Frame number to show.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')

    # 3D plot
    parser.add_argument('--show-3d-axis', type=str2bool, default=True,
                        help='Show axis on the 3D plot.')
    parser.add_argument('--revolution-rate', type=float, default=0.01,
                        help='Rate of 3D plot revolution in revolutions/step.')
    parser.add_argument('--distance', type=float, default=3.,
                        help='Camera distance in worm lengths.')

    args = parser.parse_args()
    assert args.spec is not None, 'This script requires setting --spec=path.'

    return args


def _make_info_panel(
        width: int,
        height: int,
        X: np.ndarray,
        lengths: np.ndarray,
) -> Tuple[Figure, Callable]:
    """
    Info panel.
    """
    logger.info('Building infos plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')

    def get_details(step) -> str:
        return f'Step: {step:4d} / {len(X)}\n' \
               f'Length: {lengths[step]:.3f}mm'

    # Details
    text = fig.text(0.05, 0.95, get_details(0), ha='left', va='top',
                    fontsize=18, linespacing=1.5, fontweight='bold')

    def update(step: int):
        # Update the text
        text.set_text(get_details(step))

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def _make_axes(
        fig: Scene,
        centre: np.ndarray = np.array([0, 0, 0]),
        scale: float = 1.,
        colour: str = 'darkgrey',
        tube_radius: float = 0.0008
):
    """
    Make a cuboid axes
    """
    M = np.array(list(itertools.product(*[[-1, 1]] * 3)))
    v = centre + M / 2 * scale

    # Bottom face outline
    l1 = np.stack([v[0], v[1]])
    l2 = np.stack([v[1], v[3]])
    l3 = np.stack([v[3], v[2]])
    l4 = np.stack([v[2], v[0]])

    # Top face outline
    l5 = np.stack([v[4], v[5]])
    l6 = np.stack([v[5], v[7]])
    l7 = np.stack([v[7], v[6]])
    l8 = np.stack([v[6], v[4]])

    # Connecting vertical lines
    l9 = np.stack([v[0], v[4]])
    l10 = np.stack([v[1], v[5]])
    l11 = np.stack([v[2], v[6]])
    l12 = np.stack([v[3], v[7]])

    # Stack the lines together
    lines = np.array([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])

    # Add lines to plot
    outline_opts = {
        'figure': fig,
        'color': to_rgb(colour),
        'tube_radius': tube_radius,
    }
    for l in lines:
        mlab.plot3d(*l.T, **outline_opts)


def _make_3d_plot(
        width: int,
        height: int,
        X: np.ndarray,
        T0: np.ndarray,
        M10: np.ndarray,
        curvatures: np.ndarray,
        lengths: np.ndarray,
        azim_offset: int,
        distance: float,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D worm plot.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D plot.')
    T = len(X)
    N2 = int(X.shape[1] / 2)
    M20 = np.cross(T0, M10)
    min_arrow_size = 0.05
    max_arrow_size = 0.15
    arrow_sizes = min_arrow_size + (max_arrow_size - min_arrow_size) * (lengths - lengths.min()) / (
                lengths.max() - lengths.min())
    # frame_arrow_scales = 0.15 * lengths
    frame_vectors = {
        'e0': T0 * arrow_sizes[:, None],
        'e1': M10 * arrow_sizes[:, None],
        'e2': M20 * arrow_sizes[:, None],
    }

    # Set up mlab figure
    fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))
    visual.set_viewer(fig)

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Set up the artist and add the pieces
    NF = NaturalFrame(X[0])
    fa = FrameArtistMLab(
        NF,
        use_centred_midline=False,
        midline_opts={'opacity': 1, 'line_width': 8},
        surface_opts={'radius': 0.024 * lengths[0]},
        arrow_opts={
            'opacity': 0.9,
            'radius_shaft': 0.03,
            'radius_cone': 0.13,
            'length_cone': 0.22
        },
        arrow_colours={
            'e0': 'grey',
            'e1': m1_colour,
            'e2': m2_colour,
        },
        arrow_scale=0.15,
    )

    # Add midline and surface
    fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvatures.max(), v_max=curvatures.max())

    # Add frame vectors
    arrows = {}
    for k in FRAME_COMPONENT_KEYS:
        arrows[k] = plot_arrow(
            fig=fig,
            origin=fa.X[N2],
            vec=frame_vectors[k][0],
            color=to_rgb(fa.arrow_colours[k]),
            **fa.arrow_opts
        )

    # Add the axis
    _make_axes(fig, scale=lengths[-1])

    # Aspects
    n_revolutions = T * args.revolution_rate
    azims = azim_offset + np.linspace(start=0, stop=360 * n_revolutions, num=T)
    mlab.view(figure=fig, azimuth=azims[0], distance=distance, focalpoint=np.zeros(3))

    def update(step: int):
        nonlocal arrows
        fig.scene.disable_render = True
        NF = NaturalFrame(X[step])
        fa.surface_opts['radius'] = 0.024 * lengths[step]
        fa.update(NF)
        if step > 0:
            for k in FRAME_COMPONENT_KEYS:
                arrows[k].pos = np.array(fa.X[N2]) / arrow_sizes[step]
                arrows[k].axis = [*frame_vectors[k][step]]
                arrows[k].actor.trait_set(scale=[arrow_sizes[step], arrow_sizes[step], arrow_sizes[step]])

        fig.scene.disable_render = False
        mlab.view(figure=fig, azimuth=azims[step], distance=distance, focalpoint=np.zeros(3))
        fig.scene.render()

    return fig, update


def _make_curvature_plots(
        width: int,
        height: int,
        curvatures: np.ndarray,
) -> Tuple[Figure, Callable]:
    """
    Build a curvatures plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building curvatures plot.')
    N = curvatures.shape[1]
    kappa = np.linalg.norm(curvatures, axis=-1)

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    plt.rc('axes', labelsize=18)  # fontsize of the axis labels
    plt.rc('xtick', labelsize=16)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=16)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=5, size=16)
    plt.rc('ytick.major', pad=5, size=16)

    fig, ax = plt.subplots(1, figsize=(width / 100, height / 100), gridspec_kw={
        'left': 0.13,
        'right': 0.96,
        'top': 0.98,
        'bottom': 0.21,
    })
    ax.spines['top'].set_visible(False)

    # Set up x-axis
    ax.set_xticks([])
    ax.set_xlim(left=0, right=N - 1)
    ax.set_xticks([0, N - 1])
    ax.set_xticklabels(['H', 'T'])

    # Set up y-axis
    ylim = kappa.max() * 1.05
    ax.set_ylim(bottom=0, top=ylim)
    ax.set_yticks([0, int(ylim)])
    ax.set_ylabel('Curvature $\mathregular{mm^{-1}}$')

    lines = []
    for n in range(N - 1):
        line = ax.plot(ind[n:n + 2], kappa[0, n:n + 2], c=fc[n])
        lines.append(line[0])

    # # Add n0 label
    # trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.text(start_idx, -0.08, '$n_0$', color=cmaplist[start_idx] / 255, fontsize=7, fontweight='bold',
    #         horizontalalignment='center', verticalalignment='top', transform=trans)
    # ax.axvline(x=start_idx, ymin=-0.1, ymax=0.92, linestyle=':', color='grey')

    def update(step: int):
        nonlocal lines
        for n, l in enumerate(lines):
            l.set_data(ind[n:n + 2], kappa[step, n:n + 2])

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
    images = images.transpose(1, 0, 2)
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


def prepare_panels(
        args: Namespace,
        reconstruction: Reconstruction,
        frame_num: int,
        azim_offset: int,
        distance: float,
):
    """
    Prepare the panel of plots for the given reconstruction.
    """
    logger.info(f'Preparing panel for reconstruction {reconstruction.id}.')
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions supported!'
    trial = reconstruction.trial
    F0S = F0State(reconstruction, trial.get_frame(frame_num))

    # Get variables
    X = F0S.get('points')
    T0 = F0S.get('T0')[:, 0]
    M10 = F0S.get('M10')[:, 0]
    K = F0S.get('curvatures') * (X.shape[1] - 1)
    lengths = F0S.get('length')[:, 0]
    cam_coeffs = np.concatenate([
        F0S.get(f'cam_{k}')
        for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
    ], axis=2)
    prs = ProjectRenderScoreModel(image_size=trial.crop_size)

    # Get base points from trial state
    ts = TrialState(reconstruction, start_frame=frame_num)
    points_3d_base = torch.from_numpy(ts.get('points_3d_base')[0][None, ...].astype(np.float32))
    points_2d_base = torch.from_numpy(ts.get('points_2d_base')[0][None, ...].astype(np.float32))

    # Build plots
    imgs_width = int(args.height / 3)

    fig_info, update_info_plot = _make_info_panel(
        width=int(args.width - imgs_width),
        height=int(args.height / 3 * 2),
        X=X,
        lengths=lengths
    )
    fig_3d, update_3d_plot = _make_3d_plot(
        width=int(args.width - imgs_width),
        height=int(args.height / 3 * 2),
        X=X,
        T0=T0,
        M10=M10,
        curvatures=K,
        lengths=lengths,
        azim_offset=azim_offset,
        distance=distance,
        args=args,
    )
    fig_curvatures, update_curvatures_plot = _make_curvature_plots(
        width=int(args.width - imgs_width),
        height=int(args.height / 3),
        curvatures=K,
    )

    # Load images
    img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{frame_num:06d}.npz'
    image_triplet = np.load(img_path)['images']
    if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
        raise RuntimeError('Prepared images are the wrong size, regeneration needed!')

    def prepare_step(step: int) -> np.ndarray:
        # Generate the annotated images
        points_2d = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs[step][None, ...]),
            points_3d=torch.from_numpy(X[step][None, ...]),
            points_3d_base=points_3d_base,
            points_2d_base=points_2d_base,
        )
        points_2d = points_2d[0].numpy().transpose(1, 0, 2)
        points_2d = np.round(points_2d).astype(np.int32)

        # Prepare images
        images = _generate_annotated_images(
            width=imgs_width,
            height=int(args.height),
            image_triplet=image_triplet,
            points_2d=points_2d
        )

        # Update the plots and extract renders
        update_info_plot(step)
        update_3d_plot(step)
        update_curvatures_plot(step)

        plot_info = np.asarray(fig_info.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_curvatures = np.asarray(fig_curvatures.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_3d = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_3d)
        plot_3d = cv2.resize(plot_3d, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Overlay plot info on top of 3d panel
        plot_3d = overlay_image(plot_3d, plot_info, x_offset=0, y_offset=0)
        # plot_3d[-1] = 0

        # Join plots and images and write to stream
        frame = np.concatenate([
            images,
            np.concatenate([plot_3d, plot_curvatures], axis=0)
        ], axis=1)

        return frame

    return prepare_step, len(X)


def generate_f0_video(
        args: Namespace,
        clip: Dict[str, Any],
        output_dir: PosixPath,
        clip_idx: int
):
    """
    Generate a f0 optimisation video showing a rotating 3D reconstructed
    worm and camera images with overlaid 2D midline reprojections.
    """
    reconstruction = Reconstruction.objects.get(id=clip['reconstruction'])
    update_fn, n_steps = prepare_panels(
        args=args,
        reconstruction=reconstruction,
        frame_num=clip['frame_num'],
        azim_offset=clip['azim_offset'] if 'azim_offset' in clip else 0,
        distance=clip['distance'] if 'distance' in clip else args.distance
    )

    # Initialise ffmpeg process
    output_path = output_dir / f'{clip_idx:03d}_trial={reconstruction.trial.id}_r={reconstruction.id}_f={clip["frame_num"]}'
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

    logger.info('Rendering steps.')
    for step in range(n_steps):
        if step > 0 and step % 50 == 0:
            logger.info(f'Rendering step {step + 1}/{n_steps}.')

        # Update the frame and write to stream
        frame = update_fn(step)
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
        generate_f0_video(
            args,
            clip,
            output_dir=output_dir,
            clip_idx=i
        )
        plt.close('all')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_from_spec()
