import os
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mayavi import mlab
from mayavi.core.scene import Scene
from tvtk.tools import visual

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab, plot_arrow
from wormlab3d.toolkit.plot_utils import make_box_from_pca_mlab, to_rgb
from wormlab3d.toolkit.util import print_args, to_dict, str2bool, normalise, orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.manoeuvres import get_manoeuvres
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

show_plots = False
save_plots = True


# show_plots = True
# save_plots = False


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot a turn.')
    parser.add_argument('--spec', type=str, help='Load spec from file (relative to logs path).')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')

    # Sections
    parser.add_argument('--traj-start', type=int, help='Frame number to start trajectory from.')
    parser.add_argument('--traj-end', type=int, help='Frame number to end trajectory at.')
    parser.add_argument('--inc-start', type=int, help='Frame number to start incoming box from.')
    parser.add_argument('--inc-end', type=int, help='Frame number to end incoming box at.')
    parser.add_argument('--man-start', type=int, help='Frame number to start manoeuvre box from.')
    parser.add_argument('--man-end', type=int, help='Frame number to end manoeuvre box at.')
    parser.add_argument('--out-start', type=int, help='Frame number to start outgoing box from.')
    parser.add_argument('--out-end', type=int, help='Frame number to end outgoing box at.')

    # Trajectory
    parser.add_argument('--trajectory-point', type=int, default=-1, help='Trajectory point.')
    parser.add_argument('--smoothing-window-trajectories', type=int, default=0,
                        help='Smoothing window for the trajectories.')
    parser.add_argument('--smoothing-window-postures', type=int, default=0, help='Smoothing window for the postures.')
    parser.add_argument('--smoothing-window-components', type=int, default=0,
                        help='Smoothing window for the components.')
    parser.add_argument('--smoothing-window-speed', type=int, default=25,
                        help='Smoothing window for the speed calculation.')

    # Postures
    parser.add_argument('--posture-frames', type=lambda s: [int(item) for item in s.split(',')],
                        help='Frame numbers to add postures at.')

    # 3D plots
    parser.add_argument('--width-3d', type=int, default=1000, help='Width of 3D plot (in pixels).')
    parser.add_argument('--height-3d', type=int, default=1000, help='Height of 3D plot (in pixels).')
    parser.add_argument('--distance', type=float, default=1., help='Camera distance (in worm lengths).')
    parser.add_argument('--azimuth', type=int, default=70, help='Azimuth.')
    parser.add_argument('--elevation', type=int, default=45, help='Elevation.')
    parser.add_argument('--roll', type=int, default=45, help='Roll.')
    parser.add_argument('--draw-midline', type=str2bool, default=True, help='Draw the midline.')

    # Curvature kymogram
    parser.add_argument('--K-max', type=float, help='Maximum curvature value for consistency between plots.')
    parser.add_argument('--K-ticks', type=lambda s: [float(item) for item in s.split(',')],
                        help='Curvature colourbar ticks.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')

    # Diagram args
    parser.add_argument('--frame-num', type=int, default=0, help='Find the closest manoeuvre to this frame.')
    parser.add_argument('--min-forward-frames', type=int, default=25,
                        help='Minimum number of forward frames before counting a forward locomotion section.')
    parser.add_argument('--min-forward-speed', type=float, default=0.,
                        help='Minimum speed of forward frames before counting a forward locomotion section.')
    parser.add_argument('--min-reversal-frames', type=int, default=25,
                        help='Minimum number of reversal frames to use to identify a manoeuvre.')
    parser.add_argument('--min-reversal-distance', type=float, default=0.,
                        help='Minimum reversal distance to use to identify a manoeuvre.')
    parser.add_argument('--manoeuvre-window', type=int, default=500,
                        help='Number of frames to include either side of a detected manoeuvre.')
    parser.add_argument('--arrow-idx', type=int, default=0,
                        help='Arrow index to show.')

    args = parser.parse_args()

    return args


def _get_args() -> Tuple[Namespace, Dict]:
    """
    Combine command line arguments with any from a spec.yml file.
    """
    args = parse_args()
    spec = None
    if args.spec is not None:
        spec_path = LOGS_PATH / f'spec_{args.spec}.yml'
        with open(spec_path) as f:
            logger.info(f'Using spec file: {spec_path}.')
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            setattr(args, k, v)
    print_args(args)
    return args, spec


def _make_3d_plot(
        X_postures: np.ndarray,
        X_trajectory: np.ndarray,
        inc_start_idx: int,
        inc_end_idx: int,
        man_start_idx: int,
        man_end_idx: int,
        out_start_idx: int,
        out_end_idx: int,
        posture_idxs: List[int],
        speeds: np.ndarray,
        curvatures: np.ndarray,
        lengths: np.ndarray,
        args: Namespace,
) -> Figure:
    """
    Build a 3D trajectory plot with worm using mayavi.
    """
    logger.info('Building 3D plot.')
    distance = lengths.max() * args.distance

    # Construct colours
    s = speeds
    cmap = plt.get_cmap('PRGn')
    vmax = np.abs(s).max()
    vmin = -vmax
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Set up mlab figure
    fig = mlab.figure(size=(args.width_3d * 2, args.height_3d * 2), bgcolor=(1, 1, 1))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 64
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Render the trajectory with simple lines
    path = mlab.plot3d(*X_trajectory.T, s, vmax=vmax, vmin=vmin, opacity=0.8, tube_radius=None, line_width=9)
    # path = mlab.plot3d(*X_trajectory.T, s, vmax=vmax, vmin=vmin, opacity=0.8, tube_radius=0.015)
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

    for posture_idx in posture_idxs:
        # Set up the artist and add the pieces
        NF = NaturalFrame(X_postures[posture_idx])
        fa = FrameArtistMLab(
            NF,
            use_centred_midline=False,
            midline_opts={'opacity': 1, 'line_width': 8},
            surface_opts={'radius': 0.025 * lengths.mean()}
        )
        if args.draw_midline:
            fa.add_midline(fig)
        fa.add_surface(fig, v_min=-curvatures.max(), v_max=curvatures.max())

    # Focus on the middle of the manoeuvre
    centre = X_trajectory[inc_start_idx:out_end_idx].mean(axis=0)
    # centre = X_trajectory[int((man_start_idx+man_end_idx)/2)]
    # R = np.stack(NF.pca.components_, axis=1)
    # Xt = np.einsum('ij,bj->bi', R.T, fa.X)
    # centre = (Xt.min(axis=0) + Xt.ptp(axis=0) / 2) @ R.T

    # Draw plot
    mlab.view(
        figure=fig,
        azimuth=args.azimuth,
        elevation=args.elevation,
        roll=args.roll,
        distance=distance,
        focalpoint=centre
    )

    # # Useful for getting the view parameters when recording from the gui:
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [2.142281616991216, -7.713877055407108, 2046.5581680107207]
    # scene.scene.camera.focal_point = [2.480087156597625, -1.9935202537692394, 2048.4720827891188]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.1997558322493267, 0.30026756289517975, -0.9327041321637684]
    # scene.scene.camera.clipping_range = [2.731068387833279, 9.671117651682236]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    return fig


def _make_kymogram(
        reconstruction: Reconstruction,
        K: np.ndarray,
        inc_start_idx: int,
        inc_end_idx: int,
        man_start_idx: int,
        man_end_idx: int,
        out_start_idx: int,
        out_end_idx: int,
        posture_idxs: List[int],
        args: Namespace
) -> Figure:
    """
    Plot a kymogram of the curvature over time.
    """

    # Adjust ranges and indices
    K = K[inc_start_idx:out_end_idx]
    inc_end_idx -= inc_start_idx
    man_start_idx -= inc_start_idx
    man_end_idx -= inc_start_idx
    out_start_idx -= inc_start_idx
    out_end_idx -= inc_start_idx

    # Get times
    N = len(K)
    ts = np.arange(N)
    if args.x_label == 'time':
        fps = reconstruction.trial.fps
        ts = ts / fps
    else:
        fps = 1

    # Plot
    plt.rc('axes', labelsize=8)  # fontsize of the axes labels
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)  # fontsize of the y tick labels
    fig, ax = plt.subplots(1, figsize=(4, 1.35), gridspec_kw={
        'left': 0.10,
        'right': 0.9,
        'top': 0.95,
        'bottom': 0.24,
    })

    # Plot curvature
    K_max = np.max(K) if args.K_max is None else args.K_max
    im = ax.imshow(K.T, aspect='auto', cmap='Reds', origin='lower', extent=(ts[0], ts[-1], 0, 1), vmin=0, vmax=K_max)

    # Draw boxes around the sections
    rect_args = dict(
        height=1.02,
        linewidth=3,
        facecolor='none',
        linestyle='-',
        clip_on=False
    )
    ax.add_patch(Rectangle(
        (0, -0.01),
        inc_end_idx / fps,
        edgecolor='aqua',
        **rect_args
    ))
    ax.add_patch(Rectangle(
        (man_start_idx / fps, -0.01),
        (man_end_idx - man_start_idx) / fps,
        edgecolor='hotpink',
        **rect_args
    ))
    ax.add_patch(Rectangle(
        (out_start_idx / fps, -0.01),
        (out_end_idx - out_start_idx) / fps,
        edgecolor='aqua',
        **rect_args
    ))

    # Highlight frames displaying postures
    ax.vlines(x=(np.array(posture_idxs) - inc_start_idx) / fps, ymin=0, ymax=1, colors='blue', linewidth=2)

    # Colourbar
    cax = ax.inset_axes([1.03, 0.1, 0.03, 0.8], transform=ax.transAxes)
    fig.colorbar(im, ax=ax, cax=cax, ticks=args.K_ticks)

    # Axes
    ht_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold')
    ax.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax.set_ylabel('Curvature (mm$^{-1}$)', labelpad=10)
    ax.set_yticks([0, 1])
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticklabels([])
    ax.set_xlim(left=0, right=ts[-1])
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)', labelpad=1)
    else:
        ax.set_xlabel('Frame #', labelpad=1)

    # plt.show()
    # exit()

    return fig


def plot_turn():
    """
    Plot a kymogram of the curvature over time.
    """
    args, spec = _get_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial

    # Get sections
    logger.info(f'Preparing panel for reconstruction {reconstruction.id}.')
    assert all([
        getattr(args, f'{k}_{se}') is not None
        for k in ['traj', 'inc', 'man', 'out']
        for se in ['start', 'end']
    ]), 'Missing range value(s)!'
    r_start_frame = max(reconstruction.start_frame_valid, args.traj_start)
    r_end_frame = min(reconstruction.end_frame_valid, args.traj_end)

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)
    Xrc = Xr - Xr.mean(axis=0)

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

    # Get curvature
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=args.smoothing_window_components)
    K = np.abs(Z)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        lengths = ts.get('length', r_start_frame, r_end_frame + 1)[:, 0]
    else:
        lengths = np.linalg.norm(Xpc[:, 1:] - Xpc[:, :-1], axis=-1).sum(axis=-1)

    # Calculate speed
    if args.smoothing_window_speed > 1:
        Xc_smoothed = smooth_trajectory(Xrc, window_len=args.smoothing_window_speed)
        speeds = calculate_speeds(Xc_smoothed, signed=True) * trial.fps
    else:
        speeds = calculate_speeds(Xrc, signed=True) * trial.fps

    # Convert frame numbers to indices
    man_start_idx = args.man_start - r_start_frame
    man_end_idx = args.man_end - r_start_frame
    inc_start_idx = max(0, args.inc_start - r_start_frame)
    inc_end_idx = args.inc_end - r_start_frame
    out_start_idx = args.out_start - r_start_frame
    out_end_idx = min(len(Xr), args.out_end - r_start_frame)
    posture_idxs = [f - r_start_frame for f in args.posture_frames]

    fig_k = _make_kymogram(
        reconstruction=reconstruction,
        K=K,
        inc_start_idx=inc_start_idx,
        inc_end_idx=inc_end_idx,
        man_start_idx=man_start_idx,
        man_end_idx=man_end_idx,
        out_start_idx=out_start_idx,
        out_end_idx=out_end_idx,
        posture_idxs=posture_idxs,
        args=args
    )

    fig_3d = _make_3d_plot(
        X_postures=Xp,
        X_trajectory=Xt,
        inc_start_idx=inc_start_idx,
        inc_end_idx=inc_end_idx,
        man_start_idx=man_start_idx,
        man_end_idx=man_end_idx,
        out_start_idx=out_start_idx,
        out_end_idx=out_end_idx,
        posture_idxs=posture_idxs,
        speeds=speeds,
        curvatures=K,
        lengths=lengths,
        args=args,
    )

    if save_plots:
        # Copy the spec with final args to the output dir
        spec_dir = LOGS_PATH / f'{trial.id:03d}_{reconstruction.id}'
        output_dir = spec_dir / f'{START_TIMESTAMP}_f={args.traj_start}-{args.traj_end}'
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / 'spec.yml', 'w') as f:
            spec['created'] = START_TIMESTAMP
            spec['args'] = to_dict(args)
            yaml.dump(spec, f)

        path = output_dir / '3D.png'
        logger.info(f'Saving 3D plot to {path}.')
        fig_3d.scene._lift()
        img = mlab.screenshot(figure=fig_3d, mode='rgba', antialiased=True)
        img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
        img.save(path)
        mlab.clf(fig_3d)
        mlab.close()

        path = output_dir / 'kymogram.svg'
        logger.info(f'Saving kymogram to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()
        mlab.show()


def plot_manoeuvre_diagram(
        plot_full: bool = True,
        plot_arrow_pairs: bool = True,
):
    """
    Plot a manoeuvre diagram.
    """
    args, spec = _get_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial
    r_start_frame = max(reconstruction.start_frame_valid, args.traj_start)
    r_end_frame = min(reconstruction.end_frame_valid, args.traj_end)
    keys = ['prev', 'rev', 'next']

    # Set up output directory
    if save_plots:
        spec_dir = LOGS_PATH / f'{trial.id:03d}_{reconstruction.id}_diagrams'
        output_dir = spec_dir / f'{START_TIMESTAMP}_f={args.traj_start}-{args.traj_end}_a={args.arrow_idx}'
        os.makedirs(output_dir, exist_ok=True)

        # Copy the spec with final args to the output dir
        if spec is not None:
            with open(output_dir / 'spec.yml', 'w') as f:
                spec['created'] = START_TIMESTAMP
                spec['args'] = to_dict(args)
                yaml.dump(spec, f)

    # Fetch raw posture data
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    Xr, _ = get_trajectory(**common_args)
    Xrc = Xr - Xr.mean(axis=0)

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
    Xtc = Xt - Xt.mean(axis=0)

    # Smooth the postures and centre
    if args.smoothing_window_speed > 1:
        Xp = smooth_trajectory(Xr, window_len=args.smoothing_window_speed)
    else:
        Xp = Xr
    Xpc = Xp - Xp.mean(axis=(0, 1))

    # Calculate speed
    if args.smoothing_window_speed > 1:
        Xc_smoothed = smooth_trajectory(Xrc, window_len=args.smoothing_window_speed)
        speeds = calculate_speeds(Xc_smoothed, signed=True) * trial.fps
    else:
        speeds = calculate_speeds(Xrc, signed=True) * trial.fps

    # Get the manoeuvres
    logger.info('Calculating manoeuvre info.')
    manoeuvres = get_manoeuvres(
        Xpc,
        Xtc,
        min_reversal_frames=args.min_reversal_frames,
        min_reversal_distance=args.min_reversal_distance,
        window_size=args.manoeuvre_window
    )

    # Identify the manoeuvre to use
    frame_nums = np.array([m['centre_idx'] for m in manoeuvres]) + r_start_frame
    idx = np.argmin(np.abs(frame_nums - args.frame_num))
    m = manoeuvres[idx]

    # Calculate arrows
    arrows = {}
    for k in keys:
        X = m[f'X_{k}']
        pca = m[f'pca_{k}']
        arrows[k] = []

        # Get the bounds relative to the PCA components
        R = np.stack(pca.components_, axis=1)
        X2 = np.einsum('ij,bj->bi', R.T, X)
        dims = np.ptp(X2, axis=0)
        centre = (X2.min(axis=0) + dims / 2) @ R.T

        for arrow_idx in [0, 1, 2]:
            vec = pca.components_[arrow_idx] * dims[arrow_idx]
            if vec.sum() == 0:
                continue
            if np.dot(vec, X[-1] - X[0]) < 0:
                vec *= -1

            origin = centre - vec / 2
            scale = np.linalg.norm(vec)
            arrows[k].append((origin, vec, scale))

    # Construct trajectory colours
    cmap = plt.get_cmap('PRGn')
    vmax = np.abs(speeds).max()
    vmin = -vmax
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Box args
    def cuboid_args(fig_):
        return dict(
            dimensions='extents',
            opacity=0.2,
            draw_outline=True,
            outline_opacity=0.8,
            outline_tube_radius=0.001,
            fig=fig_
        )

    box_colours = {
        'prev': {'colour': 'aqua', 'outline_colour': 'teal'},
        'rev': {'colour': 'plum', 'outline_colour': 'hotpink'},
        'next': {'colour': 'lime', 'outline_colour': 'green'},
    }

    # Arrow args
    def arrow_args(fig_, scale_):
        return dict(
            fig=fig_,
            opacity=0.8,
            radius_shaft=0.01 / scale_,
            radius_cone=0.05 / scale_,
            length_cone=np.clip(0.07 / scale_, a_min=0, a_max=1)
        )

    def _setup_fig() -> Scene:
        # Set up mlab figure
        fig_ = mlab.figure(size=(args.width_3d * 2, args.height_3d * 2), bgcolor=(1, 1, 1))
        visual.set_viewer(fig_)

        # Depth peeling required for nice opacity, the rest don't seem to make any difference
        fig_.scene.renderer.use_depth_peeling = True
        fig_.scene.renderer.maximum_number_of_peels = 64
        fig_.scene.render_window.point_smoothing = True
        fig_.scene.render_window.line_smoothing = True
        fig_.scene.render_window.polygon_smoothing = True
        fig_.scene.render_window.multi_samples = 20
        fig_.scene.anti_aliasing_frames = 20
        return fig_

    if plot_full:
        logger.info('Making 3D plot.')
        fig = _setup_fig()

        # Render the trajectory with simple lines
        path = mlab.plot3d(*Xtc.T, speeds, vmax=vmax, vmin=vmin, opacity=0.8, tube_radius=None, line_width=2)
        path.module_manager.scalar_lut_manager.lut.table = cmaplist

        # Add the boxes and arrows
        for k in keys:
            # Add box
            make_box_from_pca_mlab(
                X=m[f'X_{k}'],
                pca=m[f'pca_{k}'],
                **box_colours[k],
                **cuboid_args(fig)
            )

            # Add arrow
            origin, vec, scale = arrows[k][args.arrow_idx]
            plot_arrow(
                origin=origin,
                vec=vec,
                color=to_rgb(box_colours[k]['outline_colour']),
                **arrow_args(fig, scale)
            )

        # Focus on the middle of the manoeuvre
        centre = Xtc[m['centre_idx']]

        # Draw plot
        mlab.view(
            figure=fig,
            azimuth=args.azimuth,
            elevation=args.elevation,
            roll=args.roll,
            distance=args.distance,
            focalpoint=centre
        )

        # # Useful for getting the view parameters when recording from the gui:
        # scene = mlab.get_engine().scenes[0]
        # scene.scene.camera.position = [-5.972938416217623, -2.1295553699734877, -2.7230607255604427]
        # scene.scene.camera.focal_point = [0.09356396148032275, 0.09305462400974118, -0.029075997312929758]
        # scene.scene.camera.view_angle = 30.0
        # scene.scene.camera.view_up = [0.48178792311728674, -0.7330292120486714, -0.48015473695636424]
        # scene.scene.camera.clipping_range = [3.236838845789775, 11.032963464505043]
        # scene.scene.camera.compute_view_plane_normal()
        # scene.scene.render()
        # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
        # print(mlab.roll())
        # exit()

        if save_plots:
            path = output_dir / 'diagram.png'
            logger.info(f'Saving 3D plot to {path}.')
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)
            mlab.clf(fig)
            mlab.close()

        if show_plots:
            plt.show()
            mlab.show()

    if plot_arrow_pairs:
        # Plot pairs of arrows
        origin = np.zeros(3)
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if j <= i:
                    continue
                logger.info(f'Plotting arrows {k1}->{k2}.')
                _, v1, s1 = arrows[k1][args.arrow_idx]
                _, v2, s2 = arrows[k2][args.arrow_idx]
                fig = _setup_fig()

                # Plot pairs of arrows
                plot_arrow(
                    origin=origin,
                    vec=normalise(v1) / 2,
                    color=to_rgb(box_colours[k1]['outline_colour']),
                    **arrow_args(fig, 1 / 3)
                )
                plot_arrow(
                    origin=origin,
                    vec=normalise(v2) / 2,
                    color=to_rgb(box_colours[k2]['outline_colour']),
                    **arrow_args(fig, 1 / 3)
                )

                # Draw the angle
                b0 = normalise(v1)
                b1 = normalise(orthogonalise(v2, b0))
                b2 = np.cross(b0, b1)
                R = np.stack([b0, b1, b2])
                r = 0.2
                nc = 1000
                thetas = np.linspace(np.pi / 2, np.pi / 2 - calculate_angle(v1, v2), nc)
                xs = r * np.sin(thetas)
                ys = r * np.cos(thetas)
                zs = np.zeros(nc)
                pts = np.stack([xs, ys, zs], axis=1) @ R
                mlab.plot3d(*pts.T, color=to_rgb('orange'), tube_radius=0.01)

                # Draw plot
                mlab.view(
                    figure=fig,
                    azimuth=args.azimuth,
                    elevation=args.elevation,
                    roll=args.roll,
                    distance=3,
                    focalpoint=origin
                )

                if save_plots:
                    path = output_dir / f'arrows_{k1}-{k2}.png'
                    logger.info(f'Saving 3D plot to {path}.')
                    fig.scene._lift()
                    img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                    img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                    img.save(path)
                    mlab.clf(fig)
                    mlab.close()

                if show_plots:
                    plt.show()
                    mlab.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if show_plots:
        interactive()
    else:
        mlab.options.offscreen = True

    # plot_turn()
    plot_manoeuvre_diagram(
        plot_full=True,
        plot_arrow_pairs=True
    )
