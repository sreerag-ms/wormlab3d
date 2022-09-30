import os
import time
from argparse import Namespace
from typing import List, Union, Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from mayavi import mlab
from mayavi.core.scene import Scene
from tvtk.tools import visual

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.cache import generate_or_load_r_values, get_sim_state_from_args, get_npas_from_args
from wormlab3d.toolkit.plot_utils import overlay_image, to_rgb
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import smooth_trajectory

show_plots = False
save_plots = True
img_extension = 'png'
use_sigma_idxs = [3, 11, 19]


def _calculate_volumes(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate the spherical cut volumes.
    """
    z = np.clip(z, a_min=0, a_max=r)
    sphere_vols = 4 / 3 * np.pi * r**3
    cap_vols = 1 / 3 * np.pi * (r - z)**2 * (2 * r + z)
    return sphere_vols - 2 * cap_vols


def make_filename(
        method: str,
        args: Namespace, excludes: List[str] = None,
        extension: str = img_extension,
        timestamp: str = None
):
    if excludes is None:
        excludes = []
    if timestamp is None:
        timestamp = START_TIMESTAMP
    fn = f'{timestamp}_{method}'

    for k in ['npas', 'voxel_sizes', 'duration', 'dt', 'batch_size', 'deltas', 'delta_step',
              'targets_radii', 'n_targets', 'epsilon', 'max_nonplanar_pause_duration', 'detection_area',
              'plot_n_trajectories_per_sigma', 'pick_trajectories_on', 'approx_noise', 'smoothing_window']:
        if k in excludes:
            continue
        if k == 'npas':
            if len(args.npas) > 5:
                npas = f'{args.npas[0]:.1E}-{args.npas[-1]:.1E}'
            else:
                npas = ','.join(f'{npa:.1E}' for npa in args.npas)
            fn += f'_npas={npas}'
        elif k == 'voxel_sizes':
            if len(args.voxel_sizes) > 5:
                voxel_sizes = f'{args.voxel_sizes[0]:.1E}-{args.voxel_sizes[-1]:.1E}'
            else:
                voxel_sizes = ','.join(f'{vs:.1E}' for vs in args.voxel_sizes)
            fn += f'_vs={voxel_sizes}'
        elif k == 'duration':
            fn += f'_T={args.sim_duration:.1f}'
        elif k == 'dt':
            fn += f'_dt={args.sim_dt}'
        elif k == 'batch_size':
            fn += f'_bs={args.batch_size}'
        elif k == 'deltas':
            fn += f'_d={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            fn += f'_ds={args.delta_step}'
        elif k == 'targets_radii' and hasattr(args, 'targets_radii'):
            if len(args.targets_radii) > 5:
                targets_radii = f'{args.targets_radii[0]:.1E}-{args.targets_radii[-1]:.1E}'
            else:
                targets_radii = ','.join(f'{r:.1E}' for r in args.targets_radii)
            fn += f'_r={targets_radii}'
        elif k == 'n_targets' and hasattr(args, 'n_targets'):
            fn += f'_targets={args.n_targets}'
        elif k == 'epsilon' and hasattr(args, 'epsilon'):
            fn += f'_eps={args.epsilon}'
        elif k == 'max_nonplanar_pause_duration':
            fn += f'_p={args.nonp_pause_max:.1f}'
        elif k == 'detection_area' and hasattr(args, 'detection_area'):
            fn += f'_da={args.detection_area:.2f}'
        elif k == 'plot_n_trajectories_per_sigma' and hasattr(args, 'plot_n_trajectories_per_sigma'):
            fn += f'_nt={args.plot_n_trajectories_per_sigma}'
        elif k == 'pick_trajectories_on' and hasattr(args, 'pick_trajectories_on'):
            fn += f'_{args.pick_trajectories_on}'
        elif k == 'approx_noise' and args.approx_noise is not None:
            fn += f'_ns={args.approx_noise}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'

    if extension is not None:
        fn += '.' + extension

    return LOGS_PATH / fn


def _get_sphere_slice_border_points(
        r: float,
        h: float,
        n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = np.arccos(h / r)
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(angle, np.pi - angle, n_points)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(n_points), np.cos(v))
    return x, y, z


def _plot_sphere_slice_border(
        r: float,
        h: float,
        c: Union[str, np.ndarray],
        n_points: int,
        alpha: float = 1.,
        fig: Scene = None
):
    x, y, z = _get_sphere_slice_border_points(r, h, n_points)
    return mlab.mesh(x, y, z, color=to_rgb(c), opacity=alpha, figure=fig)


def _plot_sphere_slice_caps(
        r: float,
        h: float,
        c: Union[str, np.ndarray],
        n_points: int,
        alpha: float = 1.
) -> Tuple:
    angle = np.arccos(h / r)
    radius = r * np.sin(angle)
    r2 = np.linspace(0, radius, n_points)
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.outer(r2, np.sin(theta))
    y = np.outer(r2, np.cos(theta))
    z = h * np.ones((n_points, n_points))
    top = mlab.mesh(x, y, z, color=to_rgb(c), opacity=alpha)
    bottom = mlab.mesh(x, y, -z, color=to_rgb(c), opacity=alpha)
    return top, bottom


def _plot_sphere_slice(
        r: float,
        h: float,
        c: Union[str, np.ndarray],
        n_points: int,
        alpha: float = 1.
) -> Tuple:
    border = _plot_sphere_slice_border(r, h, c, n_points, alpha)
    top, bottom = _plot_sphere_slice_caps(r, h, c, n_points, alpha)
    return border, top, bottom


def _plot_trajectory(
        X: np.ndarray,
        alpha: float = 1.,
        c=None
):
    x, y, z = X.T
    t = np.linspace(0, 1, len(X))
    pts = mlab.plot3d(x, y, z, t, opacity=alpha, color=c, tube_radius=0.08)
    if c is None:
        cmap = plt.get_cmap('viridis_r')
        cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
        pts.module_manager.scalar_lut_manager.lut.table = cmaplist

    return pts


def _plot_arrow(
        origin=(0, 0, 0),
        dest=(1, 0, 0),
        c=None
):
    x1, y1, z1 = origin
    x2, y2, z2 = dest
    ar1 = visual.arrow(x=x1, y=y1, z=z1, color=c, radius_shaft=0.01, radius_cone=0.05)
    ar1.length_cone = 0.1
    arrow_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos / arrow_length
    ar1.axis = [x2 - x1, y2 - y1, z2 - z1]
    return ar1


def spherical_cut_plot():
    """
    3D plot of spherical cuts explored by different trajectories.
    """
    args = get_args(validate_source=False)

    npa_sigmas = get_npas_from_args(args)
    args.npas = npa_sigmas
    args.pauses = [args.nonp_pause_max]
    args.sim_durations = [args.sim_duration]
    r_values = generate_or_load_r_values(args, rebuild_cache=False, cache_only=False)
    # r_values = np.zeros((n_sigmas, n_durations, n_pauses, 3, 4))

    npa_sigmas = npa_sigmas[use_sigma_idxs]
    r0 = r_values[use_sigma_idxs, 0, 0, 0, 0]
    r2 = r_values[use_sigma_idxs, 0, 0, 2, 0]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas

    n_points = 100
    n_radius_line_points = 100
    plot_n_trajectories_per_sigma = 3

    # Plot the sphere-slices
    logger.info('Plotting results.')
    mlab.options.offscreen = save_plots
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 16
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Axis arrows
    r_max = r0.max()
    z_max = r2.max()
    angle = np.pi / 2
    z_max_offset = 4
    axis_colour = to_rgb('red')
    _plot_arrow(
        dest=(0, 0, z_max),
        c=axis_colour
    )
    _plot_arrow(
        origin=(0, 0, z_max - z_max_offset),
        dest=(r_max * np.sin(angle), r_max * np.cos(angle), z_max - z_max_offset),
        c=axis_colour
    )

    # Sphere and trajectory colours
    # sphere_colours = plt.get_cmap('Blues')(np.linspace(0, 1, n_sigmas))  # best
    # sphere_colours = plt.get_cmap('cubehelix')(np.linspace(0.3, 0.9, n_sigmas))
    sphere_colours = np.array([
        [30, 73, 21, 1],
        [210, 140, 190, 1],
        [220, 230, 235, 1],
    ]) / 255

    alphas = np.linspace(0.6, 0.3, n_sigmas)
    for i, npas in enumerate(npa_sigmas):
        colour = tuple(sphere_colours[i][:3])

        # Fetch trajectories
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)
        Xt = SS.get_Xt()

        # Pick trajectories which come closest to the average r and z values
        r0_dist = np.abs(np.max(np.abs(Xt[:, :, 0]), axis=1) - r0[i])
        r2_dist = np.abs(np.max(np.abs(Xt[:, :, 2]), axis=1) - r2[i])
        best_fit_idxs = np.argsort(r0_dist * r2_dist)[:plot_n_trajectories_per_sigma]
        for idx in best_fit_idxs:
            _plot_trajectory(Xt[idx], c=colour)
        _plot_sphere_slice(r0[i], r2[i], colour, n_points, alphas[i])

        # Plot arrows to the radius
        # _plot_arrow(*(r_arrow_coords[i]*r0[i]), c=_get_rgb('red'))
        # _plot_arrow(*(r_arrow_coords[i] * r0[i]), c=colour)

        x = r0[i] * np.sin(angle) * np.ones(n_radius_line_points)
        y = r0[i] * np.cos(angle) * np.ones(n_radius_line_points)
        z = np.linspace(0, z_max - z_max_offset, n_radius_line_points)
        mlab.plot3d(x, y, z, color=colour, tube_radius=0.1, tube_sides=12, opacity=alphas[i])

    # Axis labels
    mlab.text3d(-4 * np.sin(angle), -4 * np.cos(angle), z_max - 1.5, 'z', color=axis_colour, scale=2)
    mlab.text3d((r_max - 4) * np.sin(angle), (r_max - 4) * np.cos(angle), z_max - z_max_offset + 2, 'r',
                color=axis_colour, scale=2)

    # Set view
    engine = mlab.get_engine()
    scene = engine.scenes[0]
    scene.scene.camera.position = [20.296384025198293, -76.73687499518573, 21.010969231727806]
    scene.scene.camera.focal_point = [7.663823205133351, 18.717819517660857, -3.8185610377551735]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.017041998433029205, 0.25381652066998583, 0.967102240781393]
    scene.scene.camera.clipping_range = [3.3694726367009906, 168.98804117041172]
    scene.scene.camera.compute_view_plane_normal()

    if save_plots:
        fig.scene._lift()
        screenshot = mlab.screenshot(mode='rgb', antialiased=True)
        img = Image.fromarray(screenshot, 'RGB')
        img.save(
            make_filename(
                'spherical_cuts',
                args,
                excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon']
            )
        )

    if show_plots:
        mlab.show()


def spherical_cut_animation():
    """
    3D animation of spherical cuts explored by different trajectories.
    """
    args = get_args(validate_source=False)

    npa_sigmas = get_npas_from_args(args)
    args.npas = npa_sigmas
    args.pauses = [args.nonp_pause_max]
    args.sim_durations = [args.sim_duration]
    r_values = generate_or_load_r_values(args, rebuild_cache=False, cache_only=True)

    npa_sigmas = npa_sigmas[use_sigma_idxs]
    r0 = r_values[use_sigma_idxs, 0, 0, 0, 0]
    r2 = r_values[use_sigma_idxs, 0, 0, 2, 0]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas

    width = 1280
    height = 720
    n_points = 100
    args.plot_n_trajectories_per_sigma = 10
    traj_anim_rate = 100
    T = int(args.sim_duration / args.sim_dt)

    # Set up plot
    logger.info('Instantiating renderer.')
    mlab.options.offscreen = save_plots
    fig = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 16
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Axis arrows
    axis_colour = to_rgb('red')
    z_arrow = _plot_arrow(dest=(0, 0, 1), c=axis_colour)
    r_arrow = _plot_arrow(dest=(1, 0, 0), c=axis_colour)

    # Sphere and trajectory colours
    # sphere_colours = plt.get_cmap('Blues')(np.linspace(0, 1, n_sigmas))  # best
    # sphere_colours = plt.get_cmap('cubehelix')(np.linspace(0.3, 0.9, n_sigmas))
    sphere_colours = np.array([
        [38, 83, 31, 1],
        [210, 140, 190, 1],
        [220, 230, 235, 1],
    ]) / 255

    trajectories = {i: [] for i in range(n_sigmas)}
    paths = {i: [] for i in range(n_sigmas)}
    vols = []

    alphas = np.linspace(0.2, 0.4, n_sigmas)
    for i, npas in enumerate(npa_sigmas):
        colour = tuple(sphere_colours[i][:3])

        # Fetch trajectories
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)
        Xt = SS.get_Xt()

        # Pick trajectories which come closest to the average r and z values
        r0_dist = np.abs(np.max(np.abs(Xt[:, :, 0]), axis=1) - r0[i])
        r2_dist = np.abs(np.max(np.abs(Xt[:, :, 2]), axis=1) - r2[i])
        best_fit_idxs = np.argsort(r0_dist * r2_dist)[:args.plot_n_trajectories_per_sigma]
        for idx in best_fit_idxs:
            traj = Xt[idx]
            trajectories[i].append(traj)
            x, y, z = traj[0].T
            path = mlab.plot3d(x, y, z, color=colour, tube_radius=0.08)
            paths[i].append(path)
        vol = _plot_sphere_slice_border(0.1, 0.1, colour, n_points, alphas[i])
        vols.append(vol)

    def update_scene(t):
        fig.scene.disable_render = True

        # Update paths
        max_dist = 0
        max_r = 0
        max_z = 0
        for i, npas in enumerate(npa_sigmas):
            max_r_i = 0
            max_z_i = 0

            for j, traj in enumerate(trajectories[i]):
                X = traj[:t * traj_anim_rate]
                max_dist = max(max_dist, np.max(np.linalg.norm(X, axis=-1)))
                x, y, z = X.T
                max_z_i = max(max_z_i, np.max(np.abs(z)))
                max_r_i = max(max_r_i, np.max(np.abs(X[:, :2])), max_z_i)
                paths[i][j].mlab_source.reset(x=x, y=y, z=z)

            max_z = max(max_z, max_z_i)
            max_r = max(max_r, max_r_i, max_z)

            # Update vol
            x, y, z = _get_sphere_slice_border_points(max_r_i, max_z_i, n_points)
            vols[i].mlab_source.reset(x=x, y=y, z=z)

        # Update arrows
        z_arrow.actor.trait_set(scale=[max_z, max_z, max_z])
        r_arrow.actor.trait_set(scale=[max_r, max_r, max_r])

        fig.scene.disable_render = False

        mlab.view(
            azimuth=np.fmod(t / 2, 360),
            elevation=np.sin(t / 50) * 20 + 90,
            distance=10 + max_dist * 3,
            focalpoint=[0, 0, 0]
        )

        fig.scene.render()

    if save_plots:
        # Initialise ffmpeg process
        output_path = make_filename(
            'sim_animation',
            args,
            excludes=['voxel_sizes', 'deltas', 'delta_step', 'targets_radii',
                      'n_targets', 'epsilon', 'max_nonplanar_pause_duration', 'detection_area'],
            extension=None
        )
        output_args = {
            'pix_fmt': 'yuv444p',
            'vcodec': 'libx264',
            'r': 25,
            'metadata:g:0': 'title=Simulation output.',
            'metadata:g:1': 'artist=Leeds Wormlab',
            'metadata:g:2': f'year={time.strftime("%Y")}',
        }

        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                .output(str(output_path) + '.mp4', **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )

        fig.scene._lift()
        for t in range(1, int(T / traj_anim_rate)):
            update_scene(t)
            screenshot = mlab.screenshot(mode='rgb', antialiased=True)
            frame = Image.fromarray(screenshot, 'RGB')
            process.stdin.write(frame.tobytes())

        # Flush video
        process.stdin.close()
        process.wait()

    if show_plots:
        @mlab.animate(delay=50)
        def animate():
            for t in range(1, T):
                update_scene(t)
                yield

        animate()
        mlab.show()


def _make_label_overlay(
        width: int,
        height: int,
        text: str
) -> np.ndarray:
    """
    Label overlay.
    """
    logger.info('Building label overlay plot.')
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.text(0.1, 0.9, text, ha='left', va='top', fontsize=12, linespacing=1.5)
    fig.tight_layout()
    fig.canvas.draw()
    overlay = np.asarray(fig.canvas.renderer._renderer).take([0, 1, 2], axis=2)
    plt.close(fig)
    return overlay


def _make_traces_plots(
        width: int,
        height: int,
        sigma_labels: List[str],
        rs: np.ndarray,
        zs: np.ndarray,
        vs: np.ndarray,
        fps: int
) -> Tuple[Figure, Callable]:
    """
    Build a traces plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building traces plot.')
    n_sigmas = len(sigma_labels)
    N = rs.shape[1]
    ts = np.linspace(0, N / fps, N)

    # Plot
    fig, axes = plt.subplots(3, figsize=(width / 100, height / 100), gridspec_kw={
        'hspace': 0.6,
        'top': 0.93,
        'bottom': 0.07,
        'left': 0.1,
        'right': 0.9,
    })

    r_plots = []
    ax_r = axes[0]
    ax_r.set_title('Radius (planar distance)')
    for i, sigma in enumerate(sigma_labels):
        p = ax_r.plot(rs[i][0], label=f'$\sigma$ = {sigma}')
        r_plots.append(p[0])
    # ax_r.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_r.transAxes, fontsize=18)
    ax_r.legend(loc='upper left', fontsize=14)
    ax_r.set_xlabel('Time (s)')

    z_plots = []
    ax_z = axes[1]
    ax_z.set_title('Height (non-planar distance)')
    for i, sigma in enumerate(sigma_labels):
        p = ax_z.plot(zs[i][0], label=f'$\sigma$ = {sigma}')
        z_plots.append(p[0])
    # ax_z.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_z.transAxes)
    ax_z.set_xlabel('Time (s)')

    v_plots = []
    ax_v = axes[2]
    ax_v.set_title('Volume explored')
    for i, sigma in enumerate(sigma_labels):
        p = ax_v.plot(vs[i][0], label=f'$\sigma$ = {sigma}')
        v_plots.append(p[0])
    # ax_v.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_v.transAxes)
    ax_v.set_xlabel('Time (s)')

    def update(frame_idx: int):
        # Update the data
        for i in range(n_sigmas):
            r_plots[i].set_data(ts[:frame_idx], rs[i, :frame_idx])
            z_plots[i].set_data(ts[:frame_idx], zs[i, :frame_idx])
            v_plots[i].set_data(ts[:frame_idx], vs[i, :frame_idx])

        # Update the limits
        for ax in axes:
            ax.set_xlim(left=0, right=ts[frame_idx] + 5)
        ax_r.set_ylim(bottom=0, top=np.max(rs[:, :frame_idx]) * 1.5)
        ax_z.set_ylim(bottom=0, top=np.max(zs[:, :frame_idx]) * 1.5)
        ax_v.set_ylim(bottom=0, top=np.max(vs[:, :frame_idx]) * 1.5)

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def spherical_cut_stacked_animation():
    """
    3D stacked animation of spherical cuts explored by different trajectories.
    """
    args = get_args(validate_source=False)

    npa_sigmas = get_npas_from_args(args)
    args.npas = npa_sigmas
    args.pauses = [args.nonp_pause_max]
    args.sim_durations = [args.sim_duration]
    r_values = generate_or_load_r_values(args, rebuild_cache=False, cache_only=True)

    npa_sigmas = npa_sigmas[use_sigma_idxs]
    r0 = r_values[use_sigma_idxs, 0, 0, 0, 0]
    r2 = r_values[use_sigma_idxs, 0, 0, 2, 0]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas

    # sigma_labels = ['0.003', '0.6', '10']
    sigma_labels = [f'{s:.2E}' for s in npa_sigmas]

    width = 1280
    height = 720
    n_points = 100
    traj_anim_rate = 100
    args.plot_n_trajectories_per_sigma = 1  # 5
    args.pick_trajectories_on = ['averages', 'extremes', 'exemplars'][2]
    T = int(args.sim_duration / args.sim_dt)

    # Set up plot
    logger.info('Instantiating renderer.')
    mlab.options.offscreen = save_plots
    # visual.set_viewer(fig)

    # Sphere and trajectory colours
    colours = np.array([
        [31, 119, 180, 1],
        [255, 127, 13, 1],
        [86, 179, 86, 1],
    ]) / 255

    figs = {}
    trajectories = np.zeros((n_sigmas, args.plot_n_trajectories_per_sigma, T, 3))
    paths = {i: [] for i in range(n_sigmas)}
    vols = {}
    alphas = [0.5, 0.5, 0.5]
    label_overlays = {}

    # Select trajectories
    Xt_all = []
    for i, npas in enumerate(npa_sigmas):
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)
        Xt_all.append(SS.get_Xt())

    if args.pick_trajectories_on == 'exemplars':

        # For the low-sigma case we want a large r0, pick one closest to the average
        Xt_low = Xt_all[0]
        r0_low = np.max(np.abs(Xt_low[:, :, 0]), axis=1)
        r2_low = np.max(np.abs(Xt_low[:, :, 2]), axis=1)
        # best_fit_idxs = np.argsort(np.abs(r0_low - r0[0]))[:args.plot_n_trajectories_per_sigma]
        best_fit_idxs = np.argsort(r0_low)[-args.plot_n_trajectories_per_sigma:]
        Xt_low = Xt_low[best_fit_idxs]
        r0_low = r0_low[best_fit_idxs]
        r2_low = r2_low[best_fit_idxs]
        r_low = np.max(np.linalg.norm(Xt_low, axis=-1), axis=1)

        # For the high-sigma case we want large r2/r0 ratio
        Xt_high = Xt_all[2]
        r0_high = np.max(np.abs(Xt_high[:, :, 0]), axis=1)
        r2_high = np.max(np.abs(Xt_high[:, :, 2]), axis=1)
        best_fit_idxs = np.argsort(r2_high / r0_high)[-args.plot_n_trajectories_per_sigma:]
        Xt_high = Xt_high[best_fit_idxs]
        r0_high = r0_high[best_fit_idxs]
        r2_high = r2_high[best_fit_idxs]
        r_high = np.max(np.linalg.norm(Xt_high, axis=-1), axis=1)

        # For the middle case we want to have largest volume, but lower r0 than low-case and lower r2 than high-case
        # Eliminate out of range cases and then find example that fits closest to the middle of the other two.
        Xt_mid = Xt_all[1]
        r0_mid = np.max(np.abs(Xt_mid[:, :, 0]), axis=1)
        r2_mid = np.max(np.abs(Xt_mid[:, :, 2]), axis=1)
        vols_low = max(_calculate_volumes(r0_low, r2_low))
        vols_high = max(_calculate_volumes(r0_high, r2_high))
        vols_mid = _calculate_volumes(r0_mid, r2_mid)
        idxs_out_of_range = (vols_mid < max(vols_low, vols_high)) | (r0_mid > min(r0_low)) | (r2_mid > min(r2_high))
        if idxs_out_of_range.sum() == Xt_mid.shape[0]:
            raise RuntimeError('Unable to find mid-sigma exemplar.')
        r0_mid[idxs_out_of_range] = 0
        r2_mid[idxs_out_of_range] = 0
        r_mid = np.max(np.linalg.norm(Xt_mid, axis=-1), axis=1)
        best_r0 = np.abs(r_mid - (min(r_low) + max(r_high)) / 2)
        # best_r0 = np.abs(r0_mid - (min(r0_low) + max(r0_high)) / 2)
        best_r2 = np.abs(r2_mid - (max(r2_low) + min(r2_high)) / 2)
        best_fit_idxs = np.argsort(best_r0 * best_r2)[:args.plot_n_trajectories_per_sigma]
        Xt_mid = Xt_mid[best_fit_idxs]

        trajectories[0] = Xt_low
        trajectories[1] = Xt_mid
        trajectories[2] = Xt_high

    else:
        for i, npas in enumerate(npa_sigmas):
            Xt = Xt_all[i]
            if args.pick_trajectories_on == 'averages':
                # Pick trajectories which come closest to the average r and z values
                r_dist = np.abs(np.max(np.abs(Xt[:, :, 0]), axis=1) - r0[i])
                z_dist = np.abs(np.max(np.abs(Xt[:, :, 2]), axis=1) - r2[i])
                best_fit_idxs = np.argsort(r_dist * z_dist)[:args.plot_n_trajectories_per_sigma]
            elif args.pick_trajectories_on == 'extremes':
                # Pick trajectories which find the extremes
                r_dist = np.max(np.abs(Xt[:, :, 0]), axis=1)
                z_dist = np.max(np.abs(Xt[:, :, 2]), axis=1)

                if i == 0:
                    best_fit_idxs = np.argsort(r_dist)[-args.plot_n_trajectories_per_sigma:]
                elif i == 1:
                    vs = _calculate_volumes(r_dist, z_dist)
                    best_fit_idxs = np.argsort(vs)[-args.plot_n_trajectories_per_sigma:]
                elif i == 2:
                    best_fit_idxs = np.argsort(z_dist)[-args.plot_n_trajectories_per_sigma:]
                else:
                    raise RuntimeError('Extreme picking only works with 3 sigmas!')
            else:
                raise RuntimeError(f'Unrecognised pick_trajectories_on option: "{args.pick_trajectories_on}".')

            for j, idx in enumerate(best_fit_idxs):
                trajectories[i, j] = Xt[idx]

    # Add some noise and smooth
    if args.approx_noise is not None and args.approx_noise > 0:
        trajectories = trajectories + np.random.normal(np.zeros_like(trajectories), args.approx_noise)
    if args.smoothing_window is not None and args.smoothing_window > 0:
        for i in range(n_sigmas):
            trajectories[i] = smooth_trajectory(
                trajectories[i].transpose(1, 0, 2),
                window_len=args.smoothing_window
            ).transpose(1, 0, 2)

    # Generate separate figures for each sigma
    for i, npas in enumerate(npa_sigmas):
        colour = tuple(colours[i][:3])

        # Setup figure
        fig = mlab.figure(size=(width, height / n_sigmas * 2), bgcolor=(1, 1, 1))
        fig.scene.renderer.use_depth_peeling = True
        fig.scene.renderer.maximum_number_of_peels = 16
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20
        fig.scene.anti_aliasing_frames = 20
        figs[i] = fig

        # Add trajectory path
        for j in range(args.plot_n_trajectories_per_sigma):
            x, y, z = trajectories[i, j, 0].T
            path = mlab.plot3d(x, y, z, color=colour, tube_radius=0.08, figure=figs[i])
            paths[i].append(path)

        # Add sphere slice
        vols[i] = _plot_sphere_slice_border(0.1, 0.1, colour, n_points, alphas[i], fig=figs[i])

        # Build label overlays
        label_overlays[i] = _make_label_overlay(width / 2, height / n_sigmas, f'$\sigma_{{\phi}}$ = {sigma_labels[i]}')

    # Calculate the max r and z values over time
    logger.info('Calculating r and z values over time.')
    rs = np.maximum.accumulate(
        np.max(np.linalg.norm(trajectories, axis=-1), axis=1),
        axis=-1
    )
    zs = np.maximum.accumulate(
        np.max(np.abs(trajectories[..., 2]), axis=1),
        axis=-1
    )
    vs = _calculate_volumes(rs, zs)

    fig_traces, update_traces_plot = _make_traces_plots(
        width=int(width / 2),
        height=height,
        sigma_labels=sigma_labels,
        rs=rs,
        zs=zs,
        vs=vs,
        fps=25
    )

    def update_scenes(t):
        # Update paths
        max_dist = 0
        for i, npas in enumerate(npa_sigmas):
            figs[i].scene.disable_render = True

            for j, traj in enumerate(trajectories[i]):
                X = traj[:t]
                max_dist = max(max_dist, np.max(np.linalg.norm(X, axis=-1)))
                x, y, z = X.T
                paths[i][j].mlab_source.reset(x=x, y=y, z=z)

            max_z = zs[i, t]
            max_r = max(max_z, rs[i, t])

            # Update vol
            x, y, z = _get_sphere_slice_border_points(max_r, max_z, n_points)
            vols[i].mlab_source.reset(x=x, y=y, z=z)

            figs[i].scene.disable_render = False

            mlab.view(
                figure=figs[i],
                azimuth=np.fmod(t / traj_anim_rate / 2, 360),
                elevation=np.sin(t / traj_anim_rate / 50) * 20 + 90,
                distance=10 + max_dist * 3,
                focalpoint=[0, 0, 0]
            )

            figs[i].scene.render()

    if save_plots:
        # Initialise ffmpeg process
        output_path = make_filename(
            'sim_animation_stacked',
            args,
            excludes=['voxel_sizes', 'deltas', 'delta_step', 'targets_radii',
                      'n_targets', 'epsilon', 'max_nonplanar_pause_duration', 'detection_area'],
            extension=None
        )
        output_args = {
            'pix_fmt': 'yuv444p',
            'vcodec': 'libx264',
            'r': 25,
            'metadata:g:0': 'title=Simulation output.',
            'metadata:g:1': 'artist=Leeds Wormlab',
            'metadata:g:2': f'year={time.strftime("%Y")}',
        }

        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
                .output(str(output_path) + '.mp4', **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )

        # Overlay plot info on top of images panel
        fig.scene._lift()
        n_frames = int(T / traj_anim_rate)
        for t in range(1, n_frames):
            if t > 1 and t % 50 == 0:
                logger.info(f'Rendering frame {t}/{n_frames}.')
            update_scenes(t * traj_anim_rate)
            update_traces_plot(t * traj_anim_rate)

            # Take the traces render
            plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)

            # Collate the 3D views with overlaid labels
            frames_t = []
            for i, fig in figs.items():
                screenshot = mlab.screenshot(mode='rgb', antialiased=True, figure=figs[i])
                screenshot = cv2.resize(screenshot, dsize=(int(width / 2), int(height / n_sigmas)),
                                        interpolation=cv2.INTER_AREA)
                screenshot = overlay_image(screenshot, label_overlays[i], x_offset=0, y_offset=0)
                frames_t.append(screenshot)

            # Stitch together the data
            frame_data = np.concatenate([np.concatenate(frames_t), plot_traces], axis=1)
            if frame_data.shape[0] < height:
                frame_data = np.concatenate([
                    frame_data,
                    np.ones((height - frame_data.shape[0], width, 3), dtype=np.uint8) * 255
                ], axis=0)
            if frame_data.shape[1] < width:
                frame_data = np.concatenate([
                    frame_data,
                    np.ones((height, width - frame_data.shape[1], 3), dtype=np.uint8) * 255
                ], axis=1)
            if frame_data.shape[0] != height or frame_data.shape[1] != width:
                raise RuntimeError('Frame is the wrong shape!')

            frame = Image.fromarray(frame_data, 'RGB')
            process.stdin.write(frame.tobytes())

        # Flush video
        process.stdin.close()
        process.wait()

    if show_plots:
        @mlab.animate(delay=50)
        def animate():
            for t in range(1, T):
                update_scenes(t)
                yield

        animate()
        mlab.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # spherical_cut_plot()
    # spherical_cut_animation()
    spherical_cut_stacked_animation()
