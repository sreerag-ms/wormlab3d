import os
import time
from argparse import Namespace
from typing import List, Union, Tuple, Callable

import cv2
import ffmpeg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from mayavi import mlab
from mayavi.core.scene import Scene
from tvtk.tools import visual

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.toolkit.plot_utils import overlay_image
from wormlab3d.trajectories.args import get_args

show_plots = False
save_plots = True
img_extension = 'png'

# These are precomputed r and z values for these values of sigma and the paper model parameters
npa_sigmas_paper = np.exp(-np.linspace(np.log(1 / 1e-3), np.log(1 / 10), 40))
r_values_paper = np.array([
    [26.37047768, 6.84866905, 70.91493225],
    [27.19985962, 7.74319458, 72.75009155],
    [26.91914368, 8.41468239, 68.18685913],
    [26.36344528, 7.57023907, 64.47218323],
    [27.34775734, 7.5232935, 65.14431],
    [26.24323273, 7.579772, 64.04511261],
    [26.69629288, 8.04461288, 68.75421143],
    [27.00673866, 8.23388767, 65.1728363],
    [26.4631176, 7.39619112, 62.38437653],
    [27.02355003, 6.8745985, 67.19996643],
    [26.49530411, 9.71176338, 65.22090912],
    [27.03031731, 9.03608131, 67.66876984],
    [26.97320557, 9.03722572, 68.81982422],
    [26.10933304, 8.33324814, 68.69045258],
    [26.47162437, 9.14206982, 63.11143112],
    [26.31864929, 8.24293709, 66.95050049],
    [25.86649704, 8.35997486, 67.63995361],
    [26.26771164, 8.50308418, 55.94869614],
    [26.22031403, 8.10858822, 58.816185],
    [25.63489532, 7.80216837, 59.96321487],
    [25.50696564, 6.69798708, 61.31138229],
    [25.24788284, 7.69211674, 63.32733154],
    [25.43347359, 7.82677364, 60.58054733],
    [24.5610199, 7.82066059, 61.56003189],
    [24.34826469, 7.49853516, 58.16346741],
    [24.5643959, 8.99473953, 59.14283752],
    [23.24162102, 7.33277702, 53.19667816],
    [22.79305458, 7.15097713, 54.79700089],
    [21.61912918, 7.62393951, 62.38457108],
    [20.44259644, 6.73174572, 50.67710876],
    [19.97979736, 6.60961008, 51.74533463],
    [20.12223816, 6.02390242, 52.63835144],
    [19.81240654, 7.09848022, 44.82326508],
    [19.81694984, 6.48989868, 44.4970932],
    [19.61680984, 6.26188135, 46.92141342],
    [19.62089539, 5.56906462, 42.59923935],
    [19.76632881, 6.74197769, 42.18218231],
    [19.98529625, 5.59029579, 44.85945892],
    [19.52000427, 6.24363708, 44.64799118],
    [19.71105385, 6.21947765, 46.49293518],
])

z_values_paper = np.array([
    [1.08439319e-01, 1.87683702e-02, 6.01948261e-01, ],
    [1.41735539e-01, 3.08232307e-02, 6.11835480e-01, ],
    [1.77712068e-01, 3.77369523e-02, 6.16229773e-01, ],
    [2.25317895e-01, 5.17663956e-02, 8.93105507e-01, ],
    [2.82044739e-01, 5.75218201e-02, 1.21878231e+00, ],
    [3.64291877e-01, 8.59701633e-02, 1.65892792e+00, ],
    [4.72018182e-01, 9.90438461e-02, 2.67598438e+00, ],
    [5.79967558e-01, 1.13871574e-01, 2.49173665e+00, ],
    [7.50766039e-01, 1.72143698e-01, 2.91986942e+00, ],
    [9.23524320e-01, 1.80970252e-01, 3.47769451e+00, ],
    [1.15508449e+00, 2.55134583e-01, 5.03332901e+00, ],
    [1.48506308e+00, 3.76707554e-01, 8.42977333e+00, ],
    [1.89409244e+00, 4.10907745e-01, 9.44216537e+00, ],
    [2.37065363e+00, 5.48722506e-01, 8.62445450e+00, ],
    [2.86767054e+00, 6.58947468e-01, 9.88036060e+00, ],
    [3.58742666e+00, 7.11676121e-01, 1.66881027e+01, ],
    [4.54275608e+00, 8.75735998e-01, 1.63172379e+01, ],
    [5.42247248e+00, 1.32271576e+00, 1.72629490e+01, ],
    [6.22321367e+00, 1.88727093e+00, 1.58069534e+01, ],
    [7.10748529e+00, 1.93884349e+00, 1.95304203e+01, ],
    [7.66998672e+00, 2.59216404e+00, 1.76536484e+01, ],
    [8.16336536e+00, 2.01386356e+00, 1.94519310e+01, ],
    [8.27294540e+00, 3.10204148e+00, 1.81460800e+01, ],
    [8.35521507e+00, 3.16583729e+00, 2.00180779e+01, ],
    [8.39850426e+00, 3.53299546e+00, 2.14322472e+01, ],
    [8.13499165e+00, 3.26218414e+00, 2.00810871e+01, ],
    [7.76640797e+00, 3.31614304e+00, 1.72585583e+01, ],
    [7.56845236e+00, 3.23078966e+00, 1.81797028e+01, ],
    [7.25984192e+00, 3.31856561e+00, 1.74473343e+01, ],
    [6.92943573e+00, 2.69637632e+00, 1.60080967e+01, ],
    [6.64932537e+00, 2.12277746e+00, 1.77084217e+01, ],
    [6.67393494e+00, 2.92974925e+00, 1.60686054e+01, ],
    [6.75314474e+00, 2.66229773e+00, 1.38273354e+01, ],
    [6.64066124e+00, 2.80084324e+00, 1.52757807e+01, ],
    [6.61620855e+00, 2.96917820e+00, 1.61377792e+01, ],
    [6.48537922e+00, 2.43766141e+00, 1.49967880e+01, ],
    [6.70059824e+00, 2.64727187e+00, 1.63553982e+01, ],
    [6.48779869e+00, 2.55504751e+00, 1.69105816e+01, ],
    [6.56086636e+00, 2.55093932e+00, 1.62724724e+01, ],
    [6.57849884e+00, 2.51942348e+00, 1.49009752e+01, ],
])


def _calculate_volumes(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate the spherical cut volumes.
    """
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
              'plot_n_trajectories_per_sigma', 'pick_trajectories_on']:
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

    if extension is not None:
        fn += '.' + extension

    return LOGS_PATH / fn


def _get_rgb(c: Union[str, np.ndarray]):
    if type(c) == str:
        return mcolors.to_rgb(c)
    return c


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
    return mlab.mesh(x, y, z, color=_get_rgb(c), opacity=alpha, figure=fig)


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
    top = mlab.mesh(x, y, z, color=_get_rgb(c), opacity=alpha)
    bottom = mlab.mesh(x, y, -z, color=_get_rgb(c), opacity=alpha)
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

    use_idxs = [4, 16, 28]
    npa_sigmas = npa_sigmas_paper[use_idxs]
    z_values = z_values_paper[use_idxs]
    r_values = r_values_paper[use_idxs]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas

    n_points = 100
    n_radius_line_points = 100
    plot_n_trajectories_per_sigma = 1

    # # Sweep over the nonplanarity angle sigmas
    # for i, npas in enumerate(npa_sigmas):
    #     logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
    #     args.phi_dist_params[1] = npas
    #     pe, TC = get_trajectories_from_args(args)
    #
    #     # Find the maximums in each relative directions
    #     Xt = TC.get_Xt()
    #     Xs.append(Xt)
    #     # Xt_max = np.abs(Xt).max(axis=1)
    #
    #     # # Use first component as radius and third as height of explored disk
    #     # r_values[i] = [Xt_max[:, 0].mean(), Xt_max[:, 0].min(), Xt_max[:, 0].max()]
    #     # z_values[i] = [Xt_max[:, 2].mean(), Xt_max[:, 2].min(), Xt_max[:, 2].max()]
    #     #
    #     # if TC.needs_save:
    #     #     TC.save()
    Xs = np.load(LOGS_PATH / 'trajectories.npz', mmap_mode=True)['Xs']

    # Plot the sphere-slices
    logger.info('Plotting results.')
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Axis arrows
    r_max = r_values[:, 0].max()
    z_max = z_values.max()
    angle = np.pi / 2
    z_max_offset = 4
    axis_colour = _get_rgb('red')
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
        [38, 83, 31, 1],
        [210, 140, 190, 1],
        [220, 230, 235, 1],
    ]) / 255

    alphas = np.linspace(0.2, 0.6, n_sigmas)
    for i, npas in enumerate(npa_sigmas):
        colour = tuple(sphere_colours[i][:3])

        # Pick trajectories which come closest to the average r and z values
        r_dist = np.abs(np.max(np.abs(Xs[i][:, :, 0]), axis=1) - r_values[i, 0])
        z_dist = np.abs(np.max(np.abs(Xs[i][:, :, 2]), axis=1) - z_values[i, 0])
        best_fit_idxs = np.argsort(r_dist * z_dist)[:plot_n_trajectories_per_sigma]
        for idx in best_fit_idxs:
            _plot_trajectory(Xs[i][idx], c=colour)
        _plot_sphere_slice(r_values[i, 0], z_values[i, 0], colour, n_points, alphas[i])

        # Plot arrows to the radius
        # _plot_arrow(*(r_arrow_coords[i]*r_values[i, 0]), c=_get_rgb('red'))
        # _plot_arrow(*(r_arrow_coords[i] * r_values[i, 0]), c=colour)

        x = r_values[i, 0] * np.sin(angle) * np.ones(n_radius_line_points)
        y = r_values[i, 0] * np.cos(angle) * np.ones(n_radius_line_points)
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

    use_idxs = [4, 16, 28]
    npa_sigmas = npa_sigmas_paper[use_idxs]
    z_values = z_values_paper[use_idxs]
    r_values = r_values_paper[use_idxs]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas
    args.sim_duration = 60 * 60

    width = 1280
    height = 720
    n_points = 100
    args.plot_n_trajectories_per_sigma = 10
    traj_anim_rate = 100
    Xs = np.load(LOGS_PATH / 'trajectories.npz', mmap_mode=True)['Xs']
    T = Xs.shape[2]

    # Set up plot
    logger.info('Instantiating renderer.')
    mlab.options.offscreen = save_plots
    fig = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Axis arrows
    axis_colour = _get_rgb('red')
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

        # Pick trajectories which come closest to the average r and z values
        r_dist = np.abs(np.max(np.abs(Xs[i][:, :, 0]), axis=1) - r_values[i, 0])
        z_dist = np.abs(np.max(np.abs(Xs[i][:, :, 2]), axis=1) - z_values[i, 0])
        best_fit_idxs = np.argsort(r_dist * z_dist)[:args.plot_n_trajectories_per_sigma]
        for idx in best_fit_idxs:
            traj = Xs[i][idx]
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

    # use_idxs = [4, 16, 28]
    use_idxs = [4, 27, -1]
    npa_sigmas = npa_sigmas_paper[use_idxs]
    sigma_labels = ['0.003', '0.6', '10']
    z_values = z_values_paper[use_idxs]
    r_values = r_values_paper[use_idxs]
    n_sigmas = len(npa_sigmas)
    args.npas = npa_sigmas
    args.sim_duration = 60 * 60

    width = 1280
    height = 720
    n_points = 100
    traj_anim_rate = 100
    args.plot_n_trajectories_per_sigma = 1  # 5
    args.pick_trajectories_on = ['averages', 'extremes'][1]
    Xs = np.load(LOGS_PATH / 'trajectories.npz', mmap_mode=True)['Xs']
    T = Xs.shape[2]

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

    # Generate separate figures for each sigma
    for i, npas in enumerate(npa_sigmas):
        fig = mlab.figure(size=(width, height / n_sigmas * 2), bgcolor=(1, 1, 1))
        fig.scene.anti_aliasing_frames = 20
        figs[i] = fig
        colour = tuple(colours[i][:3])

        if args.pick_trajectories_on == 'average':
            # Pick trajectories which come closest to the average r and z values
            r_dist = np.abs(np.max(np.abs(Xs[i][:, :, 0]), axis=1) - r_values[i, 0])
            z_dist = np.abs(np.max(np.abs(Xs[i][:, :, 2]), axis=1) - z_values[i, 0])
            best_fit_idxs = np.argsort(r_dist * z_dist)[:args.plot_n_trajectories_per_sigma]
        elif args.pick_trajectories_on == 'extremes':
            # Pick trajectories which find the extremes
            r_dist = np.max(np.abs(Xs[i][:, :, 0]), axis=1)
            z_dist = np.max(np.abs(Xs[i][:, :, 2]), axis=1)

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
            traj = Xs[i][idx]
            trajectories[i, j] = traj
            x, y, z = traj[0].T
            path = mlab.plot3d(x, y, z, color=colour, tube_radius=0.08, figure=figs[i])
            paths[i].append(path)
        vols[i] = _plot_sphere_slice_border(0.1, 0.1, colour, n_points, alphas[i], fig=figs[i])

        # Build label overlays
        label_overlays[i] = _make_label_overlay(width / 2, height / n_sigmas, f'$\sigma$ = {sigma_labels[i]}')

    # Calculate the max r and z values over time
    logger.info('Calculating r and z values over time.')
    rs = np.maximum.accumulate(
        np.max(np.linalg.norm(trajectories[..., :2], axis=-1), axis=1),
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
