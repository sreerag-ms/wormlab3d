import os
import shutil
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from mayavi import mlab

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.args.parameter_args import ParameterArgs
from wormlab3d.particles.cache import get_durations_from_args, get_npas_from_args, get_sim_state_from_args
from wormlab3d.toolkit.plot_utils import make_cuboid
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'png'


def _init():
    """
    Initialise the arguments and save dir.
    """
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=False,
        include_K_options=False,
        include_planarity_options=True,
        include_helicity_options=False,
        include_manoeuvre_options=True,
        include_approximation_options=True,
        include_pe_options=True,
        include_fractal_dim_options=True,
        include_video_options=False,
        include_evolution_options=True,
        validate_source=False,
    )

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            if k in ['theta_dist_params', 'phi_dist_params']:
                v = [float(vv) for vv in v.split(',')]
            setattr(args, k, v)
    print_args(args)

    assert args.batch_size is not None, 'Must provide a batch size!'
    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Ensure that the approx_args is set for the run-tumble model if required
    ParameterArgs.from_args(args)

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    return save_dir, args


def _plot_trajectory(
        X: np.ndarray,
        alpha: float = 1.,
        c=None
):
    x, y, z = X.T
    t = np.linspace(0, 1, len(X))
    pts = mlab.plot3d(x, y, z, t, opacity=alpha, color=c, tube_radius=0.04)
    if c is None:
        cmap = plt.get_cmap('viridis_r')
        cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
        pts.module_manager.scalar_lut_manager.lut.table = cmaplist

    return pts


def _calculate_cuboid_dimensions(Xt: np.ndarray) -> np.ndarray:
    """
    Calculate the volumes of the cuboids explored by the trajectories.
    """
    ptp = np.ptp(Xt, axis=1)
    mean_dims = np.mean(ptp, axis=0)
    min_dims = np.min(ptp, axis=0)
    max_dims = np.max(ptp, axis=0)
    return np.array([mean_dims, min_dims, max_dims])


def cuboid_volumes_plot(
        k_idxs: List[int],
        n_trajectories_per_k: int = 3,
):
    """
    3D plot of cuboid volumes explored by different trajectories.
    """
    save_dir, args = _init()
    args.sim_durations = get_durations_from_args(args)
    args.pauses = [args.nonp_pause_max]  # Fix the pause duration
    args.npas = get_npas_from_args(args)  # Propagate the npas (phi factors)

    # Select the phi factors to use
    k_factors = args.npas[k_idxs]

    # Set up the plot
    mlab.options.offscreen = save_plots
    fig = mlab.figure(size=(1500, 1000), bgcolor=(1, 1, 1))
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Cuboid and trajectory colours
    colours = np.array([
        [210, 140, 190, 1],  # pink
        [220, 230, 235, 1],  # white
        [30, 73, 21, 1],  # green
    ]) / 255
    alphas = np.array([0.2, 0.7, 0.5])

    # For each factor, plot trajectories and cuboids
    dims = []
    for i in range(3):
        if i == 0:
            args.phi_factor_rt = k_factors[0]
        elif i == 1:
            args.phi_factor_rt = k_factors[2]
        elif i == 2:
            args.phi_factor_rt = k_factors[1]
        SS = get_sim_state_from_args(args, read_only=True)

        # Select some exemplar trajectories
        Xt = SS.get_Xt()
        ptp = np.ptp(Xt, axis=1)

        # Pick the trajectories with the largest ptp in the x and y-dimensions
        if i == 0:
            # traj_idxs = np.argsort(ptp[:, 0] * ptp[:, 1])[::-1][:5]
            traj_idxs = np.argsort(ptp[:, 2])[:5]

        # Pick the trajectories with the largest ptp in the z-dimension
        elif i == 1:
            traj_idxs = np.argsort(ptp[:, 2])[::-1][:5]

        # Pick some trajectories in between the two z extremes
        else:
            target_z = (dims[0][2] + dims[1][2]) / 2
            traj_idxs = np.argsort(np.abs(ptp[:, 2] - target_z))[:5]

        Xt = Xt[traj_idxs]

        # Calculate the cuboid dimensions
        dims_i = _calculate_cuboid_dimensions(Xt)[0]  # Use mean
        logger.info(f'Mean cuboid dimensions for factor={args.phi_factor_rt:.3E}: [' + ', '.join(
            [f'{d:.2f}' for d in dims_i]) + ']')
        dims.append(dims_i)

        # Draw the trajectories
        colour = tuple(colours[i][:3])
        for X in Xt:
            mp = np.ptp(X, axis=0) / 2
            X = X - X.min(axis=0) - mp
            _plot_trajectory(X, alpha=alphas[i] + 0.2, c=colour)
        make_cuboid(
            dims=dims_i,
            colour=colour,
            opacity=alphas[i],
            draw_outline=True,
            outline_colour=colour,
            outline_opacity=alphas[i] + 0.2,
            outline_tube_radius=0.02,
            fig=fig
        )

    # # Set view
    # engine = mlab.get_engine()
    # scene = engine.scenes[0]
    # scene.scene.camera.position = [28.563307971895117, -18.171434465726993, 8.876904873887577]
    # scene.scene.camera.focal_point = [-0.007448673248291016, -0.011125564575195312, -0.006509184837341309]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.21577550735433665, 0.133676574011645, 0.9672494528230491]
    # scene.scene.camera.clipping_range = [4.73000651267882, 73.18810737976897]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    # Draw plot
    mlab.view(
        figure=fig,
        distance=35,
        # focalpoint=centre,
        azimuth=-30,
        elevation=75,
        roll=-80,
    )

    if save_plots:
        fig.scene._lift()
        img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
        img.save(save_dir / (
                f'factors=[' + ','.join([f'{f:.3E}' for f in k_factors]) + f']' +
                f'_ntpk={n_trajectories_per_k}.png'))
        mlab.clf(fig)
        mlab.close()

    if show_plots:
        mlab.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    use_k_idxs = [0, 9, 19]
    cuboid_volumes_plot(k_idxs=use_k_idxs, n_trajectories_per_k=3)
