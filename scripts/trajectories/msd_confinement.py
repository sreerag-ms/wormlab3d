import itertools
import os
import random
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mayavi import mlab

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger, N_WORKERS
from wormlab3d.data.model import Trial
from wormlab3d.toolkit.plot_utils import make_box_outline, to_rgb
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.brownian_particle import BoundedParticle
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.displacement import calculate_displacements_parallel, DISPLACEMENT_AGGREGATION_SQUARED_SUM, \
    calculate_displacements
from wormlab3d.trajectories.util import DEFAULT_FPS, get_deltas_from_args

show_plots = False
save_plots = True
img_extension = 'svg'

# Off-screen rendering
mlab.options.offscreen = save_plots


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot confinement effects.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    parser.add_argument('--trials', type=lambda s: [int(item) for item in s.split(',')], required=True,
                        help='List of trial ids to show.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    # MSD options
    parser.add_argument('--min-delta', type=int, default=1, help='Minimum time lag.')
    parser.add_argument('--max-delta', type=int, default=10000, help='Maximum time lag.')
    parser.add_argument('--delta-step', type=float, default=1, help='Step between deltas. -ve=exponential steps.')

    # Model parameters
    parser.add_argument('--sim-time', type=int, default=600, help='Simulation time (seconds).')
    parser.add_argument('--diffusion', type=float, default=1e-3, help='Diffusion rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--box-size', type=float, default=10., help='Box size.')
    parser.add_argument('--n-runs', type=int, default=10, help='Number of simulation runs.')

    # 3D plots
    parser.add_argument('--width-3d', type=int, default=1000, help='Width of 3D plot (in pixels).')
    parser.add_argument('--height-3d', type=int, default=1000, help='Height of 3D plot (in pixels).')
    parser.add_argument('--distance', type=float, default=4., help='Camera distance (in mm).')
    parser.add_argument('--azimuth', type=int, default=70, help='Azimuth.')
    parser.add_argument('--elevation', type=int, default=45, help='Elevation.')
    parser.add_argument('--roll', type=int, default=45, help='Roll.')

    args = parser.parse_args()

    print_args(args)

    return args


def _set_seed(seed):
    """Set the random seed."""
    logger.info(f'Setting random seed = {seed}.')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def _get_msds_trials(
        args: Namespace,
) -> Tuple[List[Trial], Dict[int, Dict[int, List[float]]], Dict[int, List[float]]]:
    """
    MSD plot for all trials in a dataset.
    """
    deltas, delta_ts = get_deltas_from_args(args)

    # Calculate the displacements for all trials
    all_displacements = {delta: [] for delta in deltas}
    trial_displacements = {}
    trials = {}
    for i, trial_id in enumerate(args.trials):
        trial = Trial.objects.get(id=trial_id)
        logger.info(f'Calculating displacements for trial={trial.id}.')
        trials[trial.id] = trial
        trial_displacements[trial.id] = {}

        # Calculate displacements for trial
        X, _ = get_trajectory(
            trial_id=trial.id,
            smoothing_window=args.smoothing_window,
            tracking_only=True,
        )
        d = calculate_displacements_parallel(X, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM)
        for delta in deltas:
            trial_displacements[trial.id][delta] = d[delta]
            all_displacements[delta].extend(d[delta])

    # Calculate msds
    msds = {}
    for trial_id, t_displacements in trial_displacements.items():
        msds[trial_id] = {
            delta: np.mean(t_displacements[delta])
            for delta in deltas
        }
    msds_all_traj = {
        delta: np.mean(all_displacements[delta])
        for delta in deltas
    }

    return trials, msds, msds_all_traj


def _calculate_sim_msd(
        D: float,
        momentum: float,
        bounds: np.ndarray,
        sim_time: float,
        deltas: np.ndarray,
):
    """
    Generate a simulation trajectory and calculate MSD.
    """
    bs = np.ptp(bounds[0])
    x0 = np.clip(np.random.normal(np.zeros(3), bs / 3), a_min=-bs / 2 + 0.01, a_max=bs / 2 - 0.01)
    p = BoundedParticle(
        x0=x0,
        D=D,
        momentum=momentum,
        bounds=bounds
    )
    X = p.generate_trajectory(n_steps=sim_time * DEFAULT_FPS, total_time=sim_time)
    d = calculate_displacements(X, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM, quiet=True)

    return d


def _calculate_sim_msd_wrapper(args):
    return _calculate_sim_msd(*args)


def _get_msds_bounded_particles(
        args: Namespace,
) -> Tuple[Dict[int, Dict[int, List[float]]], Dict[int, List[float]]]:
    """
    Generate bounded particle trajectories and calculate the msds.
    """
    bounds = np.array([[-args.box_size / 2, args.box_size / 2]] * 3)
    deltas, delta_ts = get_deltas_from_args(args)

    logger.info('Simulating trajectories in parallel.')
    sim_args = dict(
        D=args.diffusion,
        momentum=args.momentum,
        bounds=bounds,
        sim_time=args.sim_time,
        deltas=deltas,
    )
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            _calculate_sim_msd_wrapper,
            [list(sim_args.values()) for _ in range(args.n_runs)]
        )
    sim_displacements = {i: res for i, res in enumerate(res)}
    all_displacements = {
        delta: np.concatenate([sd[delta] for sd in sim_displacements.values()])
        for delta in deltas
    }

    # Calculate msds
    msds = {}
    for i, s_displacements in sim_displacements.items():
        msds[i] = {
            delta: np.mean(s_displacements[delta])
            for delta in deltas
        }
    msds_all_sim = {
        delta: np.mean(all_displacements[delta])
        for delta in deltas
    }

    return msds, msds_all_sim


def plot_confinement():
    """
    Plot the confinement effects.
    """
    args = parse_args()
    _set_seed(args.seed)
    deltas, delta_ts = get_deltas_from_args(args)
    trials, msds, msds_all_traj = _get_msds_trials(args)
    msds_sim, msds_all_sim = _get_msds_bounded_particles(args)

    # Sort the trials by concentration
    trial_ids = list(trials.keys())
    trial_ids.sort(key=lambda tid: trials[tid].duration)
    trial_ids.sort(key=lambda tid: trials[tid].experiment.concentration)

    # Set up plots and colours
    plt.rc('axes', titlesize=9)  # fontsize of the title
    plt.rc('axes', labelsize=9, labelpad=1)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=7)  # fontsize of the legend
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), gridspec_kw={
        'wspace': 1,
        'top': 0.99,
        'bottom': 0.15,
        'left': 0.08,
        'right': 0.99,
    })

    def _complete_msd_plot(ax_):
        ax_.set_ylabel('MSD')
        ax_.set_xlabel('$\Delta\ (s)$')
        ax_.set_yscale('log')
        ax_.set_xscale('log')
        ax_.grid()

        # Highlight flattening-off region
        ylim = ax_.get_ylim()[1] * 1.1
        ax_.set_ylim(top=ylim)
        ax_.fill_between(np.arange(30, 100), ylim, color='red', alpha=0.3, zorder=-1, linewidth=0)

    # Plot the real data
    cmap = plt.get_cmap('brg')
    colours = cmap(np.linspace(0, 1, len(msds)))
    ax = axes[0]
    msd_vals_all_traj = np.array(list(msds_all_traj.values()))
    ax.plot(delta_ts, msd_vals_all_traj, label='Average',
            alpha=0.8, c='black', linestyle='--', linewidth=3, zorder=80)
    for i, trial_id in enumerate(trial_ids):
        msd_vals = np.array(list(msds[trial_id].values()))
        ax.plot(
            delta_ts,
            msd_vals,
            label=f'{trials[trial_id].experiment.concentration:.2f}% '
                  f'({trials[trial_id].duration:%M:%S})',
            alpha=0.5,
            c=colours[i]
        )
    _complete_msd_plot(ax)
    ax.legend(bbox_to_anchor=(1.04, 1))

    # Plot the simulation results
    cmap = plt.get_cmap('autumn')
    colours = cmap(np.linspace(0, 1, len(msds_sim)))
    ax = axes[1]
    msd_vals_all_sim = np.array(list(msds_all_sim.values()))
    ax.plot(delta_ts, msd_vals_all_sim, label='Average',
            alpha=0.8, c='black', linestyle='--', linewidth=3, zorder=80)
    for i in range(len(msds_sim)):
        msd_vals = np.array(list(msds_sim[i].values()))
        ax.plot(
            delta_ts,
            msd_vals,
            alpha=0.5,
            c=colours[i]
        )
    ax.set_ylim(axes[0].get_ylim())
    _complete_msd_plot(ax)

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_msds' \
                           f'_t={",".join(str(t) for t in args.trials)}' \
                           f'_sw={args.smoothing_window}' \
                           f'_d={args.min_delta}-{args.max_delta}_s{args.delta_step}' \
                           f'_st={args.sim_time}' \
                           f'_D={args.diffusion}' \
                           f'_m={args.momentum}' \
                           f'_bs={args.box_size}' \
                           f'_n={args.n_runs}' \
                           f'.{img_extension}'
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_sim_trajectory_3d():
    """
    Generate and plot trajectory of a bounded randomly generated brownian particle with momentum.
    """
    args = parse_args()
    _set_seed(args.seed)
    args.sim_time = 120
    bs = args.box_size
    bounds = np.array([[-bs / 2, bs / 2]] * 3)

    logger.info('Simulating trajectory.')
    x0 = np.clip(np.random.normal(np.zeros(3), bs / 3), a_min=-bs / 2 + 0.01, a_max=bs / 2 - 0.01)
    p = BoundedParticle(
        x0=x0,
        D=args.diffusion,
        momentum=args.momentum,
        bounds=bounds
    )
    X = p.generate_trajectory(n_steps=args.sim_time * DEFAULT_FPS, total_time=args.sim_time)

    # Set up mlab figure
    fig = mlab.figure(size=(args.width_3d, args.height_3d), bgcolor=(1, 1, 1))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 64
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    s = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
    path = mlab.plot3d(*X.T, s, opacity=0.6, tube_radius=0.025)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Add outline box aligned with PCA components
    bound_points = np.array(list(itertools.product(*[[-bs / 2, bs / 2]] * 3)))
    lines = make_box_outline(X=bound_points, use_extents=True)
    for l in lines:
        mlab.plot3d(
            *l.T,
            figure=fig,
            color=to_rgb('darkgrey'),
            tube_radius=0.005,
        )

    # Draw plot
    mlab.view(
        figure=fig,
        distance=bs * 3.3,
        focalpoint=(0, 0, 0),
        azimuth=args.azimuth,
        elevation=args.elevation,
        roll=args.roll,
    )

    # # Useful for getting the view parameters when recording from the gui:
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [6.650943730427842, 2.5356338312985516, -2.6447065760118273]
    # scene.scene.camera.focal_point = [-0.001649396310300033, 0.001798197145702285, 0.002302323061598943]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.32633346022468956, -0.1223746264199722, -0.9372998045163315]
    # scene.scene.camera.clipping_range = [4.384672931447044, 11.655698783828257]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_3d_sim' \
                           f'_st={args.sim_time}' \
                           f'_D={args.diffusion}' \
                           f'_m={args.momentum}' \
                           f'_bs={args.box_size}' \
                           f'.png'
        logger.info(f'Saving 3D plot to {path}.')
        fig.scene._lift()
        img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
        img.save(path)
        mlab.clf(fig)
        mlab.close()
    else:
        mlab.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    plot_confinement()
    # plot_sim_trajectory_3d()

    # trials=162,73,114,103,76,35,168,37
