import os
from decimal import Decimal
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.stats
import scipy.stats
import torch
from matplotlib.gridspec import GridSpec
from scipy.stats import rv_continuous

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.three_state_explorer import ThreeStateExplorer
from wormlab3d.particles.util import plot_msd
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args

plot_n_examples = 5
show_plots = False
save_plots = True
img_extension = 'png'


def _get_p_strs(pe: ThreeStateExplorer, include_names: bool = True) -> List[str]:
    p_strs = []
    for name, params in {
        'Planar angles': pe.planar_angles_dist_params,
        'Non-planar angles': pe.nonplanar_angles_dist_params,
    }.items():
        p_vals = []
        for p in params['params']:
            d = Decimal(str(p))
            p_vals.append(f'{p:.{-d.as_tuple().exponent}f}')

        p_str = params['type'] + '(' + ', '.join(p_vals) + ')'
        if include_names:
            p_str = f'{name}: ' + p_str
        p_strs.append(p_str)

    return p_strs


def _plot_angle_pdfs(pe: ThreeStateExplorer):
    """
    Plot the angle distribution functions.
    """
    p_strs = _get_p_strs(pe, include_names=False)

    def _plot_pdf(ax_, params: Dict[str, Any], label: str):
        # Plot the scipy distribution
        dist_cls: rv_continuous = getattr(scipy.stats, params['type'])
        dist = dist_cls(*params['params'])
        x = np.linspace(-np.pi, np.pi, 100)
        ax_.plot(x, dist.pdf(x), linestyle='--', alpha=0.9, label=label)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    _plot_pdf(ax, pe.planar_angles_dist_params, 'Planar\n' + p_strs[0])
    _plot_pdf(ax, pe.nonplanar_angles_dist_params, 'Non-planar\n' + p_strs[1])
    ax.set_title('Angle distributions')
    ax.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_angle_pdfs.{img_extension}')
    if show_plots:
        plt.show()


def _plot_simulation(
        ts: torch.Tensor,
        tumble_ts: torch.Tensor,
        X: torch.Tensor,
        s0_durations: torch.Tensor,
        s1_durations: torch.Tensor,
        planar_angles: torch.Tensor,
        nonplanar_angles: torch.Tensor,
        intervals: torch.Tensor,
        run_idx: int = None
):
    """
    Plot a simulation output.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(5, 5)
    fig.suptitle(f'Simulation run {run_idx}')

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    # Trace of the tumbles
    ax = fig.add_subplot(gs[0, :])
    ax.set_title('Tumble trace')
    ax.axhline(y=0, color='darkgrey')
    ax.scatter(tumble_ts, planar_angles, label='$\\theta_P$', marker='x')
    ax.scatter(tumble_ts, nonplanar_angles, label='$\\theta_{NP}$', marker='o')
    ax.set_xlim(left=ts[0], right=ts[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\\theta$')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    for t in tumble_ts:
        ax.axvline(x=t, color='pink', zorder=-1, alpha=0.4)
    ax.legend(loc=1)

    # Intervals
    ax2 = ax.twinx()
    interval_marker_ts = (tumble_ts[1:] + tumble_ts[:-1]) / 2
    # ax2.scatter(interval_marker_ts, intervals, c='green')
    ax2.plot(interval_marker_ts, intervals, c='green', marker='+', alpha=0.5)
    ax2.set_ylabel('Interval (s)')

    # Histograms of the parameters
    for i, (param_name, param) in enumerate(
            {
                'Run durations': intervals,
                'State0 durations': s0_durations,
                'State1 durations': s1_durations,
                'Planar angles': planar_angles,
                'Non-planar angles': nonplanar_angles,
            }.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(param_name)
        ax.hist(param.numpy(), bins=21, density=True, facecolor='green', alpha=0.75)
        ax.set_yscale('log')
        if param_name in ['Planar angles', 'Non-planar angles']:
            ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        # todo: fit distribution to data

    # 3D trajectory
    x, y, z = X.T
    ax = fig.add_subplot(gs[2:5, :4], projection='3d')
    ax.scatter(x, y, z, c=c, s=10, alpha=0.4, zorder=-1)
    # points = X[:, None, :]
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    # ax.add_collection(lc)
    equal_aspect_ratio(ax)

    # 2D trajectories
    for i, (view, (a, b)) in enumerate(
            {
                'xy': (x, y),
                'yz': (y, z),
                'xz': (x, z),
            }.items()):
        ax = fig.add_subplot(gs[2 + i, 4])
        ax.set_title(view)
        ax.scatter(a, b, c=c, s=5, alpha=0.7)

    fig.tight_layout()
    return fig


def _plot_simulations(
        pe: ThreeStateExplorer,
        ts: torch.Tensor,
        tumble_ts: List[torch.Tensor],
        Xs: torch.Tensor,
        durations: Dict[int, List[torch.Tensor]],
        planar_angles: torch.Tensor,
        nonplanar_angles: torch.Tensor,
        intervals: List[torch.Tensor],
):
    """
    Plot some simulation runs.
    """
    if plot_n_examples == 0:
        return

    for i in range(min(pe.batch_size, plot_n_examples)):
        title = f'Simulation run {i}.'
        logger.info(f'Plotting {title}')
        n_tumbles = len(tumble_ts[i])
        _plot_simulation(
            ts,
            tumble_ts[i],
            Xs[i],
            durations[0][i],
            durations[1][i],
            planar_angles[i, :n_tumbles],
            nonplanar_angles[i, :n_tumbles],
            intervals[i],
            i
        )

        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_sim_{i}.{img_extension}')

        if show_plots:
            plt.show()


def _plot_histograms(
        pe: ThreeStateExplorer,
        durations: Dict[int, List[torch.Tensor]],
        planar_angles: torch.Tensor,
        nonplanar_angles: torch.Tensor,
        intervals: List[torch.Tensor]
):
    """
    Plot histograms of the sampled parameters.
    """
    logger.info('Plotting histograms.')
    p_strs = _get_p_strs(pe, include_names=False)

    fig, axes = plt.subplots(5, figsize=(10, 10))

    ax = axes[0]
    ax.set_title(f'Run durations (intervals between tumbles)')
    dr = torch.cat(intervals).numpy()
    ax.hist(dr, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[1]
    ax.set_title(f'State0 durations')
    d0 = torch.cat(durations[0]).numpy()
    ax.hist(d0, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[2]
    ax.set_title(f'State1 durations')
    d1 = torch.cat(durations[1]).numpy()
    ax.hist(d1, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[3]
    ax.set_title(f'Planar Angles\n{p_strs[0]}')
    ax.hist(planar_angles.numpy().flatten(), bins=31, density=True, facecolor='green', alpha=0.75)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax.set_yscale('log')

    ax = axes[4]
    ax.set_title(f'Non-planar Angles\n{p_strs[1]}')
    ax.hist(nonplanar_angles.numpy().flatten(), bins=31, density=True, facecolor='green',
            alpha=0.75)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_histograms.{img_extension}')
    if show_plots:
        plt.show()


def _plot_msd(
        pe: ThreeStateExplorer,
        Xs: torch.Tensor,
):
    """
    MSD plot against real trajectory.
    """
    args = get_args()
    trial_ids = args.trials if args.trials is not None else [args.trial, ]
    Xs_real = []
    for trial_id in trial_ids:
        args.trial = trial_id
        X_real = get_trajectory_from_args(args)
        Xs_real.append(X_real)
    fig = plot_msd(args, Xs_real, Xs)
    fig.suptitle(
        f'$r_{{01}}={pe.rate_01}$, $r_{{10}}={pe.rate_10}$, $r_{{02}}={pe.rate_02}$, $r_{{20}}={pe.rate_20}$\n'
        + f'$s_0={pe.speed_0}$, $s_1={pe.speed_1}$\n'
        + '\n'.join(_get_p_strs(pe))
    )
    fig.tight_layout()
    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_msd.{img_extension}')
    if show_plots:
        plt.show()


def simulate(batch_size: int):
    T = 50000 / 25
    dt = 1 / 25

    pe = ThreeStateExplorer(
        batch_size=batch_size,
        rate_01=0.01,
        rate_10=0.01,
        rate_02=0.05,
        rate_20=0.9,  # not really a rate!
        speed_0=0.002,
        speed_1=0.004,
        planar_angle_dist_params={
            'type': 'levy_stable',
            'params': (2, 0, 0, 2)
        },
        nonplanar_angle_dist_params={
            'type': 'levy_stable',
            'params': (0.2, 0, 0, 0.1)
        }
    )
    # _plot_angle_pdfs(pe)

    ts, tumble_ts, Xs, states, durations, planar_angles, nonplanar_angles, intervals = pe.forward(T, dt)

    _plot_simulations(pe, ts, tumble_ts, Xs, durations, planar_angles, nonplanar_angles, intervals)
    _plot_histograms(pe, durations, planar_angles, nonplanar_angles, intervals)
    _plot_msd(pe, Xs)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    simulate(batch_size=50)
