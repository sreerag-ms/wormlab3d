import os
from decimal import Decimal
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.stats import rv_continuous

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.two_state_explorer import TwoStateExplorer
from wormlab3d.particles.util import plot_msd
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args

plot_n_examples = 5
show_plots = False
save_plots = True
img_extension = 'png'


def _get_p_strs(pe: TwoStateExplorer, include_names: bool = True) -> List[str]:
    p_strs = []
    for name, params in {
        'Run durations': pe.run_duration_dist_params,
        'Planar angles': pe.planar_angles_dist_params,
        'Non-planar angles': pe.nonplanar_angles_dist_params,
    }.items():
        p_vals = []
        for p in params['params']:
            d = Decimal(str(p))
            p_vals.append(f'{p:.{-d.as_tuple().exponent}f}')

        p_str = '(' + ', '.join(p_vals) + ')'
        if include_names:
            p_str = f'{name}: ' + p_str
        p_strs.append(p_str)

    return p_strs


def _plot_pdfs(pe: TwoStateExplorer):
    p_strs = _get_p_strs(pe, include_names=False)

    def _plot_pdf(ax_, params: Dict[str, Any], label: str):
        # Plot the scipy distribution
        dist_cls: rv_continuous = getattr(scipy.stats, params['type'])
        dist = dist_cls(*params['params'])
        if 'Run duration' in label:
            x = np.linspace(-5, 100, 100)
        else:
            x = np.linspace(-np.pi, np.pi, 100)
        ax_.plot(x, dist.pdf(x), linestyle='--', alpha=0.9, label=label)

    fig, axes = plt.subplots(2, figsize=(12, 10))

    ax = axes[0]
    _plot_pdf(ax, pe.run_duration_dist_params, 'Run duration\n' + p_strs[0])
    ax.set_title('Run duration distribution')
    ax.legend()

    ax = axes[1]
    _plot_pdf(ax, pe.planar_angles_dist_params, 'Planar\n' + p_strs[1])
    _plot_pdf(ax, pe.nonplanar_angles_dist_params, 'Non-planar\n' + p_strs[2])
    ax.set_title('Angle distributions')
    ax.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_pdfs.{img_extension}')
    if show_plots:
        plt.show()


def _plot_simulation(
        ts: np.ndarray,
        tumble_ts: np.ndarray,
        X: np.ndarray,
        run_durations: np.ndarray,
        planar_angles: np.ndarray,
        nonplanar_angles: np.ndarray,
        run_idx: int = None
):
    """
    Plot the states, speeds and angles.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4)
    fig.suptitle(f'Simulation run {run_idx}')

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    # Histograms of the parameters
    for i, (param_name, param) in enumerate(
            {
                'Run durations': run_durations,
                'Planar angles': planar_angles,
                'Non-planar angles': nonplanar_angles,
            }.items()):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_title(param_name)
        ax.hist(param.numpy(), bins=20, density=True, facecolor='green', alpha=0.75)
        if param_name != 'Run durations':
            ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        # todo: fit distribution to data

    # Trace of the tumbles
    ax = fig.add_subplot(gs[0, 1:3])
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
    ax.legend()

    # 3D trajectory
    x, y, z = X.T
    ax = fig.add_subplot(gs[1:3, 1:3], projection='3d')
    ax.scatter(x, y, z, c=c, s=10, alpha=0.4, zorder=-1)
    points = X[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)
    equal_aspect_ratio(ax)

    # 2D trajectories
    for i, (view, (a, b)) in enumerate(
            {
                'xy': (x, y),
                'yz': (y, z),
                'xz': (x, z),
            }.items()):
        ax = fig.add_subplot(gs[i, 3])
        ax.set_title(view)
        ax.scatter(a, b, c=c, s=5, alpha=0.7)

    fig.tight_layout()
    return fig


def _plot_simulations(
        pe: TwoStateExplorer,
        ts: torch.Tensor,
        tumble_ts: torch.Tensor,
        Xs: torch.Tensor,
        run_durations: torch.Tensor,
        planar_angles: torch.Tensor,
        nonplanar_angles: torch.Tensor
):
    """
    Plot some simulation runs.
    """
    if plot_n_examples == 0:
        return

    for i in range(min(pe.batch_size, plot_n_examples)):
        title = f'Simulation run {i}.'
        logger.info(f'Plotting {title}')
        _plot_simulation(ts, tumble_ts[i], Xs[i], run_durations[i], planar_angles[i], nonplanar_angles[i], i)

        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_sim_{i}.{img_extension}')

        if show_plots:
            plt.show()


def _plot_histograms(
        pe: TwoStateExplorer,
        run_durations: torch.Tensor,
        planar_angles: torch.Tensor,
        nonplanar_angles: torch.Tensor
):
    """
    Plot histograms of the sampled parameters.
    """
    p_strs = _get_p_strs(pe, include_names=False)

    fig, axes = plt.subplots(3, figsize=(10, 10))

    ax = axes[0]
    ax.set_title(f'Run durations\n{p_strs[0]}')
    ax.hist(run_durations.numpy().flatten(), bins=30, density=True, facecolor='green', alpha=0.75)

    ax = axes[1]
    ax.set_title(f'Planar Angles\n{p_strs[1]}')
    ax.hist(planar_angles.numpy().flatten(), bins=31, density=True, facecolor='green', alpha=0.75)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])

    ax = axes[2]
    ax.set_title(f'Non-planar Angles\n{p_strs[2]}')
    ax.hist(nonplanar_angles.numpy().flatten(), bins=31, density=True, facecolor='green', alpha=0.75)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])

    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_histograms.{img_extension}')
    if show_plots:
        plt.show()


def _plot_msd(
        pe: TwoStateExplorer,
        Xs: torch.Tensor,
):
    """
    MSD plot against real trajectory.
    """
    args = get_args()
    X_real = get_trajectory_from_args(args)
    fig = plot_msd(args, X_real, Xs)
    fig.suptitle('\n'.join(_get_p_strs(pe)))
    fig.tight_layout()
    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_msd.{img_extension}')
    if show_plots:
        plt.show()


def simulate(batch_size: int):
    # For reference:
    # alpha, beta, loc, scale = params
    # (stability=alpha, skew=beta, loc=loc, scale=scale)

    pe = TwoStateExplorer(
        batch_size=batch_size,
        speed=0.11,
        run_duration_dist_params={
            'type': 'levy_stable',
            'params': (1.6, 1, 2, 2)
        },
        planar_angle_dist_params={
            'type': 'levy_stable',
            'params': (2, 0, 0, 2)
        },
        nonplanar_angle_dist_params={
            'type': 'levy_stable',
            # 'params': (0.1, 0, 0, 0.001)
            'params': (0.1, 0, 0, 0.1)
            # 'params': (0.5, 0, 0, 0.1)
        }
    )
    _plot_pdfs(pe)
    # exit()

    T = 20000 / 25
    dt = 1 / 25
    ts, tumble_ts, Xs, run_durations, planar_angles, nonplanar_angles = pe.forward(T, dt)

    _plot_simulations(pe, ts, tumble_ts, Xs, run_durations, planar_angles, nonplanar_angles)
    _plot_histograms(pe, run_durations, planar_angles, nonplanar_angles)
    _plot_msd(pe, Xs)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    simulate(batch_size=20)
