import os
from argparse import Namespace
from decimal import Decimal
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.signal import find_peaks
from scipy.stats import rv_continuous
from sklearn.decomposition import PCA

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.cache import get_trajectories_from_args, TrajectoryCache
from wormlab3d.particles.gaussian_mixture import GaussianMixtureScipy
from wormlab3d.particles.three_state_explorer import ThreeStateExplorer
from wormlab3d.particles.util import plot_msd
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca
from wormlab3d.toolkit.util import normalise
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.displacement import calculate_displacements
from wormlab3d.trajectories.util import smooth_trajectory

plot_n_examples = 20
show_plots = True
save_plots = True
img_extension = 'svg'


def _get_p_strs(pe: ThreeStateExplorer, include_names: bool = True) -> List[str]:
    p_strs = []
    for name, params in {
        'Planar angles': pe.theta_dist_params,
        'Non-planar angles': pe.phi_dist_params,
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

    def _plot_pdf(ax_, params: Dict[str, Any], label: str, colour: str = None, zorder: int = None):
        # Plot the scipy distribution
        if params['type'] == '2norm':
            w1, mu1, sigma1 = params['params'][:3]
            w2, mu2, sigma2 = params['params'][3:]
            dist = GaussianMixtureScipy(
                np.array([w1, w2]),
                np.array([mu1, mu2]),
                np.array([sigma1, sigma2])
            )
        else:
            dist_cls: rv_continuous = getattr(scipy.stats, params['type'])
            dist = dist_cls(*params['params'])

        N = 501
        vals = np.zeros(N)
        for i in range(-9, 8, 2):
            x = np.linspace(i * np.pi, (i + 2) * np.pi, N)
            vals += dist.pdf(x)

        print(vals.sum() * 2 * np.pi / N)

        # x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 501)
        # ax_.plot(x, dist.pdf(x), linestyle='--', alpha=0.9, label=label, color=colour)
        # ax_.plot(x, dist.pdf(x), alpha=0.9, label=label, color=colour)
        x = np.linspace(-np.pi, np.pi, N)
        return ax_.plot(x, vals, alpha=0.9, label=label, color=colour, zorder=zorder)

    # fig, ax = plt.subplots(1, figsize=(8, 6))
    # _plot_pdf(ax, pe.planar_angles_dist_params, 'Planar\n' + p_strs[0])
    # _plot_pdf(ax, pe.nonplanar_angles_dist_params, 'Non-planar\n' + p_strs[1])
    fig, ax = plt.subplots(1, figsize=(1.7, 1.4))
    l1 = _plot_pdf(ax, pe.theta_dist_params, '$\\theta$', colour='red', zorder=2)
    t_samples = pe.theta_dist.sample((2000,)).squeeze()
    t_samples = torch.atan2(torch.sin(t_samples), torch.cos(t_samples)).numpy()
    ax.hist(t_samples, bins=51, density=True, facecolor='pink', alpha=0.5)
    l2 = _plot_pdf(ax, pe.phi_dist_params, '$\psi$', colour='green', zorder=1)
    p_samples = pe.phi_dist.sample((2000,)).squeeze()
    p_samples = torch.atan(torch.tan(p_samples)).numpy()
    ax.hist(p_samples, bins=25, density=True, facecolor='lightgreen', alpha=0.5)

    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, np.pi])
    ax.set_xticklabels(['$-\pi$', '$\pi$'])
    # ax.set_xticks([-np.pi, 0, np.pi])
    # ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])

    # ax.set_yticks([])
    ax.set_yticks([0.6])
    ax.set_yticklabels([0.6])

    # Pauses
    phis = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    if pe.nonp_pause_type == 'linear':
        pauses = (np.abs(phis) / (np.pi / 2)) * pe.nonp_pause_max
    elif pe.nonp_pause_type == 'quadratic':
        pauses = (np.abs(phis) / (np.pi / 2))**2 * pe.nonp_pause_max
    else:
        raise RuntimeError(f'Unsupported pause type: {pe.nonp_pause_type}.')
    ax2 = ax.twinx()
    l3 = ax2.plot(phis, pauses, label='$\delta(\psi)$')
    ax2.set_ylim(bottom=0, top=pe.nonp_pause_max + 0.1)
    ax2.set_yticks([pe.nonp_pause_max])
    ax2.set_yticklabels([int(pe.nonp_pause_max)])

    # ax.set_title('Angle distributions')
    # added these three lines
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_angle_pdfs.{img_extension}', transparent=True)
    if show_plots:
        plt.show()


def _plot_pause_relation(pe: ThreeStateExplorer):
    """
    Plot the relationship between non-planar angle and pause duration.
    """
    fig, ax = plt.subplots(1, figsize=(1, 1))
    phis = np.linspace(-np.pi / 2, np.pi / 2, 1000)

    if pe.nonp_pause_type == 'linear':
        pauses = (np.abs(phis) / (np.pi / 2)) * pe.nonp_pause_max
    elif pe.nonp_pause_type == 'quadratic':
        pauses = (np.abs(phis) / (np.pi / 2))**2 * pe.nonp_pause_max
    else:
        raise RuntimeError(f'Unsupported pause type: {pe.nonp_pause_type}.')

    ax.plot(phis, pauses)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, np.pi])
    ax.set_xticklabels(['$-\pi$', '$\pi$'])
    ax.set_yticks([])
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_pause_rel.{img_extension}', transparent=True)
    if show_plots:
        plt.show()


def _plot_simulation(
        TC: TrajectoryCache,
        run_idx: int = 0
):
    """
    Plot a simulation output.
    """
    ts = TC.ts
    tumble_ts = TC.tumble_ts[run_idx]
    X = TC.X[run_idx]
    s0_durations = TC.durations[0][run_idx]
    s1_durations = TC.durations[1][run_idx]
    thetas = TC.thetas[run_idx]
    phis = TC.phis[run_idx]
    intervals = TC.intervals[run_idx]

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
    ax.scatter(tumble_ts, thetas, label='$\\theta_P$', marker='x')
    ax.scatter(tumble_ts, phis, label='$\\theta_{NP}$', marker='o')
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
                'Planar angles': thetas,
                'Non-planar angles': phis,
            }.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(param_name)
        ax.hist(param, bins=21, density=True, facecolor='green', alpha=0.75)
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
        TC: TrajectoryCache,
):
    """
    Plot some simulation runs.
    """
    if plot_n_examples == 0:
        return

    for i in range(min(pe.batch_size, plot_n_examples)):
        title = f'Simulation run {i}.'
        logger.info(f'Plotting {title}')
        _plot_simulation(TC, i)

        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_sim_{i}.{img_extension}')

        if show_plots:
            plt.show()


def _plot_histograms(
        pe: ThreeStateExplorer,
        TC: TrajectoryCache,
):
    """
    Plot histograms of the sampled parameters.
    """
    logger.info('Plotting histograms.')
    p_strs = _get_p_strs(pe, include_names=False)

    fig, axes = plt.subplots(7, figsize=(10, 14))

    ax = axes[0]
    ax.set_title(f'Run durations (intervals between tumbles)')
    dr = np.concatenate(TC.intervals)
    ax.hist(dr, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[1]
    ax.set_title(f'State0 durations')
    d0 = np.concatenate(TC.durations[0])
    ax.hist(d0, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[2]
    ax.set_title(f'State1 durations')
    d1 = np.concatenate(TC.durations[1])
    ax.hist(d1, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[3]
    ax.set_title(f'Speeds')
    s = np.concatenate(TC.speeds)
    ax.hist(s, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[4]
    ax.set_title(f'Speeds (weighted)')
    ax.hist(s, weights=dr, bins=30, density=True, facecolor='green', alpha=0.75)
    ax.set_yscale('log')

    ax = axes[5]
    ax.set_title(f'$\\theta$ (Planar Angles)\n{p_strs[0]}')
    thetas = np.concatenate(TC.thetas)
    ax.hist(thetas, bins=31, density=True, facecolor='green', alpha=0.75)
    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax.set_yscale('log')

    ax = axes[6]
    ax.set_title(f'$\\phi$ (Non-planar Angles)\n{p_strs[1]}')
    phis = np.concatenate(TC.phis)
    ax.hist(phis, bins=31, density=True, facecolor='green', alpha=0.75)
    ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
    ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
    ax.set_xticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])
    ax.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_histograms.{img_extension}')
    if show_plots:
        plt.show()


def _plot_msd(
        args: Namespace,
        pe: ThreeStateExplorer,
        TC: TrajectoryCache
):
    """
    MSD plot against real trajectory.
    """
    Xs = TC.X
    trial_ids = args.trials if args.trials is not None else [args.trial, ]

    max_radius = 0
    Xs_real = []
    for trial_id in trial_ids:
        args.trial = trial_id
        X_real = get_trajectory_from_args(args)
        X_real -= X_real.mean(axis=0)
        dists = np.linalg.norm(X_real, axis=-1)
        max_radius = max(dists.max(), max_radius)
        Xs_real.append(X_real)

    logger.info(f'Maximum explored distance from centre of mass of trajectory = {max_radius:.2f}.')

    confinement_radius = max_radius / 2
    Xs_confined = []
    for X in Xs:
        pos_offset = normalise(X.mean(axis=0)) * confinement_radius / 2
        X = X - pos_offset

        dists = np.linalg.norm(X, axis=-1)
        oob_idxs = (dists > confinement_radius).nonzero()[0]
        if len(oob_idxs) > 0:
            cut_idx = oob_idxs[0]
            if cut_idx < 6000:  # discard trajectories which exited the area too quickly
                continue
            Xc = X[:cut_idx]
        else:
            Xc = X
        Xs_confined.append(Xc)
    if len(Xs_confined) == 0:
        raise RuntimeError('All simulations exited the confinement area too quickly!')
    logger.info(f'{len(Xs_confined)} simulations remained in the confinement area long enough.')

    fig = plot_msd(args, Xs_real, Xs_confined)
    fig.suptitle(
        f'$r_{{01}}={pe.rate_01}$, $r_{{10}}={pe.rate_10}$, $r_{{02}}={pe.rate_02}$, $r_{{20}}={pe.rate_20}$\n'
        # + f'$s_0={pe.speed_0}$, $s_1={pe.speed_1}$\n'
        + '\n'.join(_get_p_strs(pe))
    )
    fig.tight_layout()
    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_msd.{img_extension}')
    if show_plots:
        plt.show()


def _plot_trajectories(
        pe: ThreeStateExplorer,
        TC: TrajectoryCache,
):
    """
    Plot some simulation trajectories.
    """
    if plot_n_examples == 0:
        return

    # Construct colours
    colours = np.linspace(0, 1, TC.X.shape[1])
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    for i in range(min(pe.batch_size, plot_n_examples)):
        logger.info(f'Plotting sim run {i}.')

        # Plot the trajectory
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        x, y, z = TC.X[i].T
        ax.scatter(x, y, z, c=c, s=100, alpha=1, zorder=1)
        equal_aspect_ratio(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis('off')
        fig.tight_layout()

        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_sim_{i}.{img_extension}', transparent=True)
            np.savez(LOGS_PATH / f'{START_TIMESTAMP}_sim_{i}', X=TC.X[i])

        if show_plots:
            plt.show()


def simulate():
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=True,
        include_K_options=False,
        include_planarity_options=False,
        include_manoeuvre_options=False,
        validate_source=False,
        include_pe_options=True
    )
    pe, TC = get_trajectories_from_args(args)
    _plot_angle_pdfs(pe)
    _plot_pause_relation(pe)
    exit()
    _plot_histograms(pe, TC)
    _plot_msd(args, pe, TC)
    _plot_simulations(pe, TC)
    _plot_trajectories(pe, TC)


def _check_trajectory_for_nice_regions(
        X: np.ndarray,
        timescale: int,
        reject_s2_threshold: float
):
    """
    Check the trajectory to see if it has nice regions.
    """

    # Find regions from displacement
    d = calculate_displacements(X, deltas=timescale)
    d_mean = d.mean()
    state_up = np.r_[[0, ], d > d_mean, [0, ]]
    up_idxs, up_section_props = find_peaks(state_up, height=0.5, distance=timescale / 10, width=timescale / 4)
    state_down = np.r_[[0, ], d < d_mean, [0, ]]
    down_idxs, down_section_props = find_peaks(state_down, height=0.5, distance=timescale / 10, width=timescale / 4)

    # Plot the regions as boxes aligned with the trajectory
    for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
        for i in range(len(props['left_bases'])):
            X_region = X[props['left_bases'][i]:props['right_bases'][i] - 1]
            pca = PCA()
            pca.fit(X_region)
            if colour == 'orange' and pca.explained_variance_ratio_[2] < reject_s2_threshold:
                return False

    return True


def _get_niceness_scores(
        Xs: np.ndarray,
        timescale: int,
):
    """
    Score the trajectories on the niceness of the regions.
    """
    # nonp = np.zeros(len(Xs))
    scores = np.zeros(len(Xs))

    for i, X in enumerate(Xs):
        # Find regions from displacement
        d = calculate_displacements(X, deltas=timescale)
        d_mean = d.mean()
        state_up = np.r_[[0, ], d > d_mean, [0, ]]
        up_idxs, up_section_props = find_peaks(state_up, height=0.5, distance=timescale / 10, width=timescale / 4)
        state_down = np.r_[[0, ], d < d_mean, [0, ]]
        down_idxs, down_section_props = find_peaks(state_down, height=0.5, distance=timescale / 10, width=timescale / 4)

        up_nonps = []
        down_nonps = []

        # Calculate the non-planarities of the regions
        for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
            for j in range(len(props['left_bases'])):
                X_region = X[props['left_bases'][j]:props['right_bases'][j] - 1]
                pca = PCA()
                pca.fit(X_region)
                r = pca.explained_variance_ratio_
                nonp = r[2] / np.sqrt(r[1] * r[2])
                if colour == 'green':
                    up_nonps.append(nonp)
                else:
                    down_nonps.append(nonp)

        # Score the trajectory based on the non-planarities of the regions
        scores[i] = sum(down_nonps)

    return scores


def _plot_trajectory_with_regions(
        X: np.ndarray,
        timescale: int,
        index: int,
        score: float = None,
):
    """
    Draw the trajectory coloured by the time elapsed.
    Draw boxes around the regions.
    """

    # Find regions from displacement
    d = calculate_displacements(X, deltas=timescale)
    d_mean = d.mean()
    state_up = np.r_[[0, ], d > d_mean, [0, ]]
    up_idxs, up_section_props = find_peaks(state_up, height=0.5, distance=timescale / 10, width=timescale / 4)
    state_down = np.r_[[0, ], d < d_mean, [0, ]]
    down_idxs, down_section_props = find_peaks(state_down, height=0.5, distance=timescale / 10, width=timescale / 4)

    # Construct colours
    colours = np.linspace(0, 1, X.shape[0])
    cmap = plt.get_cmap('viridis_r')

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(6, 4)
    fig.suptitle(f'Index={index}.' + ('' if score is None else f' Score={score:.3f}.'))

    # Plot the state/displacement
    ax = fig.add_subplot(gs[0, :])
    ax.plot(d)
    ax.axhline(y=d_mean, color='red')

    # Highlight regions where above/below average
    for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
        for i in range(len(props['left_bases'])):
            ax.fill_between(
                np.arange(props['left_bases'][i], props['right_bases'][i] - 1),
                max(d),
                color=colour,
                alpha=0.3,
                zorder=-1,
                linewidth=0
            )

    # Plot the regions as boxes aligned with the trajectory
    ax = fig.add_subplot(gs[1:, :], projection='3d', azim=-125, elev=35)
    for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
        for i in range(len(props['left_bases'])):
            X_region = X[props['left_bases'][i]:props['right_bases'][i] - 1]
            pca = PCA()
            pca.fit(X_region)

            # Scatter the vertices
            x, y, z = X_region[::5].T
            ax.scatter(x, y, z, c=colour, s=10, alpha=0.4, zorder=-1)

            # Add the region box
            box = make_box_from_pca(X_region, pca, colour, scale=(1, 1, 2))
            ax.add_collection3d(box)

    # Draw lines connecting points
    points = X[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)

    # Setup axis
    equal_aspect_ratio(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_regions_ts={timescale}_sim_{i}.{img_extension}', transparent=True)
        # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_regions_sim_{i}', X=TC.X[i])

    if show_plots:
        plt.show()


def _plot_trajectory_with_regions_from_turns(
        X: np.ndarray,
        tumble_ts: np.ndarray,
        thetas: np.ndarray,
        phis: np.ndarray,
        timescale: int,
        index: int,
        score: float = None,
):
    """
    Draw the trajectory coloured by the time elapsed.
    Draw boxes around the regions.
    """

    # Find regions from tumble frequency
    z = np.zeros(len(X))
    z[tumble_ts.astype(np.int32)] = (1 + np.abs(thetas) + np.abs(phis))**2
    k = 0
    inc = 201
    while k < timescale:
        z = smooth_trajectory(z, window_len=inc)
        k += inc
    z = z.squeeze()

    z_mean = z.mean()
    state_up = np.r_[[0, ], z < z_mean, [0, ]]
    up_idxs, up_section_props = find_peaks(state_up, height=0.5, distance=timescale / 10, width=timescale / 4)
    state_down = np.r_[[0, ], z > z_mean, [0, ]]
    down_idxs, down_section_props = find_peaks(state_down, height=0.5, distance=timescale / 10, width=timescale / 4)

    # Construct colours
    colours = np.linspace(0, 1, X.shape[0])
    cmap = plt.get_cmap('viridis_r')

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(6, 4)
    fig.suptitle(f'Index={index}.' + ('' if score is None else f' Score={score:.3f}.'))

    # Plot the state/turn clustering
    ax = fig.add_subplot(gs[0, :])
    ax.plot(z)
    ax.axhline(y=z_mean, color='red')

    for tumble_t in tumble_ts:
        ax.axvline(x=tumble_t, color='black', linestyle=':', alpha=0.2)

    # Highlight regions where above/below average
    for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
        for i in range(len(props['left_bases'])):
            ax.fill_between(
                np.arange(props['left_bases'][i], props['right_bases'][i] - 1),
                max(z),
                color=colour,
                alpha=0.3,
                zorder=-1,
                linewidth=0
            )

    # Plot the regions as boxes aligned with the trajectory
    ax = fig.add_subplot(gs[1:, :], projection='3d', azim=-125, elev=35)
    for colour, props in [('green', up_section_props), ('orange', down_section_props)]:
        for i in range(len(props['left_bases'])):
            X_region = X[props['left_bases'][i]:props['right_bases'][i] - 1]
            pca = PCA()
            pca.fit(X_region)

            # Scatter the vertices
            x, y, z = X_region[::5].T
            ax.scatter(x, y, z, c=colour, s=10, alpha=0.4, zorder=-1)

            # Add the region box
            box = make_box_from_pca(X_region, pca, colour, scale=(1, 1, 2))
            ax.add_collection3d(box)

    # Draw lines connecting points
    points = X[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)

    # Setup axis
    equal_aspect_ratio(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_regions_ts={timescale}_sim_{i}.{img_extension}', transparent=True)
        # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_regions_sim_{i}', X=TC.X[i])

    if show_plots:
        plt.show()


def _plot_trajectory_with_fixed_regions(
        X: np.ndarray,
        timescale: int,
        index: int,
        score: float = None,
):
    """
    Draw the trajectory coloured by the time elapsed.
    Draw boxes around the regions.
    """

    regions = {
        26: [
            {
                'start_idx': 0,
                'end_idx': 2500,
                'colour': 'orange'
            },
            {
                'start_idx': 2600,
                'end_idx': 8208,
                'colour': 'green'
            },
            {
                'start_idx': 8208,
                'end_idx': 15550,
                'colour': 'orange'
            },
            {
                'start_idx': 15550,
                'end_idx': 18350,
                'colour': 'green'
            },
            {
                'start_idx': 18500,
                'end_idx': 20400,
                'colour': 'orange'
            },
            {
                'start_idx': 20400,
                'end_idx': 22750,
                'colour': 'green'
            },
            {
                'start_idx': 22750,
                'end_idx': 24000,
                'colour': 'orange'
            },
            {
                'start_idx': 24000,
                'end_idx': 26750,
                'colour': 'green'
            },
            {
                'start_idx': 26750,
                'end_idx': 29700,
                'colour': 'orange'
            },
            {
                'start_idx': 29700,
                'end_idx': 32600,
                'colour': 'green'
            },
            {
                'start_idx': 32600,
                'end_idx': 42000,
                'colour': 'orange'
            },
        ],
        7: [
            {
                'start_idx': 300,
                'end_idx': 2000,
                'colour': 'orange'
            },
            {
                'start_idx': 2100,
                'end_idx': 3000,
                'colour': 'green'
            },
            {
                'start_idx': 3100,
                'end_idx': 5800,
                'colour': 'orange'
            },
            {
                'start_idx': 6000,
                'end_idx': 7200,
                'colour': 'green'
            },
            {
                'start_idx': 7300,
                'end_idx': 9200,
                'colour': 'orange'
            },
            {
                'start_idx': 9300,
                'end_idx': 11200,
                'colour': 'green'
            },
            {
                'start_idx': 11100,
                'end_idx': 12700,
                'colour': 'orange'
            },
            {
                'start_idx': 12800,
                'end_idx': 14000,
                'colour': 'green'
            },
            {
                'start_idx': 14000,
                'end_idx': 19600,
                'colour': 'orange'
            },
            {
                'start_idx': 19700,
                'end_idx': 21000,
                'colour': 'green'
            },
            {
                'start_idx': 21100,
                'end_idx': len(X) - 1,
                'colour': 'orange'
            },
        ]
    }

    if index not in regions:
        raise RuntimeError(f'Regions not defined for index={index}!')

    # Find regions from displacement
    d = calculate_displacements(X, deltas=timescale)
    d_mean = d.mean()
    state_up = np.r_[[0, ], d > d_mean, [0, ]]
    up_idxs, up_section_props = find_peaks(state_up, height=0.5, distance=timescale / 10, width=timescale / 4)
    state_down = np.r_[[0, ], d < d_mean, [0, ]]
    down_idxs, down_section_props = find_peaks(state_down, height=0.5, distance=timescale / 10, width=timescale / 4)

    # Construct colours
    colours = np.linspace(0, 1, X.shape[0])
    cmap = plt.get_cmap('viridis_r')

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(6, 4)
    # fig.suptitle(f'Index={index}.' + ('' if score is None else f' Score={score:.3f}.'))

    # # Plot the state/displacement
    # ax = fig.add_subplot(gs[0, :])
    # ax.plot(d)
    # ax.axhline(y=d_mean, color='red')

    # # Highlight regions where above/below average
    # for region in regions[index]:
    #     ax.fill_between(
    #         np.arange(region['start_idx'], region['end_idx']),
    #         max(d),
    #         color=region['colour'],
    #         alpha=0.3,
    #         zorder=-1,
    #         linewidth=0
    #     )

    # Plot the regions as boxes aligned with the trajectory
    ax = fig.add_subplot(gs[:, :], projection='3d', azim=140, elev=-160)
    for region in regions[index]:
        X_region = X[region['start_idx']:region['end_idx']]
        pca = PCA()
        pca.fit(X_region)

        # Scatter the vertices
        x, y, z = X_region.T
        ax.scatter(x, y, z, c=region['colour'], s=10, alpha=0.4, zorder=-1)

        # Add the region box
        if region['colour'] == 'green':
            scale = (1, 2, 3)
        else:
            scale = (1, 2, 4)

        box = make_box_from_pca(X_region, pca, region['colour'], scale=scale)
        ax.add_collection3d(box)

    # Draw lines connecting points
    points = X[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)

    # Setup axis
    equal_aspect_ratio(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_regions_ts={timescale}_sim_{index}.{img_extension}',
                    transparent=True)

    if show_plots:
        plt.show()


def plot_trajectories_with_regions():
    """
   Draw the trajectory coloured by the time elapsed.
   Draw boxes around the regions.
   """
    args = get_args()
    pe, TC = get_trajectories_from_args(args)

    if plot_n_examples == 0:
        return

    candidate_idxs = [0, 1, ]

    plot_idxs = np.arange(args.batch_size)
    timescale = int(100 / args.sim_dt)
    s2_threshold = 0.01

    i = 7
    # # _plot_trajectory_with_regions(TC.X[i], timescale, i)
    _plot_trajectory_with_fixed_regions(TC.X[i], timescale, i)
    exit()

    scores = _get_niceness_scores(TC.X, timescale)
    plot_idxs = scores.argsort()[::-1][:plot_n_examples]

    for i in plot_idxs:
        logger.info(f'Plotting simulation idx {i}/{len(plot_idxs)}.')
        # _plot_trajectory_with_regions(TC.X[i], timescale, i, scores[i])
        _plot_trajectory_with_regions_from_turns(TC.X[i], TC.tumble_ts[i] / args.sim_dt, TC.thetas[i], TC.phis[i],
                                                 timescale, i, scores[i])
        # _plot_trajectory_with_fixed_regions(TC.X[i], timescale, i, scores[i])


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    simulate()
    # plot_trajectories_with_regions()
