import os
from argparse import Namespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from progress.bar import Bar
from scipy.stats import ttest_ind

from simple_worm.plot3d import MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.cache import get_sim_state_from_args
from wormlab3d.toolkit.plot_utils import tex_mode
from wormlab3d.toolkit.util import hash_data
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import get_deltas_from_args

show_plots = False
save_plots = True
img_extension = 'svg'


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

    for k in ['npas', 'voxel_sizes', 'duration', 'durations', 'dt', 'batch_size', 'deltas', 'delta_step',
              'targets_radii', 'n_targets', 'epsilon', 'max_nonplanar_pause_duration', 'detection_area', 'pauses']:
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
        elif k == 'durations' and hasattr(args, 'sim_durations'):
            if len(args.sim_durations) > 3:
                durations = f'{args.sim_durations[0]:.1E}-{args.sim_durations[-1]:.1E}'
            else:
                durations = ','.join(f'{d:.1E}' for d in args.sim_durations)
            fn += f'_T={durations}'
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
        elif k == 'pauses' and hasattr(args, 'pauses'):
            if len(args.pauses) > 3:
                pauses = f'{args.pauses[0]:.1E}-{args.pauses[-1]:.1E}'
            else:
                pauses = ','.join(f'{p:.1E}' for p in args.pauses)
            fn += f'_pauses={pauses}'

    return LOGS_PATH / (fn + '.' + extension)


def _get_npas_from_args(args: Namespace) -> np.ndarray:
    return np.exp(-np.linspace(np.log(1 / args.npas_min), np.log(1 / args.npas_max), args.npas_num))


def _get_voxel_sizes_from_args(args: Namespace) -> np.ndarray:
    return np.exp(-np.linspace(np.log(1 / args.vxs_max), np.log(1 / args.vxs_min), args.vxs_num))


def _get_durations_from_args(args: Namespace) -> np.ndarray:
    if args.durations_intervals == 'exponential':
        durations = np.exp(-np.linspace(np.log(1 / (args.durations_min * 60)), np.log(1 / (args.durations_min * 60)),
                                        args.durations_num))
    else:
        durations = np.arange(args.durations_min, args.durations_min + args.durations_num)**2 * 60
    durations = np.round(durations / args.sim_dt) * args.sim_dt
    return durations


def _get_pauses_from_args(args: Namespace) -> np.ndarray:
    if args.pauses_intervals == 'exponential':
        if args.pauses_min == 0:
            pauses = np.r_[[0, ], np.exp(-np.linspace(np.log(1 / 1), np.log(1 / args.pauses_max), args.pauses_num - 1))]
        else:
            pauses = np.exp(-np.linspace(np.log(1 / args.pauses_min), np.log(1 / args.pauses_max), args.pauses_num))
    else:
        pauses = np.arange(args.pauses_min, args.pauses_min + args.pauses_num)**2
    return pauses


def coverage_scores():
    """
    Simulate across a range of non-planarities and score trajectories based on how many unique voxels have been visited.
    """
    tex_mode()
    args = get_args(validate_source=False)
    deltas, delta_ts = get_deltas_from_args(args)
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    voxel_sizes = _get_voxel_sizes_from_args(args)
    n_vs = args.vxs_num

    # Outputs
    scores = np.zeros((n_sigmas, n_vs, args.batch_size))
    nonp = np.zeros((n_sigmas, args.batch_size))
    msds_all = np.zeros((n_sigmas, len(deltas)))
    msds = np.zeros((n_sigmas, args.batch_size, len(deltas)))
    Xs_all = []

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)
        Xs_all.append(SS.X)

        # Calculate non-planarity for validation
        nonp[i] = SS.get_nonp()
        if SS.needs_save:
            SS.save()

        # Calculate optimality
        logger.info('Calculating coverage.')
        bar = Bar('Calculating', max=n_vs)
        bar.check_tty = False
        for k, vs in enumerate(voxel_sizes):
            scores[i, k] = SS.get_coverage(vs)
            bar.next()
        bar.finish()
        if SS.needs_save:
            SS.save()

        # Calculate MSDs
        logger.info(f'Calculating displacements.')
        msds_all[i], msds[i] = SS.get_msds(deltas)
        if SS.needs_save:
            SS.save()
        logger.info('----')

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(3, figsize=(14, 12))
    fig.suptitle(
        f'T={args.sim_duration}s. dt={args.sim_dt:.2f}. Batch size={args.batch_size}. Pause={args.nonp_pause_max}s')
    cmap = plt.get_cmap('jet')

    # Scores
    ax = axes[0]
    ax.set_title('Scores')
    colours = cmap(np.linspace(0, 1, n_vs))
    for k, vs in enumerate(voxel_sizes):
        sk = scores[:, k]
        sk_mean = sk.mean(axis=-1)
        ax.errorbar(
            npa_sigmas,
            sk_mean,
            yerr=[sk_mean - sk.min(axis=-1), sk.max(axis=-1) - sk_mean],
            capsize=5,
            color=colours[k],
            label=f'Voxel size={vs:.1E}'
        )
    # ax.legend()
    ax.legend(bbox_to_anchor=(1.02, 1))
    ax.set_xlabel('$\sigma_{\phi}$')
    ax.set_xticks(npa_sigmas)
    ax.set_xticklabels(npa_sigmas)
    ax.set_ylabel('$\\frac{\#\\text{voxels visited}}{\\text{voxel size}}$')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.grid()

    # Non-planarity
    ax = axes[1]
    ax.set_title('Non-planarity of trajectories')
    nonp_mean = nonp.mean(axis=-1)
    ax.errorbar(
        npa_sigmas,
        nonp_mean,
        yerr=[nonp_mean - nonp.min(axis=-1), nonp.max(axis=-1) - nonp_mean],
        capsize=5,
    )
    ax.set_xlabel('$\sigma_{\phi}$')
    ax.set_xticks(npa_sigmas)
    ax.set_xticklabels(npa_sigmas)
    ax.set_ylabel('Non-planarity')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    # MSD
    ax = axes[2]
    ax.set_title('MSD')
    js = np.linspace(0, 1, args.batch_size + 4)[2:-2]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    colours = cmap(np.linspace(0, 1, n_sigmas))
    # cmaps = [plt.get_cmap(name) for name in ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds']]

    # Plot all trajectories plus the average for each sigma
    for i, npas in enumerate(npa_sigmas):
        ax.plot(delta_ts, msds_all[i], label=f'$\sigma$={npas:.1E}',
                alpha=0.8, c=colours[i], linewidth=1, zorder=100)
        # alpha=0.8, c=default_colours[i], linestyle='--', linewidth=3, zorder=100)

        # # Plot each trajectory
        # for j in range(batch_size):
        #     msd_vals = np.array(list(msds[i][j].values()))
        #     ax.plot(delta_ts, msd_vals, alpha=0.5, c=default_colours[i])

    # Complete MSD plot
    ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta\ (s)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    # ax.legend()
    ax.legend(bbox_to_anchor=(1.02, 1))

    fig.tight_layout()

    if save_plots:
        args.npas = npa_sigmas
        args.voxel_sizes = voxel_sizes
        plt.savefig(
            make_filename('coverage_results', args, excludes=['targets_radii', 'n_targets', 'epsilon']),
            transparent=True
        )

    if show_plots:
        plt.show()


def search_scores():
    """
    Simulate across a range of non-planarities and score trajectories based on how well they find targets.
    ## todo: not ported across to new SimulationState so currently broken
    """
    args = get_args(validate_source=False)

    # npa_sigmas = [1e-8, 10]
    # npa_sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = np.linspace(0.00001, 10, 20)
    # npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 20))
    npa_sigmas = np.exp(-np.linspace(np.log(1 / 1e-6), np.log(1 / 10), 8))
    # npa_sigmas = np.exp(-np.linspace(np.log(1 / 1e-6), np.log(1 / 10), 12))
    n_sigmas = len(npa_sigmas)

    targets_radii = np.linspace(0, 20, 21)
    n_radii = len(targets_radii)

    n_targets = 100  # number of targets at each radius
    epsilon = 0.1  # distance below which trajectory has "found" target

    # todo: this properly
    args.npas = npa_sigmas
    args.targets_radii = targets_radii
    args.n_targets = n_targets
    args.epsilon = epsilon

    # Outputs
    find_times = np.ones((n_sigmas, n_radii, args.batch_size, n_targets)) * -1
    finds = np.zeros((n_sigmas, n_radii, args.batch_size))
    finds_pop = np.zeros((n_sigmas, n_radii, n_targets), dtype=np.bool)

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)

        logger.info('Finding targets.')
        bar = Bar('Finding', max=n_radii)
        bar.check_tty = False

        for j, r in enumerate(targets_radii):
            finds_ij = SS.get_finds(r, n_targets, epsilon)
            finds[i, j] = finds_ij['finds']
            find_times[i, j] = finds_ij['find_times']
            finds_pop[i, j] = finds_ij['finds_pop']
            bar.next()
        bar.finish()
        logger.info('----')
        if SS.needs_save:
            SS.save()

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(3, figsize=(14, 12))
    fig.suptitle(f'T={args.sim_duration}s. dt={args.sim_dt:.2f}. Batch size={args.batch_size}. '
                 f'Detection radius={epsilon}. '
                 f'Num. targets={n_targets}. '
                 f'Pause={args.nonp_pause_max}s.')
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, n_sigmas))

    # Set up axes
    ax_finds_traj = axes[0]
    ax_finds_traj.set_title('How many targets found by a trajectory?')
    ax_finds_pop = axes[1]
    ax_finds_pop.set_title('How many targets found by the population?')
    ax_times = axes[2]
    ax_times.set_title('Time to find targets.')

    for i, npas in enumerate(npa_sigmas):
        # Number of targets found by trajectories
        fi = finds[i] / n_targets
        fi_mean = fi.mean(axis=-1)
        ax_finds_traj.errorbar(
            targets_radii + i * 0.005,
            fi_mean,
            yerr=[fi_mean - fi.min(axis=-1), fi.max(axis=-1) - fi_mean],
            marker='x',
            capsize=5,
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )

        # Number of targets found by the population
        fpi = finds_pop[i].sum(axis=-1) / n_targets
        ax_finds_pop.plot(
            targets_radii,
            fpi,
            marker='x',
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )

        # Average time taken to find targets when found
        fti_mean = np.zeros(n_radii)
        fti_min = np.zeros(n_radii)
        fti_max = np.zeros(n_radii)
        for r in range(n_radii):
            ftir = find_times[i, r][find_times[i, r] > -1]
            if len(ftir) > 0:
                fti_mean[r] = ftir.mean()
                fti_min[r] = ftir.min()
                fti_max[r] = ftir.max()

        ax_times.errorbar(
            targets_radii + i * 0.005,
            fti_mean,
            yerr=[fti_mean - fti_min, fti_max - fti_mean],
            marker='x',
            capsize=5,
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )
    # ax.legend()
    ax_finds_traj.legend(bbox_to_anchor=(1.12, 1))
    ax_finds_traj.set_xlabel('$r$')
    ax_finds_pop.set_xlabel('$r$')
    ax_times.set_xlabel('$r$')
    ax_finds_traj.set_xticks(targets_radii)
    ax_finds_pop.set_xticks(targets_radii)
    ax_times.set_xticks(targets_radii)
    ax_finds_traj.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax_finds_pop.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax_times.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax_finds_traj.set_ylabel('Found \%')
    ax_finds_pop.set_ylabel('Found \%')
    ax_times.set_ylabel('$<T>$')
    ax_finds_traj.grid()
    ax_finds_pop.grid()
    ax_times.grid()
    ax_finds_traj.set_yscale('log')
    ax_finds_pop.set_yscale('log')
    # ax_times.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('search_results', args, excludes=['voxel_sizes', 'deltas', 'delta_step']),
            transparent=True
        )

    if show_plots:
        plt.show()


def search_t_tests():
    """
    Simulate across a range of non-planarities and use t-statistics to assess if populations are significantly different.
    ## todo: not ported across to new SimulationState so currently broken
    """
    args = get_args(validate_source=False)

    # npa_sigmas = [1e-8, 10]
    # npa_sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = np.linspace(0.00001, 10, 20)
    # npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 20))
    npa_sigmas = np.exp(-np.linspace(np.log(1 / 1e-6), np.log(1 / 10), 8))
    n_sigmas = len(npa_sigmas)

    # targets_radii = np.linspace(0, 6, 25)
    targets_radii = np.linspace(0, 10, 21)
    n_radii = len(targets_radii)

    n_targets = 1000  # number of targets at each radius
    epsilon = 0.1  # distance below which trajectory has "found" target

    # todo: this properly
    args.npas = npa_sigmas
    args.targets_radii = targets_radii
    args.n_targets = n_targets
    args.epsilon = epsilon

    # Outputs
    find_times = np.ones((n_sigmas, n_radii, args.batch_size, n_targets)) * -1
    finds = np.zeros((n_sigmas, n_radii, args.batch_size))
    finds_pop = np.zeros((n_sigmas, n_radii, n_targets), dtype=np.bool)

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        pe, TC = get_sim_state_from_args(args)

        logger.info('Finding targets.')
        bar = Bar('Finding', max=n_radii)
        bar.check_tty = False

        for j, r in enumerate(targets_radii):
            finds_ij = TC.get_finds(r, n_targets, epsilon)
            finds[i, j] = finds_ij['finds']
            find_times[i, j] = finds_ij['find_times']
            finds_pop[i, j] = finds_ij['finds_pop']
            bar.next()
        bar.finish()
        logger.info('----')
        if TC.needs_save:
            TC.save()

    # Filter out not-found times
    find_times_found = {}
    for i in range(n_sigmas):
        find_times_found[i] = {}
        for r in range(n_radii):
            ftir = find_times[i, r][find_times[i, r] > -1]
            find_times_found[i][r] = ftir

    # Calculate pairwise t-statistics
    logger.info('Calculating t-statistics.')
    t_stats_times = np.zeros((n_radii, n_sigmas, n_sigmas))
    p_vals_times = np.zeros((n_radii, n_sigmas, n_sigmas))
    t_stats_finds = np.zeros((n_radii, n_sigmas, n_sigmas))
    p_vals_finds = np.zeros((n_radii, n_sigmas, n_sigmas))
    for r in range(n_radii):
        for i in range(n_sigmas):
            for j in range(i, n_sigmas):
                # Find times
                try:
                    rvs1 = find_times_found[i][r]
                    rvs2 = find_times_found[j][r]
                except IndexError:
                    continue
                res = ttest_ind(rvs1, rvs2, equal_var=False)
                t_stats_times[r, i, j] = 0 if np.isnan(res.statistic) else res.statistic
                t_stats_times[r, j, i] = 0 if np.isnan(res.statistic) else res.statistic
                p_vals_times[r, i, j] = 0 if np.isnan(res.pvalue) else res.pvalue
                p_vals_times[r, j, i] = 0 if np.isnan(res.pvalue) else res.pvalue

                # Find counts
                rvs1 = finds[i, r]
                rvs2 = finds[j, r]
                res = ttest_ind(rvs1, rvs2, equal_var=False)
                t_stats_finds[r, i, j] = res.statistic
                t_stats_finds[r, j, i] = res.statistic
                p_vals_finds[r, i, j] = res.pvalue
                p_vals_finds[r, j, i] = res.pvalue

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(4, n_radii, figsize=(n_radii * 2, 12))
    fig.suptitle(f'T={args.sim_duration}s. '
                 f'dt={args.sim_dt:.2f}. '
                 f'Batch size={args.batch_size}. '
                 f'Detection radius={epsilon}. '
                 f'Num. targets={n_targets}. '
                 f'Pause={args.nonp_pause_max}s.')

    for r, radius in enumerate(targets_radii):
        ax = axes[0, r]
        if r == 0:
            ax.set_ylabel('Finds t-statistics')
        ax.set_title(f'r={radius:.2f}')
        im = ax.imshow(t_stats_finds[r], aspect='auto', cmap='PRGn', norm=MidpointNormalize(midpoint=0))
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, r]
        if r == 0:
            ax.set_ylabel('Finds p-values')
        im = ax.imshow(p_vals_finds[r], aspect='auto', cmap='RdBu', norm=MidpointNormalize(midpoint=0.05))
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2, r]
        if r == 0:
            ax.set_ylabel('Times t-statistics')
        im = ax.imshow(t_stats_times[r], aspect='auto', cmap='PRGn', norm=MidpointNormalize(midpoint=0))
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[3, r]
        if r == 0:
            ax.set_ylabel('Times p-values')
        im = ax.imshow(p_vals_times[r], aspect='auto', cmap='RdBu', norm=MidpointNormalize(midpoint=0.05))
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('search_t_tests', args, excludes=['voxel_sizes', 'deltas', 'delta_step']),
            transparent=True
        )

    if show_plots:
        plt.show()


def surface_coverage_scores():
    """
    Simulate across a range of non-planarities and count frequency of trajectories crossing different radii.
    """
    args = get_args(validate_source=False)
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    targets_radii = np.linspace(0, 50, 51)
    n_radii = len(targets_radii)
    args.npas = npa_sigmas
    args.targets_radii = targets_radii

    # Outputs
    crossings = np.ones((n_sigmas, n_radii, args.batch_size))

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)

        logger.info('Counting crossings.')
        bar = Bar('Counting', max=n_radii)
        bar.check_tty = False

        for j, r in enumerate(targets_radii):
            crossings[i, j] = SS.get_crossings(r)
            bar.next()
        bar.finish()
        logger.info('----')
        if SS.needs_save:
            SS.save()

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(2, figsize=(14, 12))
    fig.suptitle(f'T={args.sim_duration}s. dt={args.sim_dt:.2f}. Batch size={args.batch_size}. '
                 f'Pause={args.nonp_pause_max}s.')
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, n_sigmas))

    # Set up axes
    ax_traj = axes[0]
    ax_traj.set_title('Crossings per trajectory.')
    ax_pop = axes[1]
    ax_pop.set_title('Crossings per population.')

    for i, npas in enumerate(npa_sigmas):
        # Number of targets found by trajectories
        ci = crossings[i]
        ci_mean = ci.mean(axis=-1)
        ax_traj.errorbar(
            targets_radii + i * 0.005,
            ci_mean,
            yerr=[ci_mean - ci.min(axis=-1), ci.max(axis=-1) - ci_mean],
            marker='x',
            capsize=5,
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )

        # Number of crossings by the population
        cpi = crossings[i].sum(axis=-1)
        ax_pop.plot(
            targets_radii,
            cpi,
            marker='x',
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )

    ax_traj.legend(bbox_to_anchor=(1.12, 1))
    ax_traj.set_xlabel('$r$')
    ax_pop.set_xlabel('$r$')
    ax_traj.set_xticks(targets_radii)
    ax_pop.set_xticks(targets_radii)
    ax_traj.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax_pop.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax_traj.set_ylabel('Crossings')
    ax_pop.set_ylabel('Crossings')
    ax_traj.grid()
    ax_pop.grid()
    ax_traj.set_yscale('log')
    ax_pop.set_yscale('log')
    # ax_times.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('crossings', args, excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon']),
            transparent=True
        )

    if show_plots:
        plt.show()


def crossings_nonp():
    """
    Non-planarities of crossing points relative to principal plane.
    """
    args = get_args(validate_source=False)
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    targets_radii = np.linspace(0, 30, 31)
    n_radii = len(targets_radii)
    args.npas = npa_sigmas
    args.targets_radii = targets_radii

    # Outputs
    nonp_counts = np.zeros((n_sigmas, n_radii))
    nonp_mins = np.zeros((n_sigmas, n_radii))
    nonp_maxs = np.zeros((n_sigmas, n_radii))
    nonp_means = np.zeros((n_sigmas, n_radii))

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)

        logger.info('Calculating crossing non-planarities.')
        bar = Bar('Counting', max=n_radii)
        bar.check_tty = False

        for j, r in enumerate(targets_radii):
            nonp = SS.get_crossings_nonp(r)
            nonp_counts[i, j] = len(nonp)
            if len(nonp) > 0:
                nonp_mins[i, j] = nonp.min()
                nonp_maxs[i, j] = nonp.max()
                nonp_means[i, j] = nonp.mean()
            bar.next()
        bar.finish()
        logger.info('----')
        if SS.needs_save:
            SS.save()

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(2, figsize=(14, 12))
    fig.suptitle(f'T={args.sim_duration}s. dt={args.sim_dt:.2f}. Batch size={args.batch_size}. '
                 f'Pause={args.nonp_pause_max}s.')
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, n_sigmas))

    # Set up axes
    ax = axes[0]
    ax.set_title('Non-planarities of the crossing points.')

    for i, npas in enumerate(npa_sigmas):
        # Number of targets found by trajectories
        ax.errorbar(
            targets_radii + i * 0.005,
            nonp_means[i],
            yerr=[nonp_means[i] - nonp_mins[i], nonp_maxs[i] - nonp_means[i]],
            marker='x',
            capsize=5,
            color=colours[i],
            label=f'$\sigma_{{\phi}}=${npas:.1E}',
            alpha=0.7,
        )

    ax.legend(bbox_to_anchor=(1.12, 1))
    ax.set_xlabel('$r$')
    ax.set_xticks(targets_radii)
    ax.set_xticklabels([f'{r:.2f}' for r in targets_radii])
    ax.set_ylabel('Crossing points non-planarities')
    ax.grid()
    ax.set_yscale('log')

    # Plot metrics?
    max_distances = (n_radii - (nonp_counts[:, ::-1] > 0).argmax(axis=1)) / n_radii
    max_nonp = nonp_means.mean(axis=1)**(1 / 2)
    metric = max_distances * max_nonp

    ax = axes[1]
    ax.set_title('max(dist)*mean(z)')
    ax.plot(metric, marker='x')
    ax.set_xticks(np.arange(n_sigmas))
    ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
    ax.grid()

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('crossings_nonp', args,
                          excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon']),
            transparent=True
        )

    if show_plots:
        plt.show()


def volume_metric():
    """
    Estimate the volume explored by a typical trajectory.
    """
    args = get_args(validate_source=False)
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    args.npas = npa_sigmas

    def _calculate_disk_volumes(r_, z_):
        sphere_vols = 4 / 3 * np.pi * r_**3
        cap_vols = 1 / 3 * np.pi * (r_ - z_)**2 * (2 * r_ + z_)
        return sphere_vols - 2 * cap_vols

    # Outputs
    r_values = np.zeros((n_sigmas, 3))
    z_values = np.zeros((n_sigmas, 3))
    y_values = np.zeros((n_sigmas, 3))
    r2_values = np.zeros((n_sigmas, 3))
    s_values = np.zeros((n_sigmas, 3, 3))
    sv_disk_vols = np.zeros((n_sigmas, 3))
    sv_cuboid_vols = np.zeros((n_sigmas, 3))

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        SS = get_sim_state_from_args(args)

        # Find the maximums in each relative directions
        Xt = SS.get_Xt()
        Xt_max = np.abs(Xt).max(axis=1)

        # Use first component as radius and third as height of explored disk
        r_values[i] = [Xt_max[:, 0].mean(), Xt_max[:, 0].min(), Xt_max[:, 0].max()]
        z_values[i] = [Xt_max[:, 2].mean(), Xt_max[:, 2].min(), Xt_max[:, 2].max()]

        # Use second component to validate against a cuboid metric
        y_values[i] = [Xt_max[:, 1].mean(), Xt_max[:, 1].min(), Xt_max[:, 1].max()]

        # Validation r2 is close to the non-rotated frame
        r2 = np.linalg.norm(SS.X, axis=-1).max(axis=1)
        r2_values[i] = [r2.mean(), r2.min(), r2.max()]

        # Compute singular value volumes
        logger.info('Computing singular value based volumes.')
        sv_i = np.zeros((args.batch_size, 3))
        for j in range(args.batch_size):
            sv_i[j] = SS.get_pca(j).singular_values_
        s_values[i] = np.stack([sv_i.mean(axis=0), sv_i.min(axis=0), sv_i.max(axis=0)], axis=1)
        disk_vols_i = _calculate_disk_volumes(sv_i[:, 0], sv_i[:, 2])
        cuboid_vols_i = sv_i[:, 0] * sv_i[:, 1] * sv_i[:, 2]
        sv_disk_vols[i] = [disk_vols_i.mean(), disk_vols_i.min(), disk_vols_i.max()]
        sv_cuboid_vols[i] = [cuboid_vols_i.mean(), cuboid_vols_i.min(), cuboid_vols_i.max()]

        if SS.needs_save:
            SS.save()

    # Calculate the volumes
    disk_vols = _calculate_disk_volumes(r_values, z_values)
    # sphere_vols = 4 / 3 * np.pi * r_values**3
    # cap_vols = 1 / 3 * np.pi * (r_values - z_values)**2 * (2 * r_values + z_values)
    # disk_vols = sphere_vols - 2 * cap_vols
    cuboid_vols = r_values * z_values * y_values

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(3, figsize=(14, 12))
    fig.suptitle(f'T={args.sim_duration}s. dt={args.sim_dt:.2f}. Batch size={args.batch_size}. '
                 f'Pause={args.nonp_pause_max}s.')

    # Plot radii
    ax = axes[0]
    ax.set_title('Maximum radius reached.')
    # ax.plot(r_values, marker='x')
    ax.errorbar(
        np.arange(n_sigmas),
        r_values[:, 0],
        yerr=[r_values[:, 0] - r_values[:, 1], r_values[:, 2] - r_values[:, 0]],
        marker='x',
        capsize=5,
        label='Along v1'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        r2_values[:, 0],
        yerr=[r2_values[:, 0] - r2_values[:, 1], r2_values[:, 2] - r2_values[:, 0]],
        marker='x',
        capsize=5,
        label='norm'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        s_values[:, 0, 0],
        yerr=[s_values[:, 0, 0] - s_values[:, 0, 1], s_values[:, 0, 2] - s_values[:, 0, 0]],
        marker='x',
        capsize=5,
        label='s0'
    )
    ax.legend()
    ax.set_xticks(np.arange(n_sigmas))
    ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
    ax.grid()
    ax.set_yscale('log')

    # Plot z values
    ax = axes[1]
    ax.set_title('Maximum z-values.')
    # ax.plot(z_values, marker='x')
    ax.errorbar(
        np.arange(n_sigmas),
        z_values[:, 0],
        yerr=[z_values[:, 0] - z_values[:, 1], z_values[:, 2] - z_values[:, 0]],
        marker='x',
        capsize=5,
        label='z'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        s_values[:, 2, 0],
        yerr=[s_values[:, 2, 0] - s_values[:, 2, 1], s_values[:, 2, 2] - s_values[:, 2, 0]],
        marker='x',
        capsize=5,
        label='s2'
    )
    ax.legend()
    ax.set_xticks(np.arange(n_sigmas))
    ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
    ax.grid()
    ax.set_yscale('log')

    # Plot volumes
    ax = axes[2]
    ax.set_title('Volumes explored.')
    # ax.plot(disk_vols, marker='x')
    ax.errorbar(
        np.arange(n_sigmas),
        disk_vols[:, 0],
        yerr=[disk_vols[:, 0] - disk_vols[:, 1], disk_vols[:, 2] - disk_vols[:, 0]],
        marker='x',
        capsize=5,
        label='Disks'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        cuboid_vols[:, 0],
        yerr=[cuboid_vols[:, 0] - cuboid_vols[:, 1], cuboid_vols[:, 2] - cuboid_vols[:, 0]],
        marker='x',
        capsize=5,
        label='Cuboids'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        sv_disk_vols[:, 0],
        yerr=[sv_disk_vols[:, 0] - sv_disk_vols[:, 1], sv_disk_vols[:, 2] - sv_disk_vols[:, 0]],
        marker='x',
        capsize=5,
        label='SV disks'
    )
    ax.errorbar(
        np.arange(n_sigmas),
        sv_cuboid_vols[:, 0],
        yerr=[sv_cuboid_vols[:, 0] - sv_cuboid_vols[:, 1], sv_cuboid_vols[:, 2] - sv_cuboid_vols[:, 0]],
        marker='x',
        capsize=5,
        label='SV cuboids'
    )
    ax.legend()
    ax.set_xticks(np.arange(n_sigmas))
    ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
    ax.grid()
    ax.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('volume_metric', args,
                          excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon']),
            transparent=True
        )

    if show_plots:
        plt.show()


def volume_metric_sweeps():
    """
    Estimate the volume explored by a typical trajectory.
    """
    args = get_args(validate_source=False)
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    args.npas = npa_sigmas

    # Sweep over sim durations
    sim_durations = _get_durations_from_args(args)
    n_durations = len(sim_durations)
    args.sim_durations = sim_durations

    # Sweep over the pauses
    pauses = _get_pauses_from_args(args)
    n_pauses = len(pauses)
    args.pauses = pauses

    # Outputs
    r_values = np.zeros((n_sigmas, n_durations, n_pauses, 3))
    z_values = np.zeros((n_sigmas, n_durations, n_pauses, 3))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'Simulating exploration with sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}. ({sim_idx + 1}/{n_sims}).')

                args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Find the maximums in each relative directions
                Xt = SS.get_Xt()
                Xt_max = np.abs(Xt).max(axis=1)

                # Use first component as radius and third as height of explored disk
                r_values[i, j, k] = [Xt_max[:, 0].mean(), Xt_max[:, 0].min(), Xt_max[:, 0].max()]
                z_values[i, j, k] = [Xt_max[:, 2].mean(), Xt_max[:, 2].min(), Xt_max[:, 2].max()]

                if SS.needs_save:
                    SS.save()
                sim_idx += 1

    # Calculate the volumes
    sphere_vols = 4 / 3 * np.pi * r_values**3
    cap_vols = 1 / 3 * np.pi * (r_values - z_values)**2 * (2 * r_values + z_values)
    disk_vols = sphere_vols - 2 * cap_vols
    optimal_sigmas = npa_sigmas[disk_vols[..., 0].argmax(axis=0)]

    # Plot volumes explored against sigmas
    logger.info('Plotting volumes explored results.')
    fig, axes = plt.subplots(n_durations, n_pauses, figsize=(3 + n_pauses, 2 + n_durations), sharex=True,
                             sharey=True)
    fig.suptitle(f'Batch size={args.batch_size}.')
    for j, duration in enumerate(sim_durations):
        for k, pause in enumerate(pauses):
            ax = axes[j, k]
            if k == 0:
                ax.set_ylabel(f'T={duration:.2f}s')
            if j == 0:
                ax.set_title(f'$\delta$={pause:.2f}s')
            vols = disk_vols[:, j, k].T
            ax.errorbar(
                np.arange(n_sigmas),
                vols[0],
                yerr=[vols[0] - vols[1], vols[2] - vols[0]],
                marker='x',
                capsize=5,
            )
            ax.set_xticks(np.arange(n_sigmas))
            ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
            ax.tick_params(axis='x', rotation=270)
            ax.grid()
            ax.set_yscale('log')
    fig.tight_layout()
    if save_plots:
        plt.savefig(
            make_filename('volume_sweep_volumes', args,
                          excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
            transparent=True
        )
    if show_plots:
        plt.show()

    # Plot the peak in sigma of each duration/pause combination as a surface plot
    X, Y = np.meshgrid(sim_durations, pauses)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim=30, elev=25)
    ax.set_title('Optimal sigmas.')
    ax.scatter(X, Y, optimal_sigmas, c=optimal_sigmas, cmap='Reds', s=100, marker='x')
    ax.plot_surface(X, Y, optimal_sigmas.T, cmap='coolwarm', alpha=0.5)
    ax.set_xlabel('T')
    ax.set_ylabel('$\delta$')
    ax.set_zlabel('$\sigma_\phi$')
    fig.tight_layout()
    if save_plots:
        plt.savefig(
            make_filename('volume_sweep_surface', args,
                          excludes=['voxel_sizes', 'deltas', 'delta_step',
                                    'n_targets', 'epsilon', 'duration', 'max_nonplanar_pause_duration']),
            transparent=True
        )
    if show_plots:
        plt.show()

    # Plot peaks on a 2D plot
    fig, axes = plt.subplots(2, figsize=(10, 10))
    fig.suptitle(f'Optimal $\sigma_\phi$. Batch size={args.batch_size}.')
    for i in range(2):
        ax = axes[i]
        ax.set_ylabel('$\sigma_\phi$')
        if i == 0:
            for j, duration in enumerate(sim_durations):
                ax.plot(np.arange(n_pauses), optimal_sigmas[j], label=f'T={duration:.2f}')
            ax.set_xlabel('$\delta$')
            ax.set_xticks(np.arange(n_pauses))
            ax.set_xticklabels([f'{d:.1E}' for d in pauses])
        else:
            ax.set_xlabel('T')
            for j, pause in enumerate(pauses):
                ax.plot(np.arange(n_durations), optimal_sigmas[:, j], label=f'$\delta$={pause:.2E}')
            ax.set_xticks(np.arange(n_durations))
            ax.set_xticklabels([f'{T:.2f}' for T in sim_durations])
        ax.legend()
        ax.grid()
        ax.set_yscale('log')
    fig.tight_layout()
    if save_plots:
        plt.savefig(
            make_filename('volume_sweep_peaks', args,
                          excludes=['voxel_sizes', 'deltas', 'delta_step',
                                    'n_targets', 'epsilon', 'duration', 'max_nonplanar_pause_duration']),
            transparent=True
        )
    if show_plots:
        plt.show()


def _calculate_rz_values(
        args: Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the r and z values across a range of sigmas, durations and pauses.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)

    # Outputs
    r_values = np.zeros((n_sigmas, n_durations, n_pauses, 4))
    z_values = np.zeros((n_sigmas, n_durations, n_pauses, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'Simulating exploration with sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}. '
                    f'({sim_idx + 1}/{n_sims}).'
                )

                args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Find the maximums in each relative directions
                Xt = SS.get_Xt()
                Xt_max = np.abs(Xt).max(axis=1)

                # Use first component as radius and third as height of explored disk
                r_values[i, j, k] = [Xt_max[:, 0].mean(), Xt_max[:, 0].min(), Xt_max[:, 0].max(), Xt_max[:, 0].std()]
                z_values[i, j, k] = [Xt_max[:, 2].mean(), Xt_max[:, 2].min(), Xt_max[:, 2].max(), Xt_max[:, 2].std()]

                if SS.needs_save:
                    SS.save()
                sim_idx += 1

    return r_values, z_values


def _generate_or_load_rz_values(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    keys = {
        'npas': [f'{s:.3E}' for s in args.npas],
        'durations': [f'{d:.4f}' for d in args.sim_durations],
        'pauses': [f'{p:.4f}' for p in args.pauses],
    }
    cache_path = LOGS_PATH / hash_data(keys)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    if not rebuild_cache and cache_fn.exists():
        data = np.load(cache_fn)
        r_values = data['r_values']
        z_values = data['z_values']
        logger.info('Loaded r and z values from cache.')
    else:
        if cache_only:
            raise RuntimeError('Cache does not exist!')
        logger.info('Generating r and z values.')
        r_values, z_values = _calculate_rz_values(args)
        save_arrs = {
            'r_values': r_values,
            'z_values': z_values,
        }
        logger.info(f'Saving r and z values to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return r_values, z_values


def volume_metric_sweeps2():
    """
    Estimate the volume explored by a typical trajectory.
    """
    args = get_args(validate_source=False)
    model_phi = args.phi_dist_params[1]

    # Set parameter ranges
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    args.npas = npa_sigmas
    sim_durations = _get_durations_from_args(args)
    n_durations = args.durations_num
    pauses = _get_pauses_from_args(args)
    n_pauses = args.pauses_num

    fix_duration = args.sim_duration
    fix_pause = args.nonp_pause_max

    def _calculate_volumes(r_, z_):
        sphere_vols = 4 / 3 * np.pi * r_**3
        cap_vols = 1 / 3 * np.pi * (r_ - z_)**2 * (2 * r_ + z_)
        return sphere_vols - 2 * cap_vols

    if 1:
        # Fix the pause and sweep over the durations
        args.sim_durations = sim_durations
        args.pauses = [fix_pause]
        r_values, z_values = _generate_or_load_rz_values(args, rebuild_cache=False)

        start_idx = 1
        end_idx = n_durations - 1
        n_durations = end_idx - start_idx
        sim_durations_to_plot = sim_durations[start_idx:end_idx]
        sim_durations_to_plot = [f'{int(np.round(t / 60))} min' for t in sim_durations_to_plot]
        r_values = r_values[:, start_idx:end_idx]
        z_values = z_values[:, start_idx:end_idx]

        disk_vols = _calculate_volumes(r_values, z_values)
        optimal_sigmas_idxs = disk_vols[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]
        optimal_vols = []

        # Plot the volumes
        fig, ax = plt.subplots(1, figsize=(5, 3))
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, n_durations))
        for j, duration in enumerate(sim_durations_to_plot):
            vols = disk_vols[:, j, 0].T
            optimal_vols.append(vols[0, optimal_sigmas_idxs[j]])
            # ax.plot(npa_sigmas, vols[0], label=f'T={duration:.0f}s', c=colours[j], marker='o', alpha=0.7)
            ax.plot(npa_sigmas, vols[0], label=duration, c=colours[j], marker='o', alpha=0.7)
            ax.axhline(y=vols[0, optimal_sigmas_idxs[j]], color='red', zorder=-2, linewidth=2, linestyle=':')
        # ax.scatter(optimal_sigmas, optimal_vols, marker='o', zorder=100, s=50, facecolors='none',
        #            edgecolors='red', linewidths=2)

        ax.axvline(x=model_phi, c='orange', linestyle='--', linewidth=3, zorder=-1)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.set_title(f'$\delta_\max={fix_pause:.0f}$s')
        ax.set_xlabel(f'$\sigma_\phi$')
        ax.set_xscale('log')
        # ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
        ax.set_xticks([1e-3, 1e-1, 1e1])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(model_phi, -0.07, model_phi, color='orange', fontsize='large', fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=trans)
        ax.set_ylabel('Volume explored')
        # ax.set_yticks([0, 2500, 5000, 7500, 10000])
        # ax.set_yticks([0, 2000, 4000, 6000])
        # ax.grid()
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('volume_sweep_2a', args,
                              excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
                transparent=True
            )
        if show_plots:
            plt.show()

    if 1:
        # Fix the duration and sweep over the pauses
        args.sim_durations = [fix_duration]
        args.pauses = pauses
        r_values, z_values = _generate_or_load_rz_values(args, rebuild_cache=False)
        disk_vols = _calculate_volumes(r_values, z_values)
        optimal_sigmas_idxs = disk_vols[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]
        optimal_vols = []

        # r_values = np.zeros((n_sigmas, n_durations, n_pauses, 4))

        start_idx = 0
        end_idx = 3
        n_pauses = end_idx - start_idx
        pauses_to_plot = pauses[start_idx:end_idx]
        r_values = r_values[:, :, start_idx:end_idx]
        z_values = z_values[:, :, start_idx:end_idx]

        # Plot the volumes
        fig, ax = plt.subplots(1, figsize=(5, 3))
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, n_pauses))
        for k, pause in enumerate(pauses_to_plot):
            vols = disk_vols[:, 0, k].T
            optimal_vols.append(vols[0, optimal_sigmas_idxs[k]])
            ax.plot(npa_sigmas, vols[0], label=f'$\delta_\max={pause:.0f}$s', c=colours[k], marker='o', alpha=0.7)
            if k == 1:
                ax.axhline(y=vols[0, optimal_sigmas_idxs[k]], color='red', zorder=-2, linewidth=2, linestyle=':')
        # ax.scatter(optimal_sigmas, optimal_vols, marker='o', zorder=100, s=50, facecolors='none',
        #            edgecolors='red', linewidths=2)
        ax.axvline(x=model_phi, c='orange', linestyle='--', linewidth=3, zorder=-1)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.set_title(f'T={fix_duration}s')
        # ax.set_title(f'5 minute simulation time')
        ax.set_xlabel(f'$\sigma_\phi$')
        ax.set_xscale('log')
        ax.set_xticks([1e-3, 1e-1, 1e1])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(model_phi, -0.075, model_phi, color='orange', fontsize='large', fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=trans)
        ax.set_ylabel('Volume explored')
        # ax.set_yticks([0, 200, 400])

        ax.grid()
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('volume_sweep_2b', args,
                              excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
                transparent=True
            )
        if show_plots:
            plt.show()

    # Plot combined
    if 1:
        fig, axes = plt.subplots(2, figsize=(6, 6), gridspec_kw={
            'hspace': 0.4,
            'top': 0.93,
            'bottom': 0.1,
            'left': 0.13,
            'right': 0.8,
        })

        # Fix the pause and sweep over the durations
        args.sim_durations = sim_durations
        args.pauses = [fix_pause]
        r_values, z_values = _generate_or_load_rz_values(args, rebuild_cache=False)
        disk_vols = _calculate_volumes(r_values, z_values)
        optimal_sigmas_idxs = disk_vols[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]
        optimal_vols = []

        # Plot the volumes
        ax = axes[0]
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, len(sim_durations)))
        for j, duration in enumerate(sim_durations):
            vols = disk_vols[:, j, 0].T
            optimal_vols.append(vols[0, optimal_sigmas_idxs[j]])
            ax.plot(npa_sigmas, vols[0], label=f'T={duration:.0f}s', c=colours[j], marker='o', alpha=0.7)
        ax.scatter(optimal_sigmas, optimal_vols, marker='o', zorder=100, s=50, facecolors='none', edgecolors='red',
                   linewidths=2)
        ax.axvline(x=model_phi, c='orange', linestyle='--', linewidth=3, zorder=-1)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.set_title(f'$\delta_{{max}}={fix_pause:.1f}$s')
        ax.set_xlabel(f'$\sigma_\phi$')
        ax.set_xscale('log')
        # ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
        ax.set_xticks([1e-3, 1e-1, 1e1])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(model_phi, -0.06, model_phi, color='orange', fontsize='large', fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=trans)
        ax.set_ylabel('Volume explored')
        # ax.set_yticks([5000, 10000])
        ax.grid()

        # Fix the duration and sweep over the pauses
        args.sim_durations = [fix_duration]
        args.pauses = pauses
        r_values, z_values = _generate_or_load_rz_values(args, rebuild_cache=False)
        disk_vols = _calculate_volumes(r_values, z_values)
        optimal_sigmas_idxs = disk_vols[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]
        optimal_vols = []

        # Plot the volumes
        ax = axes[1]
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, len(pauses)))
        for k, pause in enumerate(pauses):
            vols = disk_vols[:, 0, k].T
            optimal_vols.append(vols[0, optimal_sigmas_idxs[k]])
            ax.plot(npa_sigmas, vols[0], label=f'$\delta$={pause:.1f}s', c=colours[k], marker='o', alpha=0.7)
        ax.scatter(optimal_sigmas, optimal_vols, marker='o', zorder=100, s=50, facecolors='none', edgecolors='red',
                   linewidths=2)
        ax.axvline(x=model_phi, c='orange', linestyle='--', linewidth=3, zorder=-1)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        ax.set_title(f'T={fix_duration}s')
        ax.set_xlabel(f'$\sigma_\phi$')
        ax.set_xscale('log')
        ax.set_xticks([1e-3, 1e-1, 1e1])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(model_phi, -0.06, model_phi, color='orange', fontsize='large', fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=trans)
        ax.set_ylabel('Volume explored')
        # ax.set_yticks([0, 4, 8, 12])
        ax.grid()

        # fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('volume_sweep_2c', args,
                              excludes=['voxel_sizes', 'deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
                transparent=True
            )
        if show_plots:
            plt.show()


def _calculate_voxel_scores_for_spec(params):
    args, npas, duration, pause, voxel_sizes = params
    args.phi_dist_params[1] = npas
    args.sim_duration = duration
    args.nonp_pause_max = pause
    pe, TC = get_sim_state_from_args(args)
    scores = np.zeros((len(voxel_sizes), 4))

    # Calculate optimality
    for l, vs in enumerate(voxel_sizes):
        vc = TC.get_coverage(vs) / vs
        scores[l] = [vc.mean(), vc.min(), vc.max(), vc.std()]
    if TC.needs_save:
        TC.save()

    return scores


def _calculate_voxel_scores(
        args: Namespace
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the voxel scores across a range of sigmas, durations, pauses and voxel sizes.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)
    voxel_sizes = args.voxel_sizes
    n_voxel_sizes = len(voxel_sizes)

    # Outputs
    scores = np.zeros((n_sigmas, n_durations, n_pauses, n_voxel_sizes, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'{sim_idx + 1}/{n_sims}: '
                    f'sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}.'
                )
                args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Calculate optimality
                for l, vs in enumerate(voxel_sizes):
                    vc = SS.get_coverage(vs) / vs
                    scores[i, j, k, l] = [vc.mean(), vc.min(), vc.max(), vc.std()]
                if SS.needs_save:
                    SS.save()

                sim_idx += 1

    return scores


def _generate_or_load_voxel_scores(
        args: Namespace,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    keys = {
        'npas': [f'{s:.3E}' for s in args.npas],
        'durations': [f'{d:.4f}' for d in args.sim_durations],
        'pauses': [f'{p:.4f}' for p in args.pauses],
        'vs': [f'{p:.3E}' for p in args.voxel_sizes],
    }
    cache_path = LOGS_PATH / hash_data(keys)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    if not rebuild_cache and cache_fn.exists():
        data = np.load(cache_fn)
        scores = data['scores']
        logger.info('Loaded scores from cache.')
    else:
        logger.info('Generating voxel coverage values.')
        scores = _calculate_voxel_scores(args)
        save_arrs = {
            'scores': scores,
        }
        logger.info(f'Saving voxel scores to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return scores


def voxel_scores_sweeps():
    """
    Simulate across a range of non-planarities and count unique voxels visited.
    """
    args = get_args(validate_source=False)
    model_phi = args.phi_dist_params[1]

    # Set parameter ranges
    npa_sigmas = _get_npas_from_args(args)
    n_sigmas = args.npas_num
    args.npas = npa_sigmas
    sim_durations = _get_durations_from_args(args)
    n_durations = args.durations_num
    pauses = _get_pauses_from_args(args)
    n_pauses = args.pauses_num
    voxel_sizes = _get_voxel_sizes_from_args(args)
    args.voxel_sizes = voxel_sizes
    n_vs = args.vxs_num

    fix_pause = args.nonp_pause_max
    fix_duration = args.sim_duration

    if 1:
        # Fix the pause and sweep over the durations
        args.sim_durations = sim_durations
        args.pauses = [fix_pause]
        scores = _generate_or_load_voxel_scores(args, rebuild_cache=False)
        optimal_sigmas_idxs = scores[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]

        # scores = np.zeros((n_sigmas, n_durations, n_pauses, n_voxel_sizes, 4))

        # Plot the scores
        fig, axes = plt.subplots(n_vs, figsize=(10, 5 * n_vs), squeeze=False)
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, n_durations))

        for l, vs in enumerate(voxel_sizes):
            ax = axes[l, 0]
            optimal_scores = []
            for j, duration in enumerate(sim_durations):
                score = scores[:, j, 0, l, 0]
                optimal_scores.append(score[optimal_sigmas_idxs[j, l]])
                ax.plot(npa_sigmas, score, label=f'T={duration:.0f}s', c=colours[j], marker='o', alpha=0.7)
            ax.scatter(optimal_sigmas[:, l], optimal_scores, c='red', marker='o', zorder=100, s=50)
            ax.axvline(x=model_phi, c='orange', linestyle='--')
            ax.legend()
            ax.set_title(f'$\delta$={fix_pause:.1f}s. Voxel size={vs:.3E}mm.')
            ax.set_xlabel(f'$\sigma_\phi$')
            ax.set_xticks(np.arange(n_sigmas))
            ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
            ax.set_ylabel('Average voxels visited')
            ax.set_xscale('log')
            ax.grid()

        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('voxels_sweep_a', args,
                              excludes=['deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
                transparent=True
            )
        if show_plots:
            plt.show()

    if 1:
        # Fix the duration and sweep over the pauses
        args.sim_durations = [fix_duration]
        args.pauses = pauses
        scores = _generate_or_load_voxel_scores(args, rebuild_cache=False)
        optimal_sigmas_idxs = scores[..., 0].argmax(axis=0).squeeze()
        optimal_sigmas = npa_sigmas[optimal_sigmas_idxs]

        # scores = np.zeros((n_sigmas, n_durations, n_pauses, n_voxel_sizes, 4))

        # Plot the scores
        fig, axes = plt.subplots(n_vs, figsize=(10, 5 * n_vs), squeeze=False)
        cmap = plt.get_cmap('winter')
        colours = cmap(np.linspace(0, 1, n_pauses))

        for l, vs in enumerate(voxel_sizes):
            ax = axes[l, 0]
            optimal_scores = []
            for k, pause in enumerate(pauses):
                score = scores[:, 0, k, l, 0]
                optimal_scores.append(score[optimal_sigmas_idxs[k, l]])
                ax.plot(npa_sigmas, score, label=f'$\delta$={pause:.1f}s', c=colours[k], marker='o', alpha=0.7)
            ax.scatter(optimal_sigmas[:, l], optimal_scores, c='red', marker='o', zorder=100, s=50)
            ax.axvline(x=model_phi, c='orange', linestyle='--')
            ax.legend()
            ax.set_title(f'T={fix_duration}s. Voxel size={vs:.3E}mm.')
            ax.set_xlabel(f'$\sigma_\psi$')
            ax.set_xticks(np.arange(n_sigmas))
            ax.set_xticklabels([f'{npa:.1E}' for npa in args.npas])
            ax.set_ylabel('Average voxels visited')
            ax.set_xscale('log')
            ax.grid()

        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('voxel_sweep_b', args,
                              excludes=['deltas', 'delta_step', 'n_targets', 'epsilon', 'duration']),
                transparent=True
            )
        if show_plots:
            plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    # coverage_scores()
    # search_scores()  # broken
    # search_t_tests()  # broken
    surface_coverage_scores()
    crossings_nonp()
    volume_metric()
    volume_metric_sweeps()
    volume_metric_sweeps2()
    voxel_scores_sweeps()
