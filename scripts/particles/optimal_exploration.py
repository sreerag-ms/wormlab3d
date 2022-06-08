import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
from scipy.stats import ttest_ind

from simple_worm.plot3d import MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.cache import get_trajectories_from_args
from wormlab3d.toolkit.plot_utils import tex_mode
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import get_deltas_from_args

plot_n_examples = 2
show_plots = False
save_plots = True
img_extension = 'svg'

tex_mode()


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
              'targets_radii', 'n_targets', 'epsilon', 'max_nonplanar_pause_duration']:
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
        elif k == 'targets_radii':
            if len(args.targets_radii) > 5:
                targets_radii = f'{args.targets_radii[0]:.1E}-{args.targets_radii[-1]:.1E}'
            else:
                targets_radii = ','.join(f'{r:.1E}' for r in args.targets_radii)
            fn += f'_r={targets_radii}'
        elif k == 'n_targets':
            fn += f'_targets={args.n_targets}'
        elif k == 'epsilon':
            fn += f'_eps={args.epsilon}'
        elif k == 'max_nonplanar_pause_duration':
            fn += f'_p={args.nonp_pause_max:.1f}'

    return LOGS_PATH / (fn + '.' + extension)


def coverage_scores():
    """
    Simulate across a range of non-planarities and score trajectories based on how many unique voxels have been visited.
    """
    args = get_args(validate_source=False)
    deltas, delta_ts = get_deltas_from_args(args)

    # npa_sigmas = [1e-8, 10]
    # npa_sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = np.linspace(0.00001, 10, 20)
    # npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 20))
    npa_sigmas = np.exp(-np.linspace(np.log(1 / 1e-6), np.log(1 / 10), 12))
    # npa_sigmas = [1e-6, 10,]
    n_sigmas = len(npa_sigmas)

    # voxel_sizes = [1, 0.1, 0.01, 0.001]
    # voxel_sizes = np.exp(-np.linspace(np.log(1/1), np.log(1/0.01), 20))
    voxel_sizes = np.exp(-np.linspace(np.log(1 / 0.1), np.log(1 / 0.01), 6))
    # voxel_sizes = [0.1,]
    n_vs = len(voxel_sizes)

    # Outputs
    scores = np.zeros((n_sigmas, n_vs, args.batch_size))
    nonp = np.zeros((n_sigmas, args.batch_size))
    msds_all = {i: {} for i in range(n_sigmas)}
    msds = {i: {j: {} for j in range(args.batch_size)} for i in range(n_sigmas)}
    Xs_all = []

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i + 1}/{n_sigmas}).')
        args.phi_dist_params[1] = npas
        pe, TC = get_trajectories_from_args(args)
        Xs_all.append(TC.X)

        # Calculate non-planarity for validation
        nonp[i] = TC.get_nonp()
        if TC.needs_save:
            TC.save()

        # Calculate optimality
        logger.info('Calculating coverage.')
        bar = Bar('Calculating', max=n_vs)
        bar.check_tty = False
        for k, vs in enumerate(voxel_sizes):
            scores[i, k] = TC.get_coverage(vs)
            bar.next()
        bar.finish()
        if TC.needs_save:
            TC.save()

        # Calculate MSDs
        logger.info(f'Calculating displacements.')
        msds_all[i], msds[i] = TC.get_msds(deltas)
        if TC.needs_save:
            TC.save()
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
        msd_vals = np.array(list(msds_all[i].values()))
        ax.plot(delta_ts, msd_vals, label=f'$\sigma$={npas:.1E}',
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
        pe, TC = get_trajectories_from_args(args)

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
        pe, TC = get_trajectories_from_args(args)

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


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    # coverage_scores()
    # search_scores()
    search_t_tests()
