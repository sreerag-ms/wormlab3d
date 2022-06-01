import os
from argparse import Namespace
from decimal import Decimal
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib.gridspec import GridSpec
from progress.bar import Bar
from scipy.spatial.distance import cdist
from scipy.stats import rv_continuous, norm
from sklearn.decomposition import PCA

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.gaussian_mixture import GaussianMixtureScipy
from wormlab3d.particles.three_state_explorer import ThreeStateExplorer
from wormlab3d.particles.util import plot_msd
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, tex_mode
from wormlab3d.toolkit.util import normalise
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import get_deltas_from_args

plot_n_examples = 2
show_plots = True
save_plots = True
img_extension = 'svg'

tex_mode()


def make_filename(method: str, args: Namespace, excludes: List[str] = None, extension:str = img_extension):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'


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
            fn += f'_bs={args.sim_batch_size}'
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
            fn += f'_p={args.max_nonplanar_pause_duration:.1f}'

    return LOGS_PATH / (fn + '.' + extension)




def _get_explorer(
        batch_size: int,
        nonplanar_angles_sigma: float,
        max_nonplanar_pause_duration: float = 0
) -> ThreeStateExplorer:
    # Set up explorer
    speeds_0_mu = 0.002
    speeds_0_sig = 0.0005
    speeds_1_mu = 0.007
    speeds_1_sig = 0.001
    speeds_0_dist = norm(loc=speeds_0_mu, scale=speeds_0_sig)
    speeds_1_dist = norm(loc=speeds_1_mu, scale=speeds_1_sig)
    speeds0 = np.abs(speeds_0_dist.rvs(batch_size))
    speeds1 = np.abs(speeds_1_dist.rvs(batch_size))

    speeds0 = 0.0001
    speeds1 = 0.001

    pe = ThreeStateExplorer(
        batch_size=batch_size,
        rate_01=0.05,
        rate_10=0.1,
        rate_02=0.005,
        rate_20=0.8,  # not really a rate!
        speed_0=speeds0,  # 0.0001,
        speed_1=speeds1,  # 0.007,
        planar_angle_dist_params={
            # 'type': '2norm',
            # 'params': (1, 0, 1.5, 0.2, np.pi, 0.5)
            'type': 'norm',
            'params': (0, 10)
        },
        nonplanar_angle_dist_params={
            'type': 'norm',
            'params': (0, nonplanar_angles_sigma)
        },
        max_nonplanar_pause_duration=max_nonplanar_pause_duration
    )

    return pe



def coverage_scores():
    """
    Simulate across a range of non-planarities and score trajectories based on how many unique voxels have been visited.
    """
    args = get_args(validate_source=False)
    deltas, delta_ts = get_deltas_from_args(args)

    # todo: put in args
    T = 60 * 60
    dt = 1 / 25
    batch_size = 200
    max_nonplanar_pause_duration = 5
    # npa_sigmas = [1e-8, 10]
    # npa_sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = np.linspace(0.00001, 10, 20)
    # npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 20))
    npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 12))
    # npa_sigmas = [1e-6, 10,]
    N = len(npa_sigmas)

    # voxel_sizes = [1, 0.1, 0.01, 0.001]
    # voxel_sizes = np.exp(-np.linspace(np.log(1/1), np.log(1/0.01), 20))
    voxel_sizes = np.exp(-np.linspace(np.log(1/0.1), np.log(1/0.01), 6))
    # voxel_sizes = [0.1,]
    M = len(voxel_sizes)

    # Outputs
    scores = np.zeros((N, M, batch_size))
    nonp = np.zeros((N, batch_size))
    msds_all = {i: {} for i in range(N)}
    msds = {i: {j: {} for j in range(batch_size)} for i in range(N)}

    Xs_all = []

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i+1}/{N}).')
        pe = _get_explorer(batch_size, npas, max_nonplanar_pause_duration)
        ts, tumble_ts, Xs, states, durations, planar_angles, nonplanar_angles, intervals, speeds = pe.forward(T, dt)
        Xs_all.append(Xs)

        # Calculate non-planarity for validation
        logger.info('Calculating non-planarity of trajectories.')
        for j, X in enumerate(Xs):
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            pca.fit(X)
            r = pca.explained_variance_ratio_.T
            nonp[i, j] = r[2] / np.sqrt(r[1] * r[0])

        # Calculate optimality
        logger.info('Calculating scores.')
        bar = Bar('Calculating', max=M)
        bar.check_tty = False
        for k, vs in enumerate(voxel_sizes):

            # Discretise the trajectories
            Xd = np.round(Xs.numpy()/vs).astype(np.int32)

            # Score the trajectories as the sum of unique voxels visited multiplied by voxel size
            for j, X in enumerate(Xd):
                n_voxels = np.unique(X, axis=0).shape[0]
                scores[i, k, j] = n_voxels * vs
            bar.next()
        bar.finish()

        # Calculate MSDs
        logger.info(f'Calculating displacements.')
        bar = Bar('Calculating', max=len(deltas))
        bar.check_tty = False
        for delta in deltas:
            d = torch.sum((Xs[:, delta:] - Xs[:, :-delta])**2, dim=-1)
            msds_all[i][delta] = d.mean()
            for j in range(batch_size):
                msds[i][j][delta] = d[j].mean()
            bar.next()
        bar.finish()
        logger.info('----')

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(3, figsize=(14, 12))
    fig.suptitle(f'T={T}s. dt={dt:.2f}. Batch size={batch_size}. Pause={max_nonplanar_pause_duration}s')
    cmap = plt.get_cmap('jet')

    # Scores
    ax = axes[0]
    ax.set_title('Scores')
    colours = cmap(np.linspace(0,1,M))
    for k, vs in enumerate(voxel_sizes):
        sk = scores[:, k]
        sk_mean = sk.mean(axis=-1)
        ax.errorbar(
            npa_sigmas,
            sk_mean,
            yerr=[sk_mean-sk.min(axis=-1), sk.max(axis=-1)-sk_mean],
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
        yerr=[nonp_mean-nonp.min(axis=-1), nonp.max(axis=-1)-nonp_mean],
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
    js = np.linspace(0, 1, batch_size + 4)[2:-2]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    colours = cmap(np.linspace(0,1,N))
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
        args.sim_duration = T
        args.sim_dt = dt
        args.sim_batch_size = batch_size
        args.max_nonplanar_pause_duration = max_nonplanar_pause_duration
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

    # todo: put in args
    T = 30 * 60
    dt = 1 / 25
    batch_size = 500
    max_nonplanar_pause_duration = 5
    # npa_sigmas = [1e-8, 10]
    # npa_sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # npa_sigmas = np.linspace(0.00001, 10, 20)
    # npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 20))
    npa_sigmas = np.exp(-np.linspace(np.log(1/1e-6), np.log(1/10), 8))
    n_sigmas = len(npa_sigmas)

    targets_radii = np.linspace(0, 6, 25)
    n_radii = len(targets_radii)

    n_targets = 1000  # number of targets at each radius
    epsilon = 0.1  # distance below which trajectory has "found" target

    # todo: this properly
    args.npas = npa_sigmas
    args.targets_radii = targets_radii
    args.n_targets = n_targets
    args.epsilon = epsilon
    args.sim_duration = T
    args.sim_dt = dt
    args.sim_batch_size = batch_size
    args.max_nonplanar_pause_duration = max_nonplanar_pause_duration

    # Outputs
    find_times = np.ones((n_sigmas, n_radii, batch_size, n_targets)) * -1
    finds = np.zeros((n_sigmas, n_radii, batch_size))
    finds_pop = np.zeros((n_sigmas, n_radii, n_targets), dtype=np.bool)

    # Generate targets using the golden spiral method
    #  https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    idxs = np.arange(n_targets) + 0.5
    phi = np.arccos(1 - 2 * idxs / n_targets)
    theta = np.pi * (1 + 5**0.5) * idxs
    targets = np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ], axis=1)

    # Sweep over the nonplanarity angle sigmas
    for i, npas in enumerate(npa_sigmas):
        logger.info(f'Simulating exploration with nonplanar angles sigma = {npas:.2E} ({i+1}/{n_sigmas}).')
        pe = _get_explorer(batch_size, npas, max_nonplanar_pause_duration)
        ts, tumble_ts, Xs, states, durations, planar_angles, nonplanar_angles, intervals, speeds = pe.forward(T, dt)

        logger.info('Finding targets.')
        bar = Bar('Finding', max=n_radii)
        bar.check_tty = False
        for j, r in enumerate(targets_radii):
            for k, X in enumerate(Xs):
                dists = cdist(X, targets * r)
                found_idxs = (dists < epsilon).nonzero()

                found_targets = np.unique(found_idxs[1])
                finds[i, j, k] = len(found_targets)

                for l, p in enumerate(targets):
                    if l in found_targets:
                        find_times[i, j, k, l] = found_idxs[0][found_idxs[1] == l][0] * dt
                        finds_pop[i, j, l] = True
            bar.next()
        bar.finish()
        logger.info('----')

    # Save result data
    np.savez_compressed(
        make_filename('search_results', args, excludes=['voxel_sizes', 'deltas', 'delta_step']),
        targets_radii=targets_radii,
        find_times=find_times,
        finds=finds,
        finds_pop=finds_pop,
    )

    # Plot results
    logger.info('Plotting results.')
    fig, axes = plt.subplots(3, figsize=(14, 12))
    fig.suptitle(f'T={T}s. dt={dt:.2f}. Batch size={batch_size}. '
                 f'Detection radius={epsilon}. '
                 f'Num. targets={n_targets}. '
                 f'Pause={max_nonplanar_pause_duration}s.')
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0,1,n_sigmas))

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
            yerr=[fi_mean-fi.min(axis=-1), fi.max(axis=-1)-fi_mean],
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
            yerr=[fti_mean-fti_min, fti_max-fti_mean],
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


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    from simple_worm.plot3d import interactive
    interactive()
    # coverage_scores()
    search_scores()
