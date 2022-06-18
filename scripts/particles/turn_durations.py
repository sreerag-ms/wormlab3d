import os

import matplotlib.pyplot as plt
import numpy as np

from simple_worm.plot3d import Arrow3D
from wormlab3d import START_TIMESTAMP, LOGS_PATH, logger
from wormlab3d.data.model import Trial, Dataset
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca, tex_mode
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.statistics import calculate_trial_run_statistics, calculate_trial_turn_statistics

show_plots = True
save_plots = False
img_extension = 'svg'
tex_mode()


def plot_trial_turns():
    """
    Plot trial turns.
    """
    args = get_args()
    assert args.trial is not None
    assert args.trajectory_point is None  # Use the full postures
    assert args.planarity_window is not None
    trial = Trial.objects.get(id=args.trial)
    dt = 1 / trial.fps
    X = get_trajectory_from_args(args)
    window_size = int(1 / dt)
    smooth_K = 101
    height = 50

    arrow_opts = {
        'mutation_scale': 10,
        'arrowstyle': '-|>',
        'linewidth': 2,
        'alpha': 0.7
    }

    if save_plots:
        turn_plots_dir = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_ws={window_size:.2f}_s={smooth_K}'
        os.makedirs(turn_plots_dir, exist_ok=True)

    stats = calculate_trial_turn_statistics(args, smooth_K, window_size, height)
    k = stats['k']
    vertices = stats['vertices']
    tumble_idxs = stats['tumble_idxs']
    distances = stats['distances']
    thetas = stats['thetas']
    phis = stats['thetas']
    psis = stats['psis']
    etas = stats['etas']
    nonp = stats['nonp']

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Plot 3D turns
    for i, tumble_idx in enumerate(tumble_idxs):
        start_idx = stats['start_idxs'][i]
        end_idx = stats['end_idxs'][i]
        X_window = X[start_idx:end_idx]
        pca = stats['pcas'][i]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(
            f'Trial={trial.id}. '
            f'Frames={start_idx}-{end_idx}/{len(X)}.\n'
            f'$\\theta={thetas[i] / np.pi:.2f}\pi$. '
            f'$\phi={phis[i] / np.pi:.2f}\pi$. '
            f'$\psi={psis[i] / np.pi:.2f}\pi$. '
            f'$\eta={etas[i] / np.pi:.2f}\pi$. '
            f'$d={distances[i]:.3f}$mm. '
            f'$NP={nonp[i]:.3f}$.'
        )
        x, y, z = X_window.T
        ax.scatter(x, y, z, c=k[start_idx:end_idx], cmap='Reds', s=100, alpha=1, zorder=1)
        x, y, z = vertices[i + 1].T
        ax.scatter(x, y, z, color='blue', marker='x', s=700, alpha=0.7, zorder=1000, linewidth=3)
        plane = make_box_from_pca(X_window, pca, 'blue', scale=(0.1, 0.2, 0.3))
        ax.add_collection3d(plane)
        ax.add_artist(Arrow3D(origin=X[start_idx], vec=stats['v_in'][i], color='darkred', **arrow_opts))
        ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=stats['v_out'][i], color='red', **arrow_opts))
        ax.add_artist(Arrow3D(origin=X[start_idx], vec=stats['v_in_theta'][i], color='darkgreen', **arrow_opts))
        ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=stats['v_out_theta'][i], color='green', **arrow_opts))
        ax.add_artist(Arrow3D(origin=X[start_idx], vec=stats['v_in_phi'][i], color='darkblue', **arrow_opts))
        ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=stats['v_out_phi'][i], color='blue', **arrow_opts))
        ax.add_artist(Arrow3D(origin=X[start_idx], vec=stats['v_in_psi'][i], color='purple', **arrow_opts))
        ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=stats['v_out_psi'][i], color='violet', **arrow_opts))
        equal_aspect_ratio(ax)
        fig.tight_layout()

        # Save / show
        if save_plots:
            plt.savefig(
                turn_plots_dir / f'{tumble_idx:06d}.{img_extension}',
                transparent=True
            )
        if show_plots:
            plt.show()


def plot_trial_turn_correlations():
    """
    Plot trial turn correlations.
    """
    args = get_args()
    assert args.trial is not None
    assert args.trajectory_point is None  # Use the full postures
    assert args.planarity_window is not None
    trial = Trial.objects.get(id=args.trial)
    dt = 1 / trial.fps
    window_size = int(1 / dt)
    smooth_K = 101
    height = 50

    stats = calculate_trial_turn_statistics(args, smooth_K, window_size, height)
    distances = stats['distances']
    thetas = stats['thetas']
    phis = stats['thetas']
    psis = stats['psis']
    etas = stats['etas']
    nonp = stats['nonp']

    # Plot correlations
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    fig.suptitle(
        f'Trial={trial.id}. '
        f'Window size={window_size:.2f}. '
        f'smooth\_K={smooth_K}. '
        f'height={height}. '
    )

    ax = axes[0, 0]
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$d$')
    ax.scatter(thetas, distances, s=2)

    ax = axes[0, 1]
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$d$')
    ax.scatter(phis, distances, s=2)

    ax = axes[1, 0]
    ax.set_xlabel('$\psi$')
    ax.set_ylabel('$d$')
    ax.scatter(psis, distances, s=2)

    ax = axes[1, 1]
    ax.set_xlabel('$\\text{MIN}(\phi,\psi)$')
    ax.set_ylabel('$d$')
    ax.scatter(np.min(np.stack([phis, psis]), axis=0), distances, s=2)

    ax = axes[0, 2]
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$d$')
    ax.scatter(etas, distances, s=2)

    ax = axes[1, 2]
    ax.set_xlabel('$NP$')
    ax.set_ylabel('$d$')
    ax.scatter(nonp, distances, s=2)

    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_ws={window_size:.2f}_s={smooth_K}.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


def plot_dataset_turn_correlations():
    """
    Plot the dataset turn durations.
    """
    args = get_args()
    assert args.dataset is not None
    assert args.planarity_window is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.dataset = None
    dt = 1 / 25
    window_size = int(1 / dt)
    smooth_K = 101
    height = 50

    args.tracking_only = True

    # Outputs
    distances = []
    thetas = []
    phis = []
    psis = []
    etas = []
    nonp = []

    # Calculate the model for all trials
    for trial in ds.include_trials:
        args.trial = trial.id
        try:
            stats = calculate_trial_turn_statistics(args, smooth_K, window_size, height)
        except RuntimeError as e:
            logger.warning(f'Failed to find approximation: "{e}"')
        distances.append(stats['distances'])
        thetas.append(stats['thetas'])
        phis.append(stats['phis'])
        psis.append(stats['psis'])
        etas.append(stats['etas'])
        nonp.append(stats['nonp'])

    n_trajectories = len(distances)
    logger.info(f'Calculated turn statistics for {n_trajectories} out of a possible {len(ds.include_trials)}.')

    # Join outputs
    distances = np.concatenate(distances)
    thetas = np.concatenate(thetas)
    phis = np.concatenate(phis)
    psis = np.concatenate(psis)
    etas = np.concatenate(etas)
    nonp = np.concatenate(nonp)

    # Plot correlations
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    fig.suptitle(
        f'Dataset={ds.id}. '
        f'Window size={window_size:.2f}. '
        f'smooth\_K={smooth_K}. '
        f'height={height}. '
    )

    ax = axes[0, 0]
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$d$')
    ax.scatter(thetas, distances, s=2)

    ax = axes[0, 1]
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$d$')
    ax.scatter(phis, distances, s=2)

    ax = axes[1, 0]
    ax.set_xlabel('$\psi$')
    ax.set_ylabel('$d$')
    ax.scatter(psis, distances, s=2)

    ax = axes[1, 1]
    ax.set_xlabel('$\\text{MIN}(\phi,\psi)$')
    ax.set_ylabel('$d$')
    ax.scatter(np.min(np.stack([phis, psis]), axis=0), distances, s=2)

    ax = axes[0, 2]
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$d$')
    ax.scatter(etas, distances, s=2)

    ax = axes[1, 2]
    ax.set_xlabel('$NP$')
    ax.set_ylabel('$d$')
    ax.scatter(nonp, distances, s=2)

    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_ds={ds.id}_ws={window_size:.2f}_s={smooth_K}.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


def plot_dataset_turn_correlations_across_windows():
    """
    Plot the dataset turn correlations across a range of different windows.
    """
    args = get_args()
    assert args.dataset is not None
    assert args.planarity_window is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.dataset = None
    dt = 1 / 25
    window_sizes = [1, 2, 4, 8]
    smooth_K = 101
    height = 100

    args.tracking_only = True

    # Outputs
    distances = [[] for _ in window_sizes]
    nonp = [[] for _ in window_sizes]
    speeds = [[] for _ in window_sizes]

    # Calculate the model for all trials at all window sizes
    for trial in ds.include_trials:
        args.trial = trial.id
        for i, ws in enumerate(window_sizes):
            try:
                stats = calculate_trial_turn_statistics(args, smooth_K, int(ws / dt), height)
            except RuntimeError as e:
                logger.warning(f'Failed to find approximation: "{e}"')
            distances[i].append(stats['distances'])
            nonp[i].append(stats['nonp'])
            speeds[i].append(stats['speeds'])

    n_trajectories = sum([len(dists) for dists in distances])
    logger.info(
        f'Calculated turn statistics for {n_trajectories} out of a possible {len(ds.include_trials) * len(window_sizes)}.')

    # Join outputs
    distances = [np.concatenate(dists) for dists in distances]
    nonp = [np.concatenate(nonps) for nonps in nonp]

    # Calculate distances
    speeds = []
    for i, ws in enumerate(window_sizes):
        speeds.append(distances[i] / ws)

    # Plot correlations
    fig, axes = plt.subplots(2, len(window_sizes), figsize=(len(window_sizes) * 3, 8))
    fig.suptitle(
        f'Dataset={ds.id}.\n'
        f'smooth\_K={smooth_K}. '
        f'height={height}. '
    )

    for i, ws in enumerate(window_sizes):
        ax = axes[0, i]
        ax.set_title(f'Window={ws * 2}s.')
        ax.set_xlabel('NP')
        ax.set_ylabel('distance')
        ax.scatter(nonp[i], distances[i], s=1)

        ax = axes[1, i]
        ax.set_xlabel('NP')
        ax.set_ylabel('speed')
        ax.scatter(nonp[i], speeds[i], s=1)

    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}'
                        f'_ds={ds.id}'
                        f'_ws={",".join([str(ws) for ws in window_sizes])}'
                        f'_s={smooth_K}.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


def plot_dataset_run_stats():
    """
    Plot the dataset run stats.
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.dataset = None
    args.tracking_only = True
    dt = 1 / 25
    min_run_duration = int(1 / dt)
    smooth_K = 101
    k_max = 100

    # Outputs
    distances = []
    speeds = []
    nonp = []

    # Calculate the run stats for all trials
    for trial in ds.include_trials:
        args.trial = trial.id
        try:
            stats = calculate_trial_run_statistics(args, smooth_K, k_max, min_run_duration)
        except RuntimeError as e:
            logger.warning(f'Failed to calculate run stats: "{e}"')
        distances.append(stats['distances'])
        speeds.append(stats['speeds'])
        nonp.append(stats['nonp'])

    n_trajectories = len(distances)
    logger.info(f'Calculated turn statistics for {n_trajectories} out of a possible {len(ds.include_trials)}.')

    # Join outputs
    distances = np.concatenate(distances)
    speeds = np.concatenate(speeds)
    nonp = np.concatenate(nonp)

    # Plot correlations
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    fig.suptitle(
        f'Dataset={ds.id}. '
        f'Min. run duration={min_run_duration:.2f}. '
        f'smooth\_K={smooth_K}. '
        f'Max curvature={k_max}. '
    )

    ax = axes[0, 0]
    ax.set_xlabel('distances')
    ax.set_ylabel('speeds')
    ax.scatter(distances, speeds, s=2)

    ax = axes[0, 1]
    ax.set_xlabel('distances')
    ax.set_ylabel('NP')
    ax.scatter(distances, nonp, s=2)

    ax = axes[0, 2]
    ax.set_xlabel('speeds')
    ax.set_ylabel('NP')
    ax.scatter(speeds, nonp, s=2)

    ax = axes[1, 0]
    ax.set_title('Distances')
    ax.hist(distances, density=True, bins=20)

    ax = axes[1, 1]
    ax.set_title('Speeds')
    ax.hist(speeds, density=True, bins=20)

    ax = axes[1, 2]
    ax.set_title('Non-planarity')
    ax.hist(nonp, density=True, bins=20)

    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_runs_ds={ds.id}_d={min_run_duration:.2f}_k_max={k_max}_s={smooth_K}.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # plot_trial_turns()
    # plot_trial_turn_correlations()
    plot_dataset_turn_correlations()
    plot_dataset_turn_correlations_across_windows()
    plot_dataset_run_stats()
    # plot_dataset_turns_vs_runs()
