import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from simple_worm.plot3d import Arrow3D
from wormlab3d import START_TIMESTAMP, LOGS_PATH, logger
from wormlab3d.data.model import Trial, Dataset
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca, tex_mode
from wormlab3d.toolkit.util import orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args

show_plots = True
save_plots = True
img_extension = 'svg'
tex_mode()


def _calculate_trial_turn_statistics(
        args: Namespace,
        smooth_K: int,
        window_size: int,
        curvature_height: int
):
    """
    Calculate the angles and distances for each turn in a trial trajectory.
    """
    logger.info(f'Calculating trial turn statistics for trial={args.trial}.')
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Calculate the approximation, tumbles and runs
    k = calculate_curvature(e0, smooth_e0=smooth_K, smooth_K=smooth_K)
    approx = get_approximate(X, k, distance=window_size, height=curvature_height)
    X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

    # Set up outputs
    start_idxs = []
    end_idxs = []
    distances = np.zeros(len(tumble_idxs))
    pcas = []
    thetas = np.zeros(len(tumble_idxs))
    phis = np.zeros(len(tumble_idxs))
    psis = np.zeros(len(tumble_idxs))
    nonp = np.zeros(len(tumble_idxs))
    etas = np.zeros(len(tumble_idxs))
    v_in = np.zeros((len(tumble_idxs), 3))
    v_out = np.zeros((len(tumble_idxs), 3))
    v_in_theta = np.zeros((len(tumble_idxs), 3))
    v_out_theta = np.zeros((len(tumble_idxs), 3))
    v_in_phi = np.zeros((len(tumble_idxs), 3))
    v_out_phi = np.zeros((len(tumble_idxs), 3))
    v_in_psi = np.zeros((len(tumble_idxs), 3))
    v_out_psi = np.zeros((len(tumble_idxs), 3))

    # Recalculate the angles using the full trajectory and a fixed PCA window
    for i, tumble_idx in enumerate(tumble_idxs):
        # Discard first and last tumbles
        if i == 0 or i == len(tumble_idxs) - 1:
            continue

        # Get a fixed time window around the turn
        start_idx = max(0, tumble_idx - window_size)
        start_idxs.append(start_idx)
        end_idx = min(X.shape[0] - 1, tumble_idx + window_size)
        end_idxs.append(end_idx)
        X_window = X[start_idx:end_idx]

        # Calculate distance travelled across windowed trajectory
        distances[i] = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()

        # Calculate the PCA for the complete window
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)
        pcas.append(pca)

        # Nonplanarity of the manoeuvre window
        r = pca.explained_variance_ratio_.T
        nonp[i] = r[2] / np.sqrt(r[1] * r[0])

        # Calculate the PCA for the windows before and after the turn
        pca_in = PCA(svd_solver='full', copy=True, n_components=3)
        pca_in.fit(X[start_idx:tumble_idx])
        pca_out = PCA(svd_solver='full', copy=True, n_components=3)
        pca_out.fit(X[tumble_idx:end_idx])

        # Angle between incoming/outgoing planes
        etas[i] = min(
            calculate_angle(pca_in.components_[2], pca_out.components_[2]),
            calculate_angle(pca_in.components_[2], -pca_out.components_[2])
        )

        # Incoming and outgoing trajectories
        v_in_i = X[tumble_idx] - X[start_idx]
        v_out_i = X[end_idx] - X[tumble_idx]
        v_io = v_out_i - v_in_i

        # Project onto the principal plane to find planar angle.
        v_in_theta_i = orthogonalise(v_in_i, pca.components_[2])
        v_out_theta_i = orthogonalise(v_out_i, pca.components_[2])
        if np.linalg.norm(v_io) < max(np.linalg.norm(v_in_i), np.linalg.norm(v_out_i)):
            thetas[i] = min(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))
        else:
            thetas[i] = max(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))

        # Orthogonalise against 2nd principal component to find non-planar angle.
        v_in_phi_i = orthogonalise(v_in_i, pca.components_[1])
        v_out_phi_i = orthogonalise(v_out_i, pca.components_[1])
        # phis[i] = calculate_angle(v_in_phi, v_out_phi)
        phis[i] = min(calculate_angle(v_in_phi_i, v_out_phi_i), calculate_angle(v_in_phi_i, -v_out_phi_i))

        # Orthogonalise against 1st principal component to find the final angle.
        v_in_psi_i = orthogonalise(v_in_i, pca.components_[0])
        v_out_psi_i = orthogonalise(v_out_i, pca.components_[0])
        # psis[i] = calculate_angle(v_in_psi, v_out_psi)
        psis[i] = min(calculate_angle(v_in_psi_i, v_out_psi_i), calculate_angle(v_in_psi_i, -v_out_psi_i))

        v_in[i] = v_in_i
        v_out[i] = v_out_i
        v_in_theta[i] = v_in_theta_i
        v_out_theta[i] = v_out_theta_i
        v_in_phi[i] = v_in_phi_i
        v_out_phi[i] = v_out_phi_i
        v_in_psi[i] = v_in_psi_i
        v_out_psi[i] = v_out_psi_i

    return {
        'k': k,
        'vertices': vertices,
        'tumble_idxs': tumble_idxs,
        'start_idxs': start_idxs,
        'end_idxs': end_idxs,
        'distances': distances,
        'pcas': pcas,
        'thetas': thetas,
        'phis': phis,
        'psis': psis,
        'nonp': nonp,
        'etas': etas,
        'v_in': v_in,
        'v_out': v_out,
        'v_in_theta': v_in_theta,
        'v_out_theta': v_out_theta,
        'v_in_phi': v_in_phi,
        'v_out_phi': v_out_phi,
        'v_in_psi': v_in_psi,
        'v_out_psi': v_out_psi,
    }


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

    stats = _calculate_trial_turn_statistics(args, smooth_K, window_size, height)
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

    stats = _calculate_trial_turn_statistics(args, smooth_K, window_size, height)
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
            stats = _calculate_trial_turn_statistics(args, smooth_K, window_size, height)
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

    # Calculate the model for all trials at all window sizes
    for trial in ds.include_trials:
        args.trial = trial.id
        for i, ws in enumerate(window_sizes):
            try:
                stats = _calculate_trial_turn_statistics(args, smooth_K, int(ws / dt), height)
            except RuntimeError as e:
                logger.warning(f'Failed to find approximation: "{e}"')
            distances[i].append(stats['distances'])
            nonp[i].append(stats['nonp'])

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


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # plot_trial_turns()
    # plot_trial_turn_correlations()
    plot_dataset_turn_correlations()
    plot_dataset_turn_correlations_across_windows()
