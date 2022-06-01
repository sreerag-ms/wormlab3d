import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from simple_worm.plot3d import Arrow3D
from wormlab3d import START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Trial
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca, tex_mode
from wormlab3d.toolkit.util import orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args

show_plots = False
save_plots = True
img_extension = 'svg'
tex_mode()


def single_trial_turn_durations(
        plot_all_turns: bool = False
):
    """
    Plot turn durations for a single trial.
    """
    args = get_args()
    assert args.trial is not None
    assert args.trajectory_point is None  # Use the full postures
    assert args.planarity_window is not None
    trial = Trial.objects.get(id=args.trial)
    dt = 1 / trial.fps
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)
    window_size = int(1 / dt)
    smooth_K = 101
    height = 50

    arrow_opts = {
        'mutation_scale': 10,
        'arrowstyle': '-|>',
        'linewidth': 2,
        'alpha': 0.7
    }

    if save_plots and plot_all_turns:
        turn_plots_dir = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_ws={window_size:.2f}_s={smooth_K}'
        os.makedirs(turn_plots_dir, exist_ok=True)

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Calculate the approximation, tumbles and runs
    k = calculate_curvature(e0, smooth_e0=smooth_K, smooth_K=smooth_K)
    approx = get_approximate(X, k, distance=window_size, height=height)
    mse = np.mean(np.sum((X - approx[0])**2, axis=-1))
    X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

    thetas = np.zeros(len(tumble_idxs))
    phis = np.zeros(len(tumble_idxs))
    psis = np.zeros(len(tumble_idxs))
    distances = np.zeros(len(tumble_idxs))

    # Recalculate the angles using the full trajectory and a fixed PCA window
    for i, tumble_idx in enumerate(tumble_idxs):

        # Calculate the PCA components for the window around the turn
        start_idx = max(0, tumble_idx - window_size)
        end_idx = min(X.shape[0] - 1, tumble_idx + window_size)
        X_window = X[start_idx:end_idx]
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)

        # Incoming and outgoing trajectories
        v_in = X[tumble_idx] - X[start_idx]
        v_out = X[end_idx] - X[tumble_idx]

        # Project onto the principal plane to find planar angle.
        v_in_theta = orthogonalise(v_in, pca.components_[2])
        v_out_theta = orthogonalise(v_out, pca.components_[2])
        thetas[i] = calculate_angle(v_in_theta, v_out_theta)

        # Orthogonalise against 2nd principal component to find non-planar angle.
        v_in_phi = orthogonalise(v_in, pca.components_[1])
        v_out_phi = orthogonalise(v_out, pca.components_[1])
        phis[i] = calculate_angle(v_in_phi, v_out_phi)

        # Orthogonalise against 1st principal component to find the final angle.
        v_in_psi = orthogonalise(v_in, pca.components_[0])
        v_out_psi = orthogonalise(v_out, pca.components_[0])
        psis[i] = calculate_angle(v_in_psi, v_out_psi)

        # Calculate distance travelled across windowed trajectory
        distances[i] = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()

        # Plot 3D turn
        if plot_all_turns:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection='3d')
            ax.set_title(
                f'Trial={trial.id}. '
                f'Frames={start_idx}-{end_idx}/{len(X)}. '
                f'$\\theta={thetas[i] / np.pi:.2f}\pi$. '
                f'$\phi={phis[i] / np.pi:.2f}\pi$. '
                f'$\psi={psis[i] / np.pi:.2f}\pi$. '
                f'$d={distances[i]:.2f}$mm.'
            )
            x, y, z = X_window.T
            ax.scatter(x, y, z, c=k[start_idx:end_idx], cmap='Reds', s=100, alpha=1, zorder=1)
            x, y, z = vertices[i + 1].T
            ax.scatter(x, y, z, color='blue', marker='x', s=700, alpha=0.7, zorder=1000, linewidth=3)
            plane = make_box_from_pca(X_window, pca, 'blue', scale=(0.1, 0.2, 0.3))
            ax.add_collection3d(plane)
            ax.add_artist(Arrow3D(origin=X[start_idx], vec=v_in, color='darkred', **arrow_opts))
            ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=v_out, color='red', **arrow_opts))
            ax.add_artist(Arrow3D(origin=X[start_idx], vec=v_in_theta, color='darkgreen', **arrow_opts))
            ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=v_out_theta, color='green', **arrow_opts))
            ax.add_artist(Arrow3D(origin=X[start_idx], vec=v_in_phi, color='darkblue', **arrow_opts))
            ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=v_out_phi, color='blue', **arrow_opts))
            ax.add_artist(Arrow3D(origin=X[start_idx], vec=v_in_psi, color='purple', **arrow_opts))
            ax.add_artist(Arrow3D(origin=vertices[i + 1], vec=v_out_psi, color='violet', **arrow_opts))
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

    # Plot correlations
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
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

    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_ws={window_size:.2f}_s={smooth_K}.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    single_trial_turn_durations(plot_all_turns=False)
