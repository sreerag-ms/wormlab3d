import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import PEParameters
from wormlab3d.particles.cache import get_sim_state_from_args
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate, find_approximation
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import normalise
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import smooth_trajectory

show_plots = True
save_plots = True
img_extension = 'svg'


def _plot_sim_approximation(
        parameters: PEParameters,
        sim_idx: int,
        distance: int,
        tumble_idxs: np.ndarray,
        X: np.ndarray,
        X_approx: np.ndarray,
        k: np.ndarray,
        vertices: np.ndarray,
        e0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        run_steps: np.ndarray,
        run_speeds: np.ndarray,
        planar_angles: np.ndarray,
        nonplanar_angles: np.ndarray,
        twist_angles: np.ndarray,
):
    """
    Plot a simulation output.
    """
    dt = parameters.dt
    ts = np.arange(len(X)) * dt
    tumble_ts = tumble_idxs * dt
    run_durations = run_steps * dt
    run_speeds = run_speeds / dt

    # Coefficient of variation
    cv = run_durations.std() / run_durations.mean()

    # Approximation error
    mse = np.mean(np.sum((X - X_approx)**2, axis=-1))

    # Set up plot
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 6)

    fig.suptitle(f'Sim {sim_idx}. Min. tumble distance={distance * dt:.2f}. CV={cv:.2f}. MSE={mse:.4f}.')

    # Trace of the runs
    ax = fig.add_subplot(gs[0, :])
    ax.set_title('Run trace')
    vertex_ts = np.r_[[0, ], tumble_ts, ts[-1]]
    duration_ts = (vertex_ts[1:] + vertex_ts[:-1]) / 2
    l1 = ax.plot(duration_ts[1:-1], run_durations, c='green', marker='+', alpha=0.5, label='Duration')
    ax.set_ylabel('Run duration (s)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(left=ts[0], right=ts[-1])
    ax2 = ax.twinx()
    ax2.set_ylabel('Run speed (mm/s)')
    l2 = ax2.plot(duration_ts[1:-1], run_speeds, c='red', marker='x', alpha=0.5, label='Speed')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc=1)

    # Trace of the tumbles
    ax = fig.add_subplot(gs[1, :])
    ax.set_title('Tumble trace')
    ax.axhline(y=0, color='darkgrey')
    ax.scatter(tumble_ts, planar_angles, label='$\\theta$', marker='x')
    ax.scatter(tumble_ts, nonplanar_angles, label='$\phi$', marker='o')
    ax.scatter(tumble_ts, twist_angles, label='$\psi$', marker='2')
    ax.set_xlim(left=ts[0], right=ts[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\\theta$')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    for t in tumble_ts:
        ax.axvline(x=t, color='pink', zorder=-1, alpha=0.4)
    ax.legend(loc=1)

    # Histograms of the parameters
    for i, (param_name, param) in enumerate(
            {
                'Run durations': run_durations,
                'Speeds': run_speeds,
                'Speeds (weighted)': run_speeds,
            }.items()):
        ax = fig.add_subplot(gs[2 + i, 0])
        ax.set_title(param_name)
        if param_name == 'Speeds (weighted)':
            weights = run_durations
        else:
            weights = np.ones_like(param)
        ax.hist(param, weights=weights, bins=21, density=True, facecolor='green', alpha=0.75)
    for i, (param_name, param) in enumerate(
            {
                'Planar angles': planar_angles,
                'Non-planar angles': nonplanar_angles,
                'Twist angles': twist_angles,
            }.items()):
        ax = fig.add_subplot(gs[2 + i, 1])
        ax.set_title(param_name)
        # ax.set_yscale('log')
        ax.hist(param, bins=21, density=True, facecolor='green', alpha=0.75)
        if param_name in ['Planar angles', 'Twist angles']:
            ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        if param_name == 'Non-planar angles':
            ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
            ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
            ax.set_xticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])

    # 3D trajectory of approximation
    T = len(vertices)
    ax = fig.add_subplot(gs[2:, 2:4], projection='3d')
    x, y, z = vertices[:T].T
    ax.scatter(x, y, z, color='blue', marker='x', s=50, alpha=0.6, zorder=1)

    # Add frame vectors
    F = FrameNumpy(x=vertices[:T].T, e0=e0[:T].T, e1=e1[:T].T, e2=e2[:T].T)
    fa = FrameArtist(F, arrow_scale=0.01, arrow_colours={
        'e0': 'blue',
        'e1': 'red',
        'e2': 'green',
    })
    fa.add_component_vectors(ax)

    # Add approximation trajectory
    points = vertices[:T][:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, color='blue', zorder=5, linewidth=2, linestyle=':', alpha=0.5)
    ax.add_collection(lc)
    equal_aspect_ratio(ax)

    # Actual 3D trajectory
    ax = fig.add_subplot(gs[2:, 4:6], projection='3d')
    x, y, z = X.T
    ax.scatter(x, y, z, c=k, cmap='Reds', s=10, alpha=0.4, zorder=-1)
    x, y, z = vertices[:T].T
    ax.scatter(x, y, z, color='blue', marker='x', s=50, alpha=0.9, zorder=10)
    equal_aspect_ratio(ax)

    fig.tight_layout()

    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_idx={sim_idx}_distance={distance * dt:.2f}.{img_extension}')

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_sim_approximations(
        parameters: PEParameters,
        sim_idx: int,
        X: np.ndarray,
        e0_raw: np.ndarray,
        k: np.ndarray,
        distances: np.ndarray = None,
        error_limits: np.ndarray = None,
):
    assert not (distances is None and error_limits is None)
    assert not (distances is not None and error_limits is not None)
    iter_var = distances if distances is not None else error_limits

    for loop_var in iter_var:
        # Calculate the approximation, tumbles and runs
        if distances is not None:
            X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 \
                = get_approximate(X, k, distance=loop_var)
        else:
            approx, distance, height, smooth_e0, smooth_K \
                = find_approximation(X, e0_raw, loop_var, max_iterations=50)
            X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

        # Plot approximation
        _plot_sim_approximation(
            parameters=parameters,
            sim_idx=sim_idx,
            distance=distance,
            tumble_idxs=tumble_idxs,
            X=X,
            X_approx=X_approx,
            k=k,
            vertices=vertices,
            e0=e0,
            e1=e1,
            e2=e2,
            run_steps=run_durations,
            run_speeds=run_speeds,
            planar_angles=planar_angles,
            nonplanar_angles=nonplanar_angles,
            twist_angles=twist_angles,
        )


def plot_simulation_approximations(
        plot_n_examples: int = 1,
        noise_scale: float = 0.1,
        smoothing_window: int = 25,
        args: Namespace = None
):
    """
    Convert a simulation trajectory into a sequence of straight-line runs and tumbles.
    """
    if args is None:
        args = get_args(validate_source=False)
    assert args.planarity_window is not None
    SS = get_sim_state_from_args(args)

    # Add some noise to the trajectories then smooth
    Xs = SS.X.copy().astype(np.float64)
    Xs = Xs - Xs.mean(axis=1, keepdims=True)
    if noise_scale is not None and noise_scale > 0:
        Xs = Xs + np.random.normal(np.zeros_like(Xs), noise_scale)
    if smoothing_window is not None and smoothing_window > 0:
        Xs = smooth_trajectory(Xs.transpose(1, 0, 2), window_len=smoothing_window).transpose(1, 0, 2)
    e0s = normalise(np.gradient(Xs, axis=1))

    # distances = [10, 20, 50, 100, 200]
    # distances = np.array([60, 120, 250, 315, 375, 500])
    # distances = np.array([25, 50, 100, 200])
    # distances = np.arange(5, 500, 5)#
    distances = None
    error_limits = np.array([0.1, 0.05, 0.01])

    bs = SS.parameters.batch_size
    idxs = np.random.choice(bs, min(bs, plot_n_examples), replace=False)
    for idx in idxs:
        k = calculate_curvature(e0s[idx])
        _plot_sim_approximations(SS.parameters, idx, Xs[idx], e0s[idx], k, distances, error_limits)


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_simulation_approximations(
        plot_n_examples=5,
        noise_scale=0.1,
        smoothing_window=25,
    )
