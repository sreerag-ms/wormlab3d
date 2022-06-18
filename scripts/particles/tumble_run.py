import gc
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from progress.bar import Bar
from scipy.stats import expon

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import START_TIMESTAMP, LOGS_PATH, logger
from wormlab3d.data.model import Trial, Dataset
from wormlab3d.particles.cache import get_trajectories_from_args
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate, find_approximation, \
    generate_or_load_ds_statistics, generate_or_load_ds_msds
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import get_deltas_from_args

show_plots = False
save_plots = True
img_extension = 'svg'


def _plot_trial_approximation(
        trial: Trial,
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
    dt = 1 / trial.fps
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

    fig.suptitle(f'Trial {trial.id}. Min. tumble distance={distance * dt:.2f}. CV={cv:.2f}. MSE={mse:.4f}.')

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
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_distance={distance * dt:.2f}.{img_extension}')

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_trial_approximations(
        trial: Trial,
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
                = find_approximation(X, e0_raw, loop_var, max_attempts=50)
            X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

        # Plot approximation
        _plot_trial_approximation(
            trial=trial,
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


def _plot_mse(
        trial: Trial,
        distances: np.ndarray,
        X: np.ndarray,
        k: np.ndarray,
):
    """
    Calculate the approximation errors at different distances.
    """
    mse = np.zeros(len(distances))
    for i, distance in enumerate(distances):
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 \
            = get_approximate(X, k, distance=distance)
        mse[i] = np.mean(np.sum((X - X_approx)**2, axis=-1))

    fig, axes = plt.subplots(1, figsize=(12, 10))
    ax = axes
    ax.plot(distances / trial.fps, mse, marker='x')
    ax.set_title(f'MSE. Trial={trial.id}.')
    ax.set_xlabel('Min. consecutive tumble distance (s)')
    ax.set_ylabel('Error')
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_mse_distances={distances[0]}-{distances[-1]}.{img_extension}')

    if show_plots:
        plt.show()


def convert_trajectory_to_tumble_run(
        args: Namespace = None
):
    """
    Convert a trial trajectory into a sequence of straight-line runs and tumbles.
    """
    if args is None:
        args = get_args()
    assert args.trajectory_point is None  # Use the full postures
    assert args.planarity_window is not None
    trial = Trial.objects.get(id=args.trial)
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)
    k = calculate_curvature(e0)

    # distances = [10, 20, 50, 100, 200]
    # distances = np.array([60, 120, 250, 315, 375, 500])
    # distances = np.array([25, 50, 100, 200])
    # distances = np.arange(5, 500, 5)#
    distances = None
    error_limits = np.array([0.1, 0.05, 0.01])

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    _plot_trial_approximations(trial, X, e0, k, distances, error_limits)
    # _plot_mse(trial, distances, X, k)


def plot_dataset_trajectories():
    args = get_args()

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Calculate the model for all trials
    for trial in ds.include_trials:
        logger.info(f'Computing tumble-run model for trial={trial.id}.')
        args.trial = trial.id
        convert_trajectory_to_tumble_run(args)
        gc.collect()


def plot_coefficients_of_variation():
    args = get_args()
    assert args.planarity_window is not None

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # error_limits = [0.001, 0.002, 0.004, 0.008, 0.016]
    error_limits = np.array([0.016, 0.008, 0.004, 0.002, 0.001])

    # Coefficients of variation
    cov = np.zeros((4, len(ds.include_trials), len(error_limits)))

    # Calculate the model for all trials
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Computing tumble-run model for trial={trial.id}.')
        args.trial = trial.id
        X = get_trajectory_from_args(args)
        pcas = get_pca_cache_from_args(args)
        e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

        # Take centre of mass
        if X.ndim == 3:
            X = X.mean(axis=1)
        X -= X.mean(axis=0)

        # Calculate coefficients of variation for all params for all trials at all distances
        distance = 500
        height = 50
        smooth_e0 = 101
        smooth_K = 101

        for j, error_limit in enumerate(error_limits):
            approx, distance, height, smooth_e0, smooth_K \
                = find_approximation(X, e0, error_limit, distance, height, smooth_e0, smooth_K, max_attempts=10)
            X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, _, _, _ = approx

            # cov[:, i, j] = [
            #     run_durations.std() / run_durations.mean(),
            #     run_speeds.std() / run_speeds.mean(),
            #     np.abs(planar_angles).std() / np.abs(planar_angles).mean(),
            #     np.abs(nonplanar_angles).std() / np.abs(nonplanar_angles).mean(),
            # ]

            run_durations_diff = np.abs(run_durations[1:] - run_durations[:-1])
            run_speeds_diff = np.abs(run_speeds[1:] - run_speeds[:-1])
            planar_angles_diff = np.abs(planar_angles[1:] - planar_angles[:-1])
            nonplanar_angles_diff = np.abs(nonplanar_angles[1:] - nonplanar_angles[:-1])
            twist_angles_diff = np.abs(twist_angles[1:] - twist_angles[:-1])

            dt = 1 / trial.fps
            run_durations_ts = run_durations * dt

            cov[:3, i, j] = [
                run_durations.std() / run_durations.mean(),
                run_durations_diff.std() / run_durations_diff.mean(),
                run_durations_ts.std() / run_durations_ts.mean(),
                # run_speeds_diff.std() / run_speeds_diff.mean(),
                # planar_angles_diff.std() / planar_angles_diff.mean(),
                # nonplanar_angles_diff.std() / nonplanar_angles_diff.mean(),
            ]

        # if i > 5:
        #     break

    # Plot
    fig, axes = plt.subplots(3, figsize=(12, 12))
    fig.suptitle('Coefficients of Variation')

    # for i, param_name in enumerate(['Durations', 'Speeds', 'Planar Angles', 'Non-planar Angles']):
    for i, param_name in enumerate(
            ['Durations in frames', 'Consecutive durations differences (frames)', 'Durations in seconds']):
        ax = axes[i]
        ax.set_title(param_name)
        ax.axhline(y=1, linestyle='--', color='black', linewidth=2)

        cv = cov[i]

        for j in range(len(cv)):
            ax.plot(error_limits[::-1], cv[j][::-1], alpha=0.7)

        ax.set_xticks(error_limits)
        ax.set_xticklabels(error_limits)
        ax.set_xlabel('Error limit')

    fig.tight_layout()
    plt.show()


def single_trial_approximation():
    """
    Plot a single trial
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
    k = calculate_curvature(e0)
    distance = int(8 / dt)

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Calculate the approximation, tumbles and runs
    X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 \
        = get_approximate(X, k, distance=distance)

    if 0:
        np.savez(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={args.trial}_distance={distance * dt:.2f}',
            X_real=X,
            k=k,
            X_approx=X_approx,
            vertices=vertices,
            e0_approx=e0,
            e1_approx=e1,
            e2_approx=e2,
        )

    # Plot the real trajectory, coloured by the curvature
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    x, y, z = X.T
    s = ax.scatter(x, y, z, c=k, cmap='Reds', s=100, alpha=1, zorder=1)
    cb = fig.colorbar(s, shrink=0.5, ticks=[50, 100, 150])
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_ylabel('Curvature (mm$^{-1}$)', rotation=270, fontsize=16, labelpad=20)
    x, y, z = vertices.T
    ax.scatter(x, y, z, color='blue', marker='x', s=700, alpha=0.9, zorder=1000, linewidth=3)
    equal_aspect_ratio(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_distance={distance * dt:.2f}_real.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()

    # Plot the approximated trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    x, y, z = vertices.T
    ax.scatter(x, y, z, color='blue', marker='x', s=700, alpha=0.9, zorder=1, linewidth=5)

    # Add frame vectors
    F = FrameNumpy(x=vertices.T, e0=e0.T, e1=e1.T, e2=e2.T)
    fa = FrameArtist(F, arrow_scale=0.022, arrow_colours={
        'e0': 'blue',
        'e1': 'red',
        'e2': 'green',
    }, arrow_opts={
        'linewidth': 4,
        'mutation_scale': 25
    })
    fa.add_component_vectors(ax)

    # Add approximation trajectory
    points = vertices[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, color='blue', zorder=5, linewidth=5, linestyle=':', alpha=0.5)
    ax.add_collection(lc)
    equal_aspect_ratio(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.axis('off')
    fig.tight_layout()

    # Save / show
    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_distance={distance * dt:.2f}_approx.{img_extension}',
            transparent=True
        )
    if show_plots:
        plt.show()


def dataset_distributions():
    """
    Plot the distributions of values across a dataset at different error limits.
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # error_limits = [0.001, 0.002, 0.004, 0.008, 0.016]
    # error_limits = np.array([0.016, 0.008, 0.004, 0.002, 0.001])
    # error_limits = np.array([0.01,0.001,0.0001])
    error_limits = [0.5, 0.2, 0.1, 0.05, 0.01]
    planarity_window = 5
    trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles \
        = generate_or_load_ds_statistics(ds, error_limits, planarity_window)

    # Plot
    fig, axes = plt.subplots(len(error_limits), 6, figsize=(14, 2 + 2 * len(error_limits)), squeeze=False)
    fig.suptitle(f'Dataset={ds.id}. Planarity window={planarity_window}.')

    for i, (param_name, values) in enumerate({
                                                 'Durations': durations,
                                                 'Speeds': speeds,
                                                 'Speeds (weighted)': speeds,
                                                 'Planar angles': planar_angles,
                                                 'Non-planar angles': nonplanar_angles,
                                                 'Twist angles': twist_angles,
                                             }.items()):
        for j, error_limit in enumerate(error_limits):
            ax = axes[j, i]
            ax.set_title(param_name)

            if i == 0:
                ax.set_ylabel(f'Error ~ {error_limit:.4f}')

            values_j = np.array(values[j])

            # if param_name not in ['Speeds', 'Speeds (weighted)']:
            if param_name not in ['Planar angles', 'Non-planar angles', 'Twist angles']:
                ax.set_yscale('log')
            if param_name == 'Speeds (weighted)':
                weights = np.array(durations[j])
            else:
                weights = np.ones_like(values_j)

            ax.hist(values_j, weights=weights, bins=21, density=True, facecolor='green', alpha=0.75)
            ax.set_title(param_name)

            if param_name in ['Planar angles', 'Twist angles']:
                ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
            if param_name == 'Non-planar angles':
                ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
                ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
                ax.set_xticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            LOGS_PATH / f'{START_TIMESTAMP}_histograms_ds={ds.id}_err={error_limit:.2f}.{img_extension}',
            transparent=True
        )

    if show_plots:
        plt.show()


def plot_dataset_angle_comparisons():
    """
    Plot the P vs NP angles as scatter plots
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    error_limits = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles \
        = generate_or_load_ds_statistics(ds, error_limits, planarity_window=5, rebuild_cache=False)

    fig, axes = plt.subplots(len(error_limits), 3, figsize=(4, 3 * len(error_limits)))

    for i, lim in enumerate(error_limits):
        ax = axes[i, 0]
        ax.set_title(f'Error limit={lim:.2f}.')
        ax.scatter(x=planar_angles[i], y=nonplanar_angles[i], s=1, alpha=0.5)
        ax.set_xlabel('$\\theta$')
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax.set_ylabel('$\\phi$')
        ax.set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax.set_yticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])

        ax = axes[i, 1]
        ax.set_title(f'Error limit={lim:.2f}.')
        ax.scatter(x=planar_angles[i], y=twist_angles[i], s=1, alpha=0.5)
        ax.set_xlabel('$\\theta$')
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
        ax.set_ylabel('$\\psi$')
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])

        ax = axes[i, 2]
        ax.set_title(f'Error limit={lim:.2f}.')
        ax.scatter(x=nonplanar_angles[i], y=twist_angles[i], s=1, alpha=0.5)
        ax.set_xlabel('$\\phi$')
        ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
        ax.set_xticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])
        ax.set_ylabel('$\\psi$')
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])

    fig.tight_layout()
    plt.show()


def dataset_against_three_state_comparison():
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    deltas, delta_ts = get_deltas_from_args(args)
    planarity_window = 7

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # error_limits = [0.001, 0.002, 0.004, 0.008, 0.016]
    # error_limits = np.array([0.016, 0.008, 0.004, 0.002, 0.001])
    # error_limits = np.array([0.01,0.001,0.0001])
    error_limits = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    # error_limits = [0.05,]

    # Generate or load tumble/run values
    trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles \
        = generate_or_load_ds_statistics(ds, error_limits, planarity_window=planarity_window, rebuild_cache=False)

    # Generate or load MSDs
    msds_all_real, msds_real = generate_or_load_ds_msds(ds, args, rebuild_cache=False)

    # Now make a simulator to match the results
    args.sim_dt = 1 / 25
    args.sim_duration = max(trajectory_lengths) * args.sim_dt
    pe, TC = get_trajectories_from_args(args)

    # Plot histograms
    if 1:
        fig, axes = plt.subplots(len(error_limits), 6, figsize=(14, 2 + 2 * len(error_limits)), squeeze=False)
        fig.suptitle(f'Dataset={ds.id}. Planarity windows={args.planarity_window} frames, {planarity_window} vertices.')

        for i, (param_name, values) in enumerate({
                                                     'Durations': [durations, TC.intervals],
                                                     'Speeds': [speeds, TC.speeds],
                                                     'Speeds (weighted)': [speeds, TC.speeds],
                                                     'Planar angles': [planar_angles, TC.thetas],
                                                     'Non-planar angles': [nonplanar_angles, TC.phis],
                                                     'Twist angles': [twist_angles, [twist_angles[0], ]],
                                                 }.items()):
            for j, error_limit in enumerate(error_limits):
                ax = axes[j, i]
                ax.set_title(param_name)

                if i == 0:
                    ax.set_ylabel(f'Error ~ {error_limit:.4f}')

                values_ds = np.array(values[0][j])
                values_sim = np.concatenate(values[1])

                if param_name not in ['Planar angles', 'Non-planar angles', 'Twist angles']:
                    ax.set_yscale('log')
                if param_name == 'Speeds (weighted)':
                    weights = [
                        np.array(durations[j]),
                        np.concatenate(TC.intervals),
                    ]
                else:
                    weights = [
                        np.ones_like(values_ds),
                        np.ones_like(values_sim),
                    ]

                ax.hist([values_ds, values_sim], weights=weights, bins=21, density=True, alpha=0.75)
                ax.set_title(param_name)

                if param_name in ['Planar angles', 'Twist angles']:
                    ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
                    ax.set_xticks([-np.pi, 0, np.pi])
                    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
                if param_name == 'Non-planar angles':
                    ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
                    ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
                    ax.set_xticklabels(['$-\\frac{\pi}{2}$', '0', '$\\frac{\pi}{2}$'])

        fig.tight_layout()

        if save_plots:
            plt.savefig(
                LOGS_PATH / f'{START_TIMESTAMP}_histograms_comparison_ds={ds.id}.{img_extension}',
                transparent=True
            )

        if show_plots:
            plt.show()

    exit()

    rs = len(ds.include_trials)

    # Model the distribution of trajectory lengths with a skewnorm
    # T_dist = skewnorm(*skewnorm.fit(trajectory_lengths))
    T_dist = expon(*expon.fit(trajectory_lengths))

    # Sample some trajectory lengths
    T_sim_lengths = T_dist.rvs(args.batch_size).round().astype(np.uint32)

    msds_all_sim = np.zeros(len(deltas))
    msds_sim = {}
    d_all_sim = {delta: [] for delta in deltas}

    logger.info(f'Calculating MSDs for simulation runs.')
    bar = Bar('Calculating', max=args.batch_size)
    bar.check_tty = False
    for i, X in enumerate(TC.X):
        # Pick a random length
        X = X[:T_sim_lengths[i]]

        msds_i = []
        for delta in deltas:
            if delta > X.shape[0] / 3:
                continue
            d = np.sum((X[delta:] - X[:-delta])**2, axis=-1)
            d_all_sim[delta].append(d)
            msds_i.append(d.mean())
        msds_sim[i] = np.array(msds_i)
        bar.next()
    bar.finish()

    for i, delta in enumerate(deltas):
        msds_all_sim[i] = np.concatenate(d_all_sim[delta]).mean()

    # Plot MSDs
    if 1:
        # Set up plots and colours
        logger.info('Plotting MSD.')
        fig, ax = plt.subplots(1, figsize=(12, 10))
        cmap = plt.get_cmap('winter')
        colours_real = cmap(np.linspace(0, 1, rs))
        cmap = plt.get_cmap('autumn')
        colours_sim = cmap(np.linspace(0, 1, args.batch_size))

        # Plot average of the real MSDs
        ax.plot(delta_ts, msds_all_real, label='Trajectory average',
                alpha=0.8, c='black', linestyle='--', linewidth=3, zorder=60)

        # Plot MSD for each real trajectory
        for i, (trial_id, msd_vals_real) in enumerate(msds_real.items()):
            ax.plot(delta_ts[:len(msd_vals_real)], msd_vals_real, alpha=0.4, c=colours_real[i])

        # Plot MSD for each simulation
        for i, (idx, msd_vals_sim) in enumerate(msds_sim.items()):
            # msd_vals = np.array(list(msd_vals_sim.values()))
            ax.plot(delta_ts[:len(msd_vals_sim)], msd_vals_sim, alpha=0.4, c=colours_sim[i])

        # Plot average of the simulation MSDs
        # msd_vals_all_sim = np.array(list(msds_all_sim.values()))
        ax.plot(delta_ts, msds_all_sim, label='Simulation average',
                alpha=0.9, c='black', linestyle=':', linewidth=3, zorder=80)

        # Complete MSD plot
        ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta\ (s)$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

        fig.tight_layout()
        plt.show()

        # Set up plots and colours
        logger.info('Plotting MSD ranges.')
        fig, ax = plt.subplots(1, figsize=(12, 10))

        # Calculate stds
        msds_real_std = np.zeros(len(delta_ts))
        msds_sim_std = np.zeros(len(delta_ts))
        for i, delta in enumerate(delta_ts):
            vals_d = []
            for trial_id, msd_vals_real in msds_real.items():
                if len(msd_vals_real) > i:
                    vals_d.append(msd_vals_real[i])
            msds_real_std[i] = np.std(vals_d)

            vals_d = []
            for sim_idx, msd_vals_sim in msds_sim.items():
                if len(msd_vals_sim) > i:
                    vals_d.append(msd_vals_sim[i])
            msds_sim_std[i] = np.std(vals_d)

        # Plot average of the real MSDs
        ax.plot(delta_ts, msds_all_real, label='Real',
                alpha=0.8, c='blue', linestyle='--', linewidth=3, zorder=60)
        ax.fill_between(delta_ts, msds_all_real - msds_real_std, msds_all_real + msds_real_std, color='blue', alpha=0.2)

        # Plot average of the simulation MSDs
        ax.plot(delta_ts, msds_all_sim, label='Simulation',
                alpha=0.9, c='orange', linestyle=':', linewidth=3, zorder=80)
        ax.fill_between(delta_ts, msds_all_sim - msds_sim_std, msds_all_sim + msds_sim_std, color='orange', alpha=0.2)

        # Complete MSD plot
        ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta\ (s)$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

        fig.tight_layout()
        plt.show()

    # exit()

    # Paper quality plots
    if 1:
        error_limit = 0.05
        assert error_limit in error_limits
        j = np.argmin(np.abs(error_limits - error_limit))

        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colours = prop_cycle.by_key()['color']
        colour_real = default_colours[0]
        colour_sim = default_colours[1]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        for i, (param_name, values) in enumerate({
                                                     'Run durations': [durations, TC.intervals],
                                                     'Run speeds': [speeds, TC.speeds],
                                                     'Planar angles': [planar_angles, TC.thetas],
                                                     'Non-planar angles': [nonplanar_angles, TC.phis]
                                                 }.items()):
            ax = axes[i]
            ax.set_title(param_name)

            if i == 0:
                ax.set_ylabel(f'Error ~ {error_limit:.4f}')

            values_ds = np.array(values[0][j])
            values_sim = np.concatenate(values[1])

            if param_name not in ['Planar angles', 'Non-planar angles']:
                ax.set_yscale('log')
            if param_name == 'Speeds':
                weights = [
                    np.array(durations[j]),
                    np.concatenate(TC.intervals),
                ]
            else:
                weights = [
                    np.ones_like(values_ds),
                    np.ones_like(values_sim),
                ]

            ax.hist(
                [values_ds, values_sim],
                weights=weights,
                color=[colour_real, colour_sim],
                bins=11,
                density=True,
                alpha=0.75,
                label=['Real', 'Simulation']
            )
            ax.set_title(param_name)
            ax.set_ylabel('Density')
            ax.legend()

            if param_name == 'Run durations':
                ax.set_xticks([0, 50, 100])
                ax.set_xticklabels(['0', '50', '100'])
                ax.set_xlabel('s')
            if param_name == 'Run speeds':
                ax.set_xticks([0, 0.1, 0.2])
                ax.set_xticklabels(['0', '0.1', '0.2'])
                ax.set_xlabel('mm/s')
            if param_name == 'Planar angles':
                ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
                ax.set_xticks([-np.pi, np.pi])
                ax.set_xticklabels(['$-\pi$', '$\pi$'])
                ax.set_xlabel('$\\theta$')
            if param_name == 'Non-planar angles':
                ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
                ax.set_xticks([-np.pi / 2, np.pi / 2])
                ax.set_xticklabels(['$-\\frac{\pi}{2}$', '$\\frac{\pi}{2}$'])
                ax.set_xlabel('$\\phi$')

        fig.tight_layout()

        if save_plots:
            plt.savefig(
                LOGS_PATH / f'{START_TIMESTAMP}_histogram_comparison_ds={ds.id}_err={error_limit:.2f}.{img_extension}',
                transparent=True
            )

        if show_plots:
            plt.show()

        # exit()

        # Set up MSD plot
        logger.info('Plotting MSD ranges.')
        fig, ax = plt.subplots(1, figsize=(5, 4))

        # Calculate stds
        msds_real_std = np.zeros(len(delta_ts))
        msds_sim_std = np.zeros(len(delta_ts))
        for i, delta in enumerate(delta_ts):
            vals_d = []
            for trial_id, msd_vals_real in msds_real.items():
                if len(msd_vals_real) > i:
                    vals_d.append(msd_vals_real[i])
            msds_real_std[i] = np.std(vals_d)

            vals_d = []
            for sim_idx, msd_vals_sim in msds_sim.items():
                if len(msd_vals_sim) > i:
                    vals_d.append(msd_vals_sim[i])
            msds_sim_std[i] = np.std(vals_d)

        # Plot average of the real MSDs
        ax.plot(delta_ts, msds_all_real, label='Real',
                alpha=0.8, c=colour_real, linestyle='--', linewidth=3, zorder=60)
        ax.fill_between(delta_ts, msds_all_real - msds_real_std, msds_all_real + msds_real_std, color=colour_real,
                        alpha=0.2)

        # Plot average of the simulation MSDs
        ax.plot(delta_ts, msds_all_sim, label='Simulation',
                alpha=0.9, c=colour_sim, linestyle=':', linewidth=3, zorder=80)
        ax.fill_between(delta_ts, msds_all_sim - msds_sim_std, msds_all_sim + msds_sim_std, color=colour_sim, alpha=0.2)

        # Complete MSD plot
        ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta\ (s)$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

        fig.tight_layout()

        if save_plots:
            plt.savefig(
                LOGS_PATH / f'{START_TIMESTAMP}_msd_comparison_ds={ds.id}_err={error_limit:.2f}.{img_extension}',
                transparent=True
            )

        if show_plots:
            plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # convert_trajectory_to_tumble_run()

    # plot_dataset_trajectories()
    # plot_coefficients_of_variation()

    # single_trial_approximation()

    # dataset_distributions()
    # plot_dataset_angle_comparisons()
    dataset_against_three_state_comparison()
