import os
from argparse import Namespace
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit, minimize

from simple_worm.plot3d import MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.toolkit.util import to_dict, hash_data
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.statistics import calculate_trial_turn_statistics, calculate_trial_run_statistics, \
    calculate_windowed_statistics, calculate_windowed_helicity
from wormlab3d.trajectories.util import get_deltas_from_args

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'svg'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['dataset', 'trial', 'frames', 'src', 'deltas', 'delta_step', 'window_size', 'u', 'smoothing_window',
              'directionality', 'error_limit', 'smooth_K', 'pw_vertices', 'approx_distance', 'approx_height']:
        if k in excludes:
            continue
        if k == 'dataset' and args.dataset is not None:
            fn += f'_dataset={args.dataset}'
        elif k == 'trial' and args.trial is not None:
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            fn += f'_{args.midline3d_source}'
        elif k == 'window_size' and hasattr(args, 'window_size'):
            fn += f'_ws={args.window_size}'
        elif k == 'deltas':
            fn += f'_d={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            fn += f'_ds={args.delta_step}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'error_limit' and args.approx_error_limit is not None:
            fn += f'_err={args.approx_error_limit}'
        elif k == 'smooth_K':
            fn += f'_smooth_K={args.smoothing_window_K}'
        elif k == 'pw_vertices':
            fn += f'_pwv={args.planarity_window_vertices}'
        elif k == 'approx_distance':
            fn += f'_dist={args.approx_distance}'
        elif k == 'approx_height':
            fn += f'_height={args.approx_curvature_height}'

    return LOGS_PATH / (fn + '.' + img_extension)


def nonplanarity_trajectory():
    """
    Plot the non-planarity changes along a single trajectory as the time window changes.
    """
    args = get_args()
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    delta_ts = deltas / 25

    # Calculate planarities across different time windows
    nonp = []
    for delta in deltas:
        args.planarity_window = int(delta)
        logger.info(f'Fetching PCA data for delta = {int(delta)}.')
        pca_cache = get_pca_cache_from_args(args)
        nonp.append(pca_cache.nonp)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot()

    parts = plt.violinplot(nonp, delta_ts, widths=args.delta_step / 25, showmeans=True, showmedians=True)
    parts['cmedians'].set_color('green')
    parts['cmedians'].set_alpha(0.7)
    parts['cmedians'].set_linestyle(':')
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_alpha(0.7)
    parts['cmeans'].set_linestyle('--')

    ax.set_xlabel('$\Delta s$')
    ax.set_ylabel('Non-planarity')
    ax.set_title(f'Non-planarity vs Delta (time window). Trial {args.trial}.')
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('planarity_vs_delta', args))
    if show_plots:
        plt.show()


def nonplanarity_dataset():
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate the non-planarity for all trials
    nonp = {}
    all_nonp = {delta: [] for delta in deltas}
    n_trials = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating non-planarity for trial={trial.id}.')
        args.trial = trial.id

        # Group results by concentration
        c = trial.experiment.concentration
        if c not in nonp:
            nonp[c] = {delta: [] for delta in deltas}
        if c not in n_trials:
            n_trials[c] = 0
        n_trials[c] += 1

        # Calculate non-planarity across different time windows
        for delta in deltas:
            args.planarity_window = int(delta)
            logger.info(f'Fetching PCA data for window size = {int(delta)}.')
            try:
                pca_cache = get_pca_cache_from_args(args)
                nonp[c][delta].extend(pca_cache.nonp)
                all_nonp[delta].extend(pca_cache.nonp)
            except AssertionError as e:
                # Window size is greater than trajectory length, so break here
                logger.warning(e)
                break

    # Sort by concentration
    nonp = {k: v for k, v in sorted(list(nonp.items()))}

    # Set up plots
    n_rows = 1 + len(nonp)
    fig, axes = plt.subplots(n_rows, figsize=(14, n_rows * 3), sharex=False, sharey=True)

    def _violinplot(ax_, vals_):
        pos = []
        vvals = []
        for ii, v in enumerate(vals_):
            if len(v) > 0:
                pos.append(delta_ts[ii])
                vvals.append(v)
        if len(vvals) == 0:
            logger.warning('No data available to make violin plot!')
            return
        if args.delta_step < 0:
            parts = ax_.violinplot(vvals, widths=1, showmeans=True, showmedians=True)
            ax_.set_xticks(np.arange(1, len(pos) + 1))
            ax_.set_xticklabels(pos)
        else:
            parts = ax_.violinplot(vvals, pos, widths=args.delta_step / 25, showmeans=True, showmedians=True)
        parts['cmedians'].set_color('green')
        parts['cmedians'].set_alpha(0.7)
        parts['cmedians'].set_linestyle(':')
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_alpha(0.7)
        parts['cmeans'].set_linestyle('--')
        ax_.set_ylabel('Non-planarity')
        ax_.set_xlabel('$\Delta s$')

    # Aggregate results at each concentration
    n_trials_total = 0
    for i, (c, nonp_c) in enumerate(nonp.items()):
        ax = axes[i + 1]
        n_trials_total += n_trials[c]
        ax.set_title(f'Concentration = {c:.2f}% ({n_trials[c]} trials)')
        _violinplot(ax, nonp_c.values())

    # Top plot shows aggregation from all results
    ax = axes[0]
    ax.set_title(f'All concentrations ({n_trials_total} trials)')
    _violinplot(ax, all_nonp.values())

    fig.tight_layout()

    if save_plots:
        args.trial = None
        args.reconstruction = None
        plt.savefig(
            make_filename('nonplanarity_dataset', args, excludes=['trial', 'deltas'])
        )
    if show_plots:
        plt.show()


def nonplanarity_postures_vs_trajectories():
    """
    Plot the dataset turns vs runs.
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.tracking_only = False
    args.trajectory_point = None
    dt = 1 / 25
    window_size = int(2 / dt)

    # Outputs
    speeds_turns = []
    distances_turns = []
    nonp_trajectories_turns = []
    nonp_postures_turns = []
    nonp_postures_max_turns = []
    speeds_runs = []
    distances_runs = []
    nonp_trajectories_runs = []
    nonp_postures_runs = []
    nonp_postures_max_runs = []

    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        args.trial = reconstruction.trial.id

        # Calculate turn statistics
        try:
            turn_stats = calculate_trial_turn_statistics(args, window_size)
            speeds_turns.append(turn_stats['speeds'])
            distances_turns.append(turn_stats['distances'])
            nonp_trajectories_turns.append(turn_stats['nonp'])
            nonp_postures_turns.append(turn_stats['nonp_postures'])
            nonp_postures_max_turns.append(turn_stats['nonp_postures_max'])

        except RuntimeError as e:
            logger.warning(f'Failed to find approximation: "{e}"')

        # Calculate runs statistics
        try:
            run_stats = calculate_trial_run_statistics(args)
            speeds_runs.append(run_stats['speeds'])
            distances_runs.append(run_stats['distances'])
            nonp_trajectories_runs.append(run_stats['nonp'])
            nonp_postures_runs.append(run_stats['nonp_postures'])
            nonp_postures_max_runs.append(run_stats['nonp_postures_max'])
            # if (run_stats['nonp'] > 0.1).any():
            #     for i in range(len(run_stats['nonp'])):
            #         if run_stats['nonp'][i] > 0.1:
            #             X = get_trajectory_from_args(args)
            #             X_window = X[run_stats['start_idxs'][i]:run_stats['end_idxs'][i]]
            #             Xs = X_window.transpose(0, 2, 1)
            #             FS = FrameSequenceNumpy(x=Xs)
            #             generate_interactive_scatter_clip(FS, fps=25)
        except RuntimeError as e:
            logger.warning(f'Failed to calculate run stats: "{e}"')

    n_turn_trajectories = len(speeds_turns)
    n_run_trajectories = len(speeds_runs)
    logger.info(f'Calculated turn statistics for {n_turn_trajectories} '
                f'and run statistics for {n_run_trajectories} '
                f'out of a possible {len(ds.reconstructions)}.')

    # Join outputs
    speeds_turns = np.concatenate(speeds_turns)
    distances_turns = np.concatenate(distances_turns)
    nonp_trajectories_turns = np.concatenate(nonp_trajectories_turns)
    nonp_postures_turns = np.concatenate(nonp_postures_turns)
    nonp_postures_max_turns = np.concatenate(nonp_postures_max_turns)
    speeds_runs = np.concatenate(speeds_runs)
    distances_runs = np.concatenate(distances_runs)
    nonp_trajectories_runs = np.concatenate(nonp_trajectories_runs)
    nonp_postures_runs = np.concatenate(nonp_postures_runs)
    nonp_postures_max_runs = np.concatenate(nonp_postures_max_runs)

    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4)

    # Scatter plot of NP-T vs NP-P during runs
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f'Runs (min={args.approx_min_run_duration * dt:.2f}s).')
    ax.set_xlabel('NP-trajectories')
    ax.set_ylabel('NP-postures')
    s = ax.scatter(nonp_trajectories_runs, nonp_postures_runs, c=speeds_runs, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    # Scatter plot of NP-T vs NP-P during turns
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'Turns (window={window_size * dt:.2f}s).')
    ax.set_xlabel('NP-trajectories')
    ax.set_ylabel('NP-postures')
    s = ax.scatter(nonp_trajectories_turns, nonp_postures_turns, c=speeds_turns, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    # Scatter plot of speed vs NP-P during runs
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f'Runs (min={args.approx_min_run_duration * dt:.2f}s).')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('NP-postures')
    s = ax.scatter(speeds_runs, nonp_postures_runs, c=distances_runs, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Distance (mm)', rotation=270)

    # Scatter plot of speed vs NP-P during turns
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title(f'Turns (window={window_size * dt:.2f}s).')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('NP-postures')
    s = ax.scatter(speeds_turns, nonp_postures_turns, c=distances_turns, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Distance (mm)', rotation=270)

    # Scatter plot of NP-T vs MAX(NP-P) during runs
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title(f'Runs (min={args.approx_min_run_duration * dt:.2f}s).')
    ax.set_xlabel('NP-trajectories')
    ax.set_ylabel('MAX(NP-postures)')
    s = ax.scatter(nonp_trajectories_runs, nonp_postures_max_runs, c=speeds_runs, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    # Scatter plot of NP-T vs MAX(NP-P) during turns
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title(f'Turns (window={window_size * dt:.2f}s).')
    ax.set_xlabel('NP-trajectories')
    ax.set_ylabel('MAX(NP-postures)')
    s = ax.scatter(nonp_trajectories_turns, nonp_postures_max_turns, c=speeds_turns, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    # Scatter plot of speed vs MAX(NP-P) during runs
    ax = fig.add_subplot(gs[0, 3])
    ax.set_title(f'Runs (min={args.approx_min_run_duration * dt:.2f}s).')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('MAX(NP-postures)')
    s = ax.scatter(speeds_runs, nonp_postures_max_runs, c=distances_runs, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Distance (mm)', rotation=270)

    # Scatter plot of speed vs MAX(NP-P) during turns
    ax = fig.add_subplot(gs[1, 3])
    ax.set_title(f'Turns (window={window_size * dt:.2f}s).')
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('MAX(NP-postures)')
    s = ax.scatter(speeds_turns, nonp_postures_max_turns, c=distances_turns, s=2)
    cb = fig.colorbar(s)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('Distance (mm)', rotation=270)

    fig.tight_layout()

    if save_plots:
        args.trial = None
        args.reconstruction = None
        plt.savefig(
            make_filename('nonplanarity_postures_vs_trajectories', args, excludes=['trial', 'deltas'])
        )
    if show_plots:
        plt.show()


def nonplanarity_postures_vs_trajectory_windows():
    """
    Plot the non-planarity of postures vs trajectory windows.
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.tracking_only = False

    dt = 1 / 25
    # window_sizes = (np.array([1,2,4,8])/dt).astype(np.int32)
    window_sizes = (np.array([8, 16, 32, 64]) / dt).astype(np.int32)

    # Outputs
    speeds = {ws: [] for ws in window_sizes}
    distances = {ws: [] for ws in window_sizes}
    nonp_trajectories = {ws: [] for ws in window_sizes}
    nonp_postures_mean = {ws: [] for ws in window_sizes}
    nonp_postures_max = {ws: [] for ws in window_sizes}

    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        args.trial = reconstruction.trial.id

        # Calculate windowed statistics
        stats = calculate_windowed_statistics(args, window_sizes)
        for ws in window_sizes:
            speeds[ws].append(stats['speeds'][ws])
            # distances[ws].append(stats['distances'][ws])
            nonp_trajectories[ws].append(stats['nonp_trajectories'][ws])
            nonp_postures_mean[ws].append(stats['nonp_postures_mean'][ws])
            nonp_postures_max[ws].append(stats['nonp_postures_max'][ws])

    # Join outputs
    for ws in window_sizes:
        speeds[ws] = np.concatenate(speeds[ws])
        # distances[ws] = np.concatenate(distances[ws])
        nonp_trajectories[ws] = np.concatenate(nonp_trajectories[ws])
        nonp_postures_mean[ws] = np.concatenate(nonp_postures_mean[ws])
        nonp_postures_max[ws] = np.concatenate(nonp_postures_max[ws])

    # Plot
    fig = plt.figure(figsize=(len(window_sizes) * 4, 10))
    gs = GridSpec(2, len(window_sizes))

    for i, ws in enumerate(window_sizes):
        # Scatter plot of NP-T vs MEAN NP-P
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f'Window size = {ws}')
        ax.set_xlabel('NP-trajectories')
        ax.set_ylabel('MEAN(NP-postures)')
        s = ax.scatter(nonp_trajectories[ws], nonp_postures_mean[ws], c=speeds[ws], s=2, cmap='PRGn',
                       norm=MidpointNormalize(midpoint=0))
        cb = fig.colorbar(s)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

        # Scatter plot of NP-T vs MAX NP-P
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f'Window size = {ws}')
        ax.set_xlabel('NP-trajectories')
        ax.set_ylabel('MAX(NP-postures)')
        s = ax.scatter(nonp_trajectories[ws], nonp_postures_max[ws], c=speeds[ws], s=2, cmap='PRGn',
                       norm=MidpointNormalize(midpoint=0))
        cb = fig.colorbar(s)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    fig.tight_layout()

    if save_plots:
        args.trial = None
        args.reconstruction = None
        plt.savefig(
            make_filename('nonplanarity_postures_vs_trajectory_windows', args, excludes=['trial', 'deltas'])
        )
    if show_plots:
        plt.show()


def helicity_postures_vs_trajectory_windows():
    """
    Plot the helicity of postures vs trajectory windows.
    """
    args = get_args()
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)
    args.tracking_only = False

    dt = 1 / 25
    # window_sizes = (np.array([1,4])/dt).astype(np.int32)
    window_sizes = (np.array([1, 2, 4, 8, 16]) / dt).astype(np.int32)
    # window_sizes = (np.array([8, 16, 32, 64])/dt).astype(np.int32)

    # Outputs
    speeds = {ws: [] for ws in window_sizes}
    Ht = {ws: [] for ws in window_sizes}
    Hp_mean = {ws: [] for ws in window_sizes}
    Hp_max = {ws: [] for ws in window_sizes}

    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        args.trial = reconstruction.trial.id

        # Calculate windowed statistics
        stats = calculate_windowed_helicity(args, window_sizes)
        for ws in window_sizes:
            speeds[ws].append(stats['speeds'][ws])
            Ht[ws].append(stats['Ht'][ws])
            Hp_mean[ws].append(stats['Hp_mean'][ws])
            Hp_max[ws].append(stats['Hp_max'][ws])

    # Join outputs
    for ws in window_sizes:
        speeds[ws] = np.concatenate(speeds[ws])
        Ht[ws] = np.concatenate(Ht[ws])
        Hp_mean[ws] = np.concatenate(Hp_mean[ws])
        Hp_max[ws] = np.concatenate(Hp_max[ws])

    # Plot
    fig = plt.figure(figsize=(len(window_sizes) * 4, 10))
    gs = GridSpec(2, len(window_sizes))

    for i, ws in enumerate(window_sizes):
        # Scatter plot of Ht vs MEAN Hp
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f'Window size = {ws}')
        ax.set_xlabel('Ht')
        ax.set_ylabel('MEAN(Hp)')
        s = ax.scatter(Ht[ws], Hp_mean[ws], c=speeds[ws], s=1, cmap='PRGn', norm=MidpointNormalize(midpoint=0))
        cb = fig.colorbar(s)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

        # Scatter plot of Ht vs MAX Hp
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f'Window size = {ws}')
        ax.set_xlabel('Ht')
        ax.set_ylabel('MAX(Hp)')
        s = ax.scatter(Ht[ws], Hp_max[ws], c=speeds[ws], s=1, cmap='PRGn', norm=MidpointNormalize(midpoint=0))
        cb = fig.colorbar(s)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270)

    fig.tight_layout()

    if save_plots:
        args.trial = None
        args.reconstruction = None
        plt.savefig(
            make_filename('helicities_postures_vs_trajectory_windows', args, excludes=['trial', 'deltas'])
        )
    if show_plots:
        plt.show()


def _generate_or_load_dataset_turn_stats(
        args: Namespace,
        ds: Dataset,
        rebuild_cache: bool = False
) -> Dict[str, np.ndarray]:
    spec = to_dict(args)
    stat_keys = ['distances', 'speeds', 'nonp', 'run_distances', 'run_speeds', 'nonp_runs']

    cache_path = LOGS_PATH / hash_data(spec)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    data = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            outputs = {k: data[k] for k in stat_keys}
            logger.info('Loaded stats from cache.')
        except Exception as e:
            logger.warning(f'Failed to load stats: {e}')
            data = None
    if data is None:
        logger.info('Generating dataset stats.')

        # Calculate the model for all trials
        outputs = {k: [] for k in stat_keys}
        for trial in ds.include_trials:
            args.trial = trial.id
            args.reconstruction = ds.get_reconstruction_id_for_trial(trial)
            args.tracking_only = args.reconstruction is None
            try:
                stats = calculate_trial_turn_statistics(args, args.window_size)
            except RuntimeError as e:
                logger.warning(f'Failed to find approximation: "{e}"')
            for k in stat_keys:
                outputs[k].append(stats[k])

        n_trajectories = len(outputs['distances'])
        logger.info(f'Calculated turn statistics for {n_trajectories} out of a possible {len(ds.include_trials)}.')

        # Join outputs
        for k in stat_keys:
            outputs[k] = np.concatenate(outputs[k])

        logger.info(f'Saving stats to {cache_path}.')
        np.savez(cache_path, **outputs)

    return outputs


def speed_vs_nonplanarity_of_turns_and_runs():
    """
    Plot the dataset non-planarities of turns and runs using approximations.
    """
    args = get_args()
    assert args.dataset is not None
    assert args.planarity_window is not None
    ds = Dataset.objects.get(id=args.dataset)
    # args.dataset = None
    dt = 1 / 25
    ws = 10
    args.window_size = int(ws / dt)

    stats = _generate_or_load_dataset_turn_stats(args, ds, rebuild_cache=False)

    # Set up plots
    plt.rc('axes', titlesize=7)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=1,
        ncols=1,
        wspace=0,
        hspace=0,
        top=0.91,
        bottom=0.15,
        left=0.16,
        right=0.97,
    )
    fig = plt.figure(figsize=(2.56, 2.13))

    # Plot correlations
    scatter_args = dict(s=4, alpha=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Speed vs non-planarity')
    ax.set_xlabel('Non-planarity')
    ax.set_ylabel('Speed (mm/s)')
    ax.scatter(stats['nonp'], stats['speeds'], c='fuchsia', marker='o', facecolors='none', label='Turns',
               **scatter_args)
    ax.scatter(stats['nonp_runs'], stats['run_speeds'], c='lawngreen', marker='x', label='Runs', **scatter_args)
    legend = ax.legend()
    legend.legendHandles[0].set_sizes([10.0])
    legend.legendHandles[0].set_alpha(1.)
    legend.legendHandles[0].set_facecolors('none')
    legend.legendHandles[0].set_edgecolors('purple')
    legend.legendHandles[1].set_sizes([10.0])
    legend.legendHandles[1].set_alpha(1.)
    legend.legendHandles[1].set_edgecolors('green')

    # Add fit lines
    def funcinv(x_, a_, b_, k_):
        return a_ + k_ / (x_ + b_)

    # Filter the data
    z_turns = stats['nonp'] * stats['speeds']
    idxs_turns = np.argsort(z_turns)[-100:]
    nonp_turns = stats['nonp'][idxs_turns]
    speeds_turns = stats['speeds'][idxs_turns]
    z_runs = stats['nonp_runs'] * stats['run_speeds']
    idxs_runs = np.argsort(z_runs)[-100:]
    nonp_runs = stats['nonp_runs'][idxs_runs]
    speeds_runs = stats['run_speeds'][idxs_runs]

    bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
    p_turns = curve_fit(funcinv, nonp_turns, speeds_turns, bounds=bounds)[0]
    p_runs = curve_fit(funcinv, nonp_runs, speeds_runs, bounds=bounds)[0]

    def func_min_turns(x0_):
        return (stats['speeds'].max() - funcinv(x0_, *p_turns))**2

    def func_min_runs(x0_):
        return (stats['run_speeds'].max() - funcinv(x0_, *p_runs))**2

    res_turns = minimize(func_min_turns, 0.)
    x_turns = np.linspace(res_turns.x[0], stats['nonp'].max(), 1000)
    ax.plot(x_turns, funcinv(x_turns, *p_turns), color='purple', linewidth=3, linestyle='--')

    res_runs = minimize(func_min_runs, 0.)
    x_runs = np.linspace(res_runs.x[0], stats['nonp_runs'].max() * 1.2, 1000)
    ax.plot(x_runs, funcinv(x_runs, *p_runs), color='green', linewidth=3, linestyle=':')

    # Save / show
    if save_plots:
        plt.savefig(
            make_filename('speed_vs_nonp', args, excludes=['trial', 'frames', 'src', 'deltas', 'delta_step']),
            transparent=True
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    # nonplanarity_dataset()
    # nonplanarity_trajectory()
    # nonplanarity_postures_vs_trajectories()
    # nonplanarity_postures_vs_trajectory_windows()
    # helicity_postures_vs_trajectory_windows()
    speed_vs_nonplanarity_of_turns_and_runs()
