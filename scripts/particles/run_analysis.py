import os
import shutil
from argparse import Namespace
from typing import Dict

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset
from wormlab3d.particles.tumble_run import generate_or_load_ds_statistics
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_speeds

show_plots = True
# save_plots = False
# show_plots = False
save_plots = True
interactive_plots = False
img_extension = 'png'

DATA_CACHE_PATH = LOGS_PATH / 'cache'
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# DIST_KEYS = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']
DATA_KEYS = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles', 'tumble_idxs']


def _identifiers(args: Namespace) -> str:
    return (f'ds={args.dataset}'
            f'_{args.approx_method}'
            f'_e={args.approx_error_limit}'
            f'_pw={args.planarity_window_vertices}'
            f'_d={args.approx_distance}'
            f'_h={args.approx_curvature_height}'
            f'_sw={args.smoothing_window_K}'
            f'_a={"euler" if args.approx_use_euler_angles else "project"}'
            f'_pT={args.pause_speed_threshold}'
            f'_pD={args.min_pause_duration}')


def _calculate_pause_data(
        args: Namespace,
        ds: Dataset,
        data_values: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Calculate the pause data.
    """
    logger.info('Calculating pause data.')
    tumble_idxs = data_values['tumble_idxs']
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']

    pause_durations = []
    pause_positions = []
    thetas_pre = []
    thetas_post = []
    phis_pre = []
    phis_post = []

    first_pause_durations = []
    first_pause_positions = []
    first_thetas_pre = []
    first_thetas_post = []
    first_phis_pre = []
    first_phis_post = []

    last_pause_durations = []
    last_pause_positions = []
    last_thetas_pre = []
    last_thetas_post = []
    last_phis_pre = []
    last_phis_post = []

    run_durations = []  # Includes the start and end runs that aren't bounded by tumbles
    run_pause_ratios = []

    # Loop over the reconstruction approximations and extract the pause statistics
    ii = 0
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Computing parameters for trial={trial.id}.')
        tumble_idxs_i = tumble_idxs[i]
        thetas_i = thetas[ii: ii + len(tumble_idxs_i)]
        phis_i = phis[ii: ii + len(tumble_idxs_i)]
        args.trial = trial.id
        X = get_trajectory_from_args(args)
        dt = 1 / trial.fps
        min_pause_duration = args.min_pause_duration / dt

        # Take centre of mass
        if X.ndim == 3:
            X = X.mean(axis=1)
        X -= X.mean(axis=0)

        # Split up the trajectory into the run sections to calculate pauses
        logger.info('Calculating pauses.')
        speed = calculate_speeds(X) / dt
        v0 = 0
        for j in range(len(tumble_idxs_i) + 1):
            v1 = tumble_idxs_i[j] if j < len(tumble_idxs_i) else -1
            run_speed = speed[v0:v1]

            # Calculate the pauses
            is_paused = run_speed <= args.pause_speed_threshold
            while True:
                pause_idxs, pause_props = find_peaks(is_paused, height=0.5, width=1)
                unchanged = True
                for k in range(len(pause_idxs)):
                    # If the pause is too small either merge it with an adjacent pause or discard it
                    if pause_props['widths'][k] < min_pause_duration:
                        pause_start = pause_props['left_bases'][k] + 1
                        pause_end = pause_props['right_bases'][k]
                        unchanged = False

                        # Check if previous pause is close enough
                        if k > 0:
                            prev_pause_end = pause_props['right_bases'][k - 1]
                            if pause_start - prev_pause_end < min_pause_duration:
                                is_paused[prev_pause_end:pause_start] = True
                                break

                        # Check if next pause is close enough
                        if k < len(pause_idxs) - 1:
                            next_pause_start = pause_props['left_bases'][k + 1] + 1
                            if next_pause_start - pause_end < min_pause_duration:
                                is_paused[pause_end:next_pause_start] = True
                                break

                        # Discard the pause
                        is_paused[pause_start:pause_end] = False
                        break
                if unchanged:
                    break

            # Collect the pause data
            for k in range(len(pause_idxs)):
                pause_durations.append(pause_props['widths'][k] * dt)
                pause_positions.append(pause_idxs[k] / len(run_speed))
                if j > 0:
                    thetas_pre.append(thetas_i[j - 1])
                    phis_pre.append(phis_i[j - 1])
                else:
                    thetas_pre.append(np.nan)
                    phis_pre.append(np.nan)
                if j < len(tumble_idxs_i):
                    thetas_post.append(thetas_i[j])
                    phis_post.append(phis_i[j])
                else:
                    thetas_post.append(np.nan)
                    phis_post.append(np.nan)

            # Collect the first and last pause data
            if len(pause_idxs) > 0:
                first_pause_durations.append(pause_props['widths'][0] * dt)
                first_pause_positions.append(pause_idxs[0] / len(run_speed))
                first_thetas_pre.append(thetas_i[j - 1] if j > 0 else np.nan)
                first_phis_pre.append(phis_i[j - 1] if j > 0 else np.nan)
                first_thetas_post.append(thetas_i[j] if j < len(tumble_idxs_i) else np.nan)
                first_phis_post.append(phis_i[j] if j < len(tumble_idxs_i) else np.nan)
                last_pause_durations.append(pause_props['widths'][-1] * dt)
                last_pause_positions.append(pause_idxs[-1] / len(run_speed))
                last_thetas_pre.append(thetas_i[j - 1] if j > 0 else np.nan)
                last_phis_pre.append(phis_i[j - 1] if j > 0 else np.nan)
                last_thetas_post.append(thetas_i[j] if j < len(tumble_idxs_i) else np.nan)
                last_phis_post.append(phis_i[j] if j < len(tumble_idxs_i) else np.nan)

            # Collect the percentage of the run that was spent in the pause state
            run_durations.append(len(run_speed) * dt)
            run_pause_ratios.append(np.sum(is_paused) / len(is_paused))

            v0 = v1
        ii += len(tumble_idxs_i)

    return {
        'pause_durations': np.array(pause_durations),
        'pause_positions': np.array(pause_positions),
        'thetas_pre': np.array(thetas_pre),
        'thetas_post': np.array(thetas_post),
        'phis_pre': np.array(phis_pre),
        'phis_post': np.array(phis_post),
        'first_pause_durations': np.array(first_pause_durations),
        'first_pause_positions': np.array(first_pause_positions),
        'first_thetas_pre': np.array(first_thetas_pre),
        'first_thetas_post': np.array(first_thetas_post),
        'first_phis_pre': np.array(first_phis_pre),
        'first_phis_post': np.array(first_phis_post),
        'last_pause_durations': np.array(last_pause_durations),
        'last_pause_positions': np.array(last_pause_positions),
        'last_thetas_pre': np.array(last_thetas_pre),
        'last_thetas_post': np.array(last_thetas_post),
        'last_phis_pre': np.array(last_phis_pre),
        'last_phis_post': np.array(last_phis_post),
        'run_durations': np.array(run_durations),
        'run_pause_ratios': np.array(run_pause_ratios),
    }


def _generate_or_load_pause_data(
        args: Namespace,
        ds: Dataset,
        data_values: Dict[str, np.ndarray],
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    cache_path = DATA_CACHE_PATH / ('pause_data_' + _identifiers(args))
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    data = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        data = _calculate_pause_data(args, ds, data_values)
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return data


def _init(include_all_runs: bool = False):
    """
    Initialise the arguments, save dir and load the dataset statistics.
    """
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=False,
        include_K_options=False,
        include_planarity_options=True,
        include_helicity_options=False,
        include_manoeuvre_options=True,
        include_approximation_options=True,
        include_pe_options=True,
        include_fractal_dim_options=False,
        include_video_options=False,
        include_evolution_options=True,
        validate_source=False,
    )

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Add new arguments
    args.pause_speed_threshold = 0.02
    args.min_pause_duration = 0.5

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            if k in ['theta_dist_params', 'phi_dist_params']:
                v = [float(vv) for vv in v.split(',')]
            setattr(args, k, v)
    print_args(args)

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    approx_args = dict(
        approx_method=args.approx_method,
        error_limits=[args.approx_error_limit],
        planarity_window=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.smoothing_window_K,
        smooth_K_first=args.smoothing_window_K,
        use_euler_angles=args.approx_use_euler_angles,
    )
    if include_all_runs:
        approx_args['min_run_speed_duration'] = (0, 10000)

    # Fetch dataset
    ds = Dataset.objects.get(id=args.dataset)

    # Generate or load tumble/run values - ensures the cache is built
    ds_stats = generate_or_load_ds_statistics(
        ds=ds,
        rebuild_cache=args.regenerate,
        **approx_args
    )
    # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles, tumble_idxs
    data = {k: ds_stats[i + 1][0] for i, k in enumerate(DATA_KEYS)}

    # Generate or load pause data
    pause_data = _generate_or_load_pause_data(args, ds, data, rebuild_cache=False, cache_only=False)
    data.update(pause_data)

    return save_dir, data


def plot_pause_durations_and_positions():
    save_dir, data = _init()
    durations = data['pause_durations']
    positions = data['pause_positions']
    logger.info('Plotting pause durations and positions.')

    # Plot the results
    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 3),
        height_ratios=(3, 4),
        wspace=0,
        hspace=0,
        top=0.92,
        bottom=0.12,
        left=0.08,
        right=0.96,
    )
    fig = plt.figure(figsize=(10, 6))

    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)
    # scatter_args = dict(s=10, c=durations, alpha=0.6)
    scatter_args = dict(s=10, alpha=0.6, marker='$\u25EF$')
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Scatter plot - positions/durations
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_scat.scatter(positions, durations, **scatter_args)
    ax_scat.set_yscale('log')
    ax_scat.set_xlabel('Position')
    ax_scat.set_xticks([0, 0.5, 1])
    ax_scat.set_ylabel('Duration (s)')
    ax_scat.spines['top'].set_visible(False)
    ax_scat.spines['right'].set_visible(False)

    def _make_hist(ax_, vals):
        ax_.hist(vals, **hist_args)
        ax_.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_.spines['bottom'].set(linestyle='--', color='grey')

    # Pause position histogram
    ax_hist_pos = fig.add_subplot(gs[0, 0], sharex=ax_scat)
    _make_hist(ax_hist_pos, positions)
    ax_hist_pos.set_ylabel('Density')
    ax_hist_pos.set_title('Pause position in run')

    # Pause duration histogram
    ax_hist_dur = fig.add_subplot(gs[1, 1])  # , sharey=ax_scat)
    ax_hist_dur.hist(np.log(durations), orientation='horizontal', color='green', **hist_args)
    ax_hist_dur.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_dur.spines['left'].set(linestyle='--', color='grey')
    ax_hist_dur.set_xlabel('Density')
    ax_hist_dur.set_title('Pause duration')

    if save_plots:
        plt.savefig(save_dir / f'pause_durations_and_positions.{img_extension}')
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_pause_durations_and_angles():
    save_dir, data = _init()
    endpoints_only = False
    pos_thresh = 0.3

    if endpoints_only:
        d_early = data['first_pause_durations']
        d_late = data['last_pause_durations']
        thetas_pre = np.abs(data['first_thetas_pre'])
        thetas_post = np.abs(data['first_thetas_post'])
        phis_pre = np.abs(data['first_phis_pre'])
        phis_post = np.abs(data['first_phis_post'])
    else:
        durations = data['pause_durations']
        positions = data['pause_positions']
        d_early = durations[positions < pos_thresh]
        d_late = durations[positions > 1 - pos_thresh]
        thetas_pre = np.abs(data['thetas_pre'][positions < pos_thresh])
        thetas_post = np.abs(data['thetas_post'][positions > 1 - pos_thresh])
        phis_pre = np.abs(data['phis_pre'][positions < pos_thresh])
        phis_post = np.abs(data['phis_post'][positions > 1 - pos_thresh])

    logger.info('Plotting pause durations and positions against angles.')

    # Plot the results
    gs = GridSpec(
        nrows=2,
        ncols=2,
        hspace=0.4,
        top=0.9,
        bottom=0.08,
        left=0.08,
        right=0.98,
    )
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Pause durations and angles\n('
                 + ('first and last pauses only' if endpoints_only else
                    f'positions < {pos_thresh} or > {1 - pos_thresh}') + ')')
    scatter_args = dict(s=10, alpha=0.6, marker='$\u25EF$')

    def _make_scat(ax_, d, a, a_type, prev_or_next):
        nan_idxs = np.isnan(a)
        d = d[~nan_idxs]
        a = a[~nan_idxs]
        res = pearsonr(d, a)
        ax_.scatter(d, a, **scatter_args)
        ax_.set_xlabel('Pause duration (s)')
        ax_.set_xscale('log')

        if prev_or_next == 'prev':
            title = 'Previous'
        else:
            title = 'Next'
        if a_type == 'theta':
            title += ' planar angles'
            ax_.set_yticks([0, np.pi / 2, np.pi])
            ax_.set_yticklabels(['0', '$\pi/2$', '$\pi$'])
        else:
            title += ' non-planar angles'
            ax_.set_yticks([0, np.pi / 4, np.pi / 2])
            ax_.set_yticklabels(['0', '$\pi/4$', '$\pi/2$'])
        ax_.set_title(title + f'\nR={res[0]:.2E}, p={res[1]:.2E}')
        ax_.set_ylabel('$\\' + a_type + '_{{' + prev_or_next + '}}$')

    ax = fig.add_subplot(gs[0, 0])
    _make_scat(ax, d_early, thetas_pre, 'theta', 'prev')

    ax = fig.add_subplot(gs[1, 0])
    _make_scat(ax, d_early, phis_pre, 'phi', 'prev')

    ax = fig.add_subplot(gs[0, 1])
    _make_scat(ax, d_late, thetas_post, 'theta', 'next')

    ax = fig.add_subplot(gs[1, 1])
    _make_scat(ax, d_late, phis_post, 'phi', 'next')

    if save_plots:
        plt.savefig(save_dir / f'pause_durations_and_angles.{img_extension}')
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_pause_proportions():
    save_dir, data = _init()
    logger.info('Plotting pause proportions.')
    run_durations = data['run_durations']
    pause_ratios = data['run_pause_ratios']
    run_durations = run_durations[pause_ratios > 0]
    pause_ratios = pause_ratios[pause_ratios > 0]

    # Plot the results
    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 3),
        height_ratios=(3, 4),
        wspace=0,
        hspace=0,
        top=0.92,
        bottom=0.12,
        left=0.08,
        right=0.96,
    )
    fig = plt.figure(figsize=(10, 6))

    scatter_args = dict(s=10, alpha=0.6, marker='$\u25EF$')
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Scatter plot - positions/durations
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_scat.scatter(run_durations, pause_ratios, **scatter_args)
    ax_scat.set_xscale('log')
    # ax_scat.set_yscale('log')
    ax_scat.set_xlabel('Run duration (s)')
    # ax_scat.set_xticks([0, 0.5, 1])
    ax_scat.set_ylabel('Pause ratio')
    ax_scat.spines['top'].set_visible(False)
    ax_scat.spines['right'].set_visible(False)

    # Run duration histogram
    ax_hist_dur = fig.add_subplot(gs[0, 0])  # , sharex=ax_scat)
    ax_hist_dur.hist(np.log(run_durations), **hist_args)
    ax_hist_dur.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_hist_dur.spines['bottom'].set(linestyle='--', color='grey')
    ax_hist_dur.set_ylabel('Density')
    ax_hist_dur.set_title('Run duration')

    # Pause ratio histogram
    ax_hist_ratio = fig.add_subplot(gs[1, 1])  # , sharey=ax_scat)
    # ax_hist_ratio.hist(np.log(pause_ratios[pause_ratios>0]), orientation='horizontal', color='green', **hist_args)
    ax_hist_ratio.hist(pause_ratios, orientation='horizontal', color='green', **hist_args)
    ax_hist_ratio.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_ratio.spines['left'].set(linestyle='--', color='grey')
    ax_hist_ratio.set_xlabel('Density')
    ax_hist_ratio.set_title('Ratio of run spent paused')

    if save_plots:
        plt.savefig(save_dir / f'pause_ratios.{img_extension}')
    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    # plot_pause_durations_and_positions()
    # plot_pause_durations_and_angles()
    plot_pause_proportions()
