import os
from argparse import Namespace
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import GridSpec
from numpy.lib.stride_tricks import sliding_window_view

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.manoeuvres import get_forward_stats, get_manoeuvres
from wormlab3d.trajectories.util import smooth_trajectory

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

show_plots = True
save_plots = False
img_extension = 'svg'


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    trajectory_point = args.trajectory_point
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    args.trajectory_point = trajectory_point
    return X_full, X_slice


def _run_stats_identifiers(args: Namespace) -> str:
    return f'runs_ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_ff={args.min_forward_frames}' \
           f'_fs={args.min_forward_speed}'


def _manoeuvre_stats_identifiers(args: Namespace) -> str:
    return f'manoeuvres_ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_rf={args.min_reversal_frames}' \
           f'_rd={args.min_reversal_distance}' \
           f'_ff={args.min_forward_frames}' \
           f'_fd={args.min_forward_distance}' \
           f'_mw={args.manoeuvre_window}'


def _generate_or_load_run_stats(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _run_stats_identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = ['times', 'durations', 'distances', 'speeds']
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = {k: [] for k in keys}

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)
            runs = get_forward_stats(
                X_full,
                X_slice,
                min_forward_frames=args.min_forward_frames,
                min_speed=args.min_forward_speed
            )
            for r in runs:
                res['times'].append(r['end_idx'] / fps)
                res['durations'].append(r['duration'] / fps)
                res['distances'].append(r['distance'])
                res['speeds'].append(r['speed'] * fps)

        for k in keys:
            res[k] = np.array(res[k])
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def _generate_or_load_manoeuvre_stats(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _manoeuvre_stats_identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = ['trials', 'times', 'durations', 'distances']
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = {k: [] for k in keys}

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)
            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=args.min_reversal_frames,
                min_reversal_distance=args.min_reversal_distance,
                min_forward_frames=args.min_forward_frames,
                min_forward_distance=args.min_forward_distance,
                window_size=args.manoeuvre_window,
                cut_windows_at_manoeuvres=True,
                align_components_with_traj=True
            )

            # Loop over manoeuvres
            for m in manoeuvres:
                res['trials'].append(reconstruction.trial.id)
                res['times'].append(m['centre_idx'] / fps)
                res['durations'].append(m['reversal_duration'] / fps)
                res['distances'].append(m['reversal_distance'])

        for k in keys:
            res[k] = np.array(res[k])
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def _generate_or_load_manoeuvre_rates(
        args: Namespace,
        rate_window: int = 60,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / (f'rates_{rate_window}_' + _manoeuvre_stats_identifiers(args))
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = ['rates_padded', 'divisors', 'rate_means']
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        rates = []

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)
            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=args.min_reversal_frames,
                min_reversal_distance=args.min_reversal_distance,
                min_forward_frames=args.min_forward_frames,
                min_forward_distance=args.min_forward_distance,
                window_size=args.manoeuvre_window,
                cut_windows_at_manoeuvres=True,
                align_components_with_traj=True
            )

            N = len(X_full)
            events = np.zeros(N)
            for m in manoeuvres:
                events[m['centre_idx']] = 1

            rw = int(rate_window * fps)
            pad = int(rw / 2)
            events = np.concatenate([np.zeros(pad), events, np.zeros(pad - np.remainder(N, 2))])

            events = smooth_trajectory(events, window_len=rw - 1 + np.remainder(rw, 2), window_type='hanning')
            # events = events/events.max()

            events_windowed = sliding_window_view(events, rw)
            rates.append(events_windowed.sum(axis=1))

        max_length = max([len(r) for r in rates])
        divisors = np.zeros(max_length)
        rates_padded = []
        for r in rates:
            divisors[:len(r)] += 1
            rp = np.concatenate([r, np.zeros(max_length - len(r))])
            rates_padded.append(rp)
        rates_padded = np.stack(rates_padded)
        rps = rates_padded.sum(axis=0)
        rate_means = rps / divisors

        res = {
            'rates_padded': rates_padded,
            'divisors': divisors,
            'rate_means': rate_means,
        }
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def plot_lengths_against_start_times(
        runs_or_manoeuvres: str = 'runs',
        distances_or_durations: str = 'distances',
        log_x: bool = False,
        log_y: bool = False,
        layout: str = 'paper'
):
    """
    Plot the correlations between run and manoeuvre distances/durations and where they occurred in the trial.
    """
    args = get_args(validate_source=False)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    if runs_or_manoeuvres == 'runs':
        res = _generate_or_load_run_stats(args, rebuild_cache=False, cache_only=False)
    else:
        res = _generate_or_load_manoeuvre_stats(args, rebuild_cache=False, cache_only=False)
    times = res['times']
    data = res[distances_or_durations]

    # Set up plot
    if layout == 'paper':
        plt.rc('axes', titlesize=7)  # fontsize of the title
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        gs = GridSpec(
            nrows=1,
            ncols=1,
            top=0.92,
            bottom=0.16,
            left=0.2,
            right=0.96,
        )
        fig = plt.figure(figsize=(1.74, 2.19))
    else:
        plt.rc('axes', titlesize=9)  # fontsize of the title
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6.5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6.5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        gs = GridSpec(
            nrows=1,
            ncols=1,
            top=0.91,
            bottom=0.16,
            left=0.14,
            right=0.99,
        )
        fig = plt.figure(figsize=(2.9, 2.3))

    # Plot correlations
    logger.info('Plotting')
    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(times, data, alpha=0.5, s=2)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel('Time elapsed (s)')
    ylabel_prefix = 'Run' if runs_or_manoeuvres == 'runs' else 'Reversal'
    if distances_or_durations == 'distances':
        ax.set_ylabel(f'{ylabel_prefix} distance (mm)')
    else:
        ax.set_ylabel(f'{ylabel_prefix} duration (s)')

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_{runs_or_manoeuvres}_{distances_or_durations}' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_ff={args.min_forward_frames}' \
               f'_fs={args.min_forward_speed}' \
               f'_N={len(data)}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_rates(
        rate_window: int = 60,
        runs_or_manoeuvres: str = 'runs',
        log_x: bool = False,
        log_y: bool = False,
        layout: str = 'paper'
):
    """
    Plot the runs/manoeuvre rates over time.
    """
    args = get_args(validate_source=False)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    if runs_or_manoeuvres == 'runs':
        exit()
    else:
        res = _generate_or_load_manoeuvre_rates(args, rate_window=rate_window, rebuild_cache=False, cache_only=False)

    plt.plot(res['rate_means'])
    plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    # plot_lengths_against_start_times(
    #     runs_or_manoeuvres='manoeuvres',
    #     distances_or_durations='durations',
    #     log_x=False,
    #     log_y=False,
    #     layout='thesis'
    # )

    plot_rates(
        runs_or_manoeuvres='manoeuvres',
        log_x=False,
        log_y=False,
        layout='thesis'
    )
