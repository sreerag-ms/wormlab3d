import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Dataset
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['dataset', 'trial', 'frames', 'src', 'deltas', 'delta_step', 'u', 'smoothing_window', 'directionality']:
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
        r = pca_cache.explained_variance_ratio.T
        nonp_delta = r[2] / np.sqrt(r[1] * r[0])
        nonp.append(nonp_delta)

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

    # Use exponentially-spaced deltas
    if args.delta_step < 0:
        delta = args.min_delta
        deltas = []
        while delta < args.max_delta:
            deltas.append(delta)
            delta = delta**(-args.delta_step)
        deltas = np.array(deltas).astype(np.int64)

    # Use equally-spaced deltas
    else:
        deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    delta_ts = deltas / 25

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
    for trial in ds.include_trials:
        logger.info(f'Calculating non-planarity for trial={trial.id}.')
        args.trial = trial.id

        # Group results by concentration
        c = trial.experiment.concentration
        if c not in nonp:
            nonp[c] = {delta: [] for delta in deltas}

        # Calculate non-planarity across different time windows
        for delta in deltas:
            args.planarity_window = int(delta)
            logger.info(f'Fetching PCA data for window size = {int(delta)}.')
            try:
                pca_cache = get_pca_cache_from_args(args)
                r = pca_cache.explained_variance_ratio.T
                nonp_delta = r[2] / np.sqrt(r[1] * r[0])
                nonp[c][delta].extend(nonp_delta)
                all_nonp[delta].extend(nonp_delta)
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
        n_trials = len(nonp_c)
        n_trials_total += n_trials
        ax.set_title(f'Concentration = {c:.2f}% ({n_trials} trials)')
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


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    args_ = get_args()
    if args_.dataset is not None:
        nonplanarity_dataset()
    else:
        nonplanarity_trajectory()
