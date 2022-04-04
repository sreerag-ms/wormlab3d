import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.trajectories.angles import calculate_trajectory_angles_parallel
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args

show_plots = True
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['dataset', 'trial', 'frames', 'src', 'directionality', 'aggregation', 'deltas', 'delta_range',
              'delta_step', 'u']:
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
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'aggregation':
            fn += f'_{args.aggregation}'
        elif k == 'deltas':
            fn += f'_d={",".join([str(d) for d in args.deltas])}'
        elif k == 'delta_range':
            fn += f'_dr={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            if args.delta_step < 0:
                fn += f'_ds={args.delta_step:.2f}'
            else:
                fn += f'_ds={int(args.delta_step)}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'projection':
            fn += f'_p={args.projection}'

    return LOGS_PATH / (fn + '.' + img_extension)


def plot_angle_distributions_dataset():
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

    # Unset midline source args and use centre-of-mass point
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.trajectory_point = -1

    # Use the reconstruction from the dataset where possible
    reconstructions = {}
    for r_ref in ds.reconstructions:
        r = Reconstruction.objects.get(id=r_ref.id)
        reconstructions[r.trial.id] = r.id

    # Calculate the angles for all trials
    angles = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating angles for trial={trial.id}.')
        if trial.id in reconstructions:
            args.reconstruction = reconstructions[trial.id]
            args.trial = None
            args.tracking_only = False
        else:
            args.trial = trial.id
            args.reconstruction = None
            args.tracking_only = True

        # Calculate angles for trial
        trajectory = get_trajectory_from_args(args)
        d = calculate_trajectory_angles_parallel(trajectory, deltas)
        # d = calculate_trajectory_angles(trajectory, deltas)
        c = trial.experiment.concentration
        if c not in angles:
            angles[c] = []
        angles[c].append(d)

    # Sort by concentration
    angles = {k: v for k, v in sorted(list(angles.items()))}

    # Set up plots
    n_rows = 1 + len(angles)
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
        ax_.set_ylabel('$\\theta=\measuredangle\left(x_t-x_{t+\Delta}, x_t-x_{t+\Delta})\\right)$')
        ax_.set_xlabel('$\Delta s$')

    n_trials_total = 0
    all_vals = {}

    # Aggregate results at each concentration
    for i, (c, ds_c) in enumerate(angles.items()):
        ax = axes[i + 1]
        n_trials = len(ds_c)
        n_trials_total += n_trials
        ax.set_title(f'Concentration = {c:.2f}% ({n_trials} trials)')
        c_vals = {}
        for d in ds_c:
            for delta, d_vals in d.items():
                if delta not in c_vals:
                    c_vals[delta] = []
                if delta not in all_vals:
                    all_vals[delta] = []
                c_vals[delta].extend(d_vals)
                all_vals[delta].extend(d_vals)
        _violinplot(ax, c_vals.values())

    # Top plot shows aggregation from all results
    ax = axes[0]
    ax.set_title(f'All concentrations ({n_trials_total} trials)')
    _violinplot(ax, all_vals.values())

    fig.tight_layout()

    if save_plots:
        args.trial = None
        args.reconstruction = None
        plt.savefig(
            make_filename('violin_dataset', args, excludes=['trial', 'deltas'])
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_angle_distributions_dataset()
