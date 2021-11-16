import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = LOGS_PATH + '/' + START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'aggregation', 'deltas', 'u', 'smoothing_window', 'directionality',
              'projection']:
        if k in excludes:
            continue
        if k == 'trial':
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
        elif k == 'aggregation':
            fn += f'_{args.aggregation}'
        elif k == 'deltas':
            fn += f'_d={",".join([str(d) for d in args.deltas])}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'projection':
            fn += f'_p={args.projection}'

    return fn + '.' + img_extension


def planarity_vs_delta():
    """
    Plot the planarity across different time windows.
    """
    args = get_args()
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    # deltas = np.array([101])
    delta_ts = deltas / 25

    # Calculate planarities across different time windows
    planarities = []
    for window_size in deltas:
        args.window_size = int(window_size)
        logger.info(f'Fetching PCA data for window size = {int(window_size)}.')
        pca_cache = get_pca_cache_from_args(args)
        planarities.append(1 - pca_cache.explained_variance_ratio[:, 2])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot()

    parts = plt.violinplot(planarities, delta_ts, widths=args.delta_step / 25, showmeans=True, showmedians=True)
    parts['cmedians'].set_color('green')
    parts['cmedians'].set_alpha(0.7)
    parts['cmedians'].set_linestyle(':')
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_alpha(0.7)
    parts['cmeans'].set_linestyle('--')

    ax.set_xlabel('$\Delta s$')
    ax.set_ylabel('Planarity')
    ax.set_title(f'Planarity vs Delta (time window). Trial {args.trial}.')
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('planarity_vs_delta', args))
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    planarity_vs_delta()
