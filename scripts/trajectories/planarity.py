import os
from argparse import Namespace
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger, N_WORKERS
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_planarity

# tex_mode()

show_plots = True
save_plots = False
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = LOGS_PATH + '/' + START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'aggregation', 'deltas', 'u']:
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
        elif k == 'projection':
            fn += f'_p={args.projection}'

    return fn + '.' + img_extension


def calculate_planarity_wrapper(args):
    logger.info(f'Calculating planarity for window size = {args[1]}.')
    return calculate_planarity(args[0], window_size=args[1])


def calculate_planarity_parallel(
        X: np.ndarray,
        deltas: np.ndarray,
):
    """
    Calculate the planarities in parallel.
    """
    N = len(X)
    with Pool(processes=N_WORKERS) as pool:
        res_list = pool.map(
            calculate_planarity_wrapper,
            [[X, delta] for delta in deltas]
        )

    res = np.zeros((2, len(deltas) * N))
    for i, delta in enumerate(deltas):
        res[:, i * N:(i + 1) * N] = np.array([
            np.ones(N) * delta,
            res_list[i],
        ])
    return res


def planarity_vs_delta():
    args = get_args()
    X = get_trajectory_from_args(args)
    N = len(X)
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)

    res = calculate_planarity_parallel(X, deltas)

    # res = np.zeros((2, len(deltas) * N))
    # for i, delta in enumerate(deltas):
    #     res[:, i * N:(i+1)*N] = np.array([
    #         np.ones(N) * delta,
    #         calculate_planarity(X, window_size=delta),
    #     ])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=res[0], y=res[1], s=2, alpha=0.4)
    ax.set_xlabel('$\Delta s$')
    ax.set_ylabel('Planarity')
    ax.set_title(f'Planarity vs Delta (time window). Trial {args.trial}.')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('planarity_vs_delta', args)
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    planarity_vs_delta()
