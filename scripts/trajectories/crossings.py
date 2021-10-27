import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform, pdist
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_speeds

show_plots = True
save_plots = True
img_extension = 'png'


def check_crossings():
    """
    """
    args = get_args()
    args.trajectory_point = None  # Just use the centre-of-mass point
    X = get_trajectory_from_args(args)
    # X = X[:100]
    com = X.mean(axis=1)
    N = len(X)
    fps = 25
    ts = np.linspace(0, N / fps, N)

    min_dt = 30
    min_steps = min_dt * fps

    # Calculate the distances from each trajectory point to each other trajectory point
    logger.info('Calculating pairwise distances.')
    dists = squareform(pdist(com, metric='euclidean'))

    # Remove distances to close (in time) trajectory points
    removals = np.eye(N)
    for step in range(1, min_steps + 1):
        removals += np.eye(N, k=step)
        removals += np.eye(N, k=-step)
    dists = dists * (1 - removals)

    # Calculate the minimum nonzero distance at each point along the trajectory
    logger.info('Calculating minimum distances.')
    min_dists = np.min(np.where(dists > 0, dists, np.inf), axis=1)

    # Calculate speeds
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True)

    logger.info('Plotting.')
    fig, axes = plt.subplots(2, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.set_title(f'Closest distance to trajectory > {min_dt}s away.')
    ax.plot(ts, min_dists)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Euclidean distance')

    ax = axes[1]
    ax.set_title(f'Speed. Smoothing window {args.smoothing_window / fps:.2f}s.')
    ax.plot(ts, speeds)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_sw={args.smoothing_window}'
            f'_min-dt={min_dt}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    check_crossings()
