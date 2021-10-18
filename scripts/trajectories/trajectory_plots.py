import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectory.util import generate_or_load_trajectory_cache, smooth_trajectory


show_plots = False
save_plots = True
img_extension = 'png'


def get_args():
    parser = build_target_arguments_parser()
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--n-frames', type=int, help='Number of frames to use.')
    args = parser.parse_args()
    assert not (args.trial is None and args.frame_sequence is None), 'Trial or FS must be specified.'
    return args


def get_trajectory(args: Namespace) -> np.ndarray:
    """
    Load the full 3D trajectory.
    """
    X, meta = generate_or_load_trajectory_cache(
        args.trial,
        args.midline3d_source,
        args.midline3d_source_file,
        args.rebuild_cache
    )
    return X


def plot_trajectory_head_tail_distance(X: np.ndarray, args: Namespace):
    """
    Draw the trajectory coloured by the head-to-tail distance.
    """
    X = smooth_trajectory(X)
    ht_distances = np.linalg.norm(X[:, 0] - X[:, -1], axis=1)
    # scores = (ht_distances - ht_distances.min()) / (ht_distances.max() - ht_distances.min())
    x, y, z = X.mean(axis=1).T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=ht_distances, cmap='OrRd_r', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Head-tail distance. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_head-tail-distance'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_signed_speed(X: np.ndarray, args: Namespace):
    """
    Draw the trajectory coloured by the signed speed.
    """
    X = smooth_trajectory(X)
    com = X.mean(axis=1)
    directional_gradients = np.gradient(com, axis=0)
    speeds = np.linalg.norm(directional_gradients, axis=1)
    ht_directions = X[:, 0] - X[:, -1]
    ht_dot_dir = np.einsum('ni,ni->n', ht_directions, directional_gradients)
    fwd_or_back = np.sign(ht_dot_dir)
    signed_speed = speeds * fwd_or_back
    x, y, z = com.T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=signed_speed, cmap='PRGn', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Speed. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_signed-speed'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def head_tail_distance_plot():
    args = get_args()
    X = get_trajectory(args)
    interactive()
    plot_trajectory_head_tail_distance(X, args)


def trajectory_signed_speed_plot():
    args = get_args()
    X = get_trajectory(args)
    interactive()
    plot_trajectory_signed_speed(X, args)


if __name__ == '__main__':
    head_tail_distance_plot()
    trajectory_signed_speed_plot()
