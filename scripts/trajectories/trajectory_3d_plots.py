import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_planarity, calculate_speeds, calculate_htd

show_plots = True
save_plots = False
img_extension = 'png'


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    return X_full, X_slice


def plot_trajectory_head_tail_distance():
    """
    Draw the trajectory coloured by the head-to-tail distance.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    htd = calculate_htd(X_full)
    x, y, z = X_slice.T
    # scores = (ht_distances - ht_distances.min()) / (ht_distances.max() - ht_distances.min())

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=htd, cmap='OrRd_r', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Head-tail distance. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_HTD'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_signed_speed():
    """
    Draw the trajectory coloured by the signed speed.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)
    x, y, z = X_slice.T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=signed_speeds, cmap='PRGn', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Speed. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_signed-speed'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_K():
    """
    Draw the trajectory coloured by the estimate of K.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    x, y, z = X_slice.T

    K_ests = np.log(get_K_estimates_from_args(args))
    K_ests = np.r_[
        np.ones(int(np.floor(args.K_sample_frames / 2))) * K_ests[0],
        K_ests,
        np.ones(int(np.ceil(args.K_sample_frames / 2))) * K_ests[-1],
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=K_ests, cmap='OrRd_r', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'log(K_est). Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_K-est'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_planarity():
    """
    Draw the trajectory coloured by the planarity.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    x, y, z = X_slice.T
    planarity = calculate_planarity(X_full, window_size=args.planarity_window)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=planarity, cmap='OrRd', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Planarity. Trial {args.trial}. PCA window: {args.planarity_window} frames.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_planarity'
            f'_w={args.planarity_window}'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_brownian_trajectory():
    """
    Generate and plot trajectory of a randomly generated brownian particle.
    """
    D = 100
    n_steps = 1000
    total_time = 1
    p = BrownianParticle(D=D)
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
    x, y, z = X.T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    s = ax.scatter(x, y, z, c=np.linspace(0, 1, len(X)), cmap='jet', s=10, alpha=0.4)
    fig.colorbar(s)
    ax.set_title(f'Brownian particle. D={D}, n_steps={n_steps}, total_time={total_time}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_brownian_particle'
            f'_D={D}_n={n_steps}_T={total_time}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_trajectory_head_tail_distance()
    plot_trajectory_signed_speed()
    plot_trajectory_K()
    plot_trajectory_planarity()
    plot_brownian_trajectory()
