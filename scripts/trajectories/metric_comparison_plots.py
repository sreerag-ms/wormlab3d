import os

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_speeds, calculate_htd, calculate_planarity

show_plots = True
save_plots = False
img_extension = 'png'


def plot_speed_vs_K():
    """
    Plot speed against K estimate.
    """
    args = get_args()
    K_ests = get_K_estimates_from_args(args)
    X = get_trajectory_from_args(args)
    speeds = calculate_speeds(X, signed=True)

    # Trim the speeds to match number of K_ests
    speeds = speeds[int(np.floor(args.K_sample_frames / 2)):-int(np.ceil(args.K_sample_frames / 2))]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=speeds, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('Speed')
    ax.set_ylabel('K_est')
    ax.set_title(f'Speed vs K estimate. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_speed_vs_K'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_speed_vs_HTD():
    """
    Plot speed against HTD.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    htd = calculate_htd(X)
    speeds = calculate_speeds(X, signed=False)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=speeds, y=htd, s=2, alpha=0.4)
    ax.set_xlabel('Speed')
    ax.set_ylabel('HTD')
    ax.set_title(f'Speed vs HTD. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_speed_vs_HTD'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_HTD_vs_K():
    """
    Plot HTD against K estimate.
    """
    args = get_args()
    K_ests = get_K_estimates_from_args(args)
    X = get_trajectory_from_args(args)
    htd = calculate_htd(X)

    # Trim the HTDs to match number of K_ests
    htd = htd[int(np.floor(args.K_sample_frames / 2)):-int(np.ceil(args.K_sample_frames / 2))]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=htd, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('HTD')
    ax.set_ylabel('K_est')
    ax.set_title(f'HTD vs K estimate. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_HTD_vs_K'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_speed_vs_planarity():
    """
    Plot speeds against planarity.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    speeds = calculate_speeds(X, signed=False)
    planarity = calculate_planarity(X, window_size=args.planarity_window)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=speeds, y=planarity, s=2, alpha=0.4)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Planarity')
    ax.set_title(f'Speed vs Planarity ({args.planarity_window} frames). Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_speed_vs_planarity'
            f'_w={args.planarity_window}'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_planarity_vs_K():
    """
    Plot planarity against K estimate.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    planarity = calculate_planarity(X, window_size=args.planarity_window)
    K_ests = get_K_estimates_from_args(args)

    # Trim the planarities to match number of K_ests
    planarity = planarity[int(np.floor(args.K_sample_frames / 2)):-int(np.ceil(args.K_sample_frames / 2))]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=planarity, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('Planarity')
    ax.set_ylabel('K_est')
    ax.set_title(f'Planarity ({args.planarity_window} frames) vs K estimate. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_planarity_vs_K'
            f'_w={args.planarity_window}'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_planarity_vs_HTD():
    """
    Plot planarity against HTD.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    planarity = calculate_planarity(X, window_size=args.planarity_window)
    htd = calculate_htd(X)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=planarity, y=htd, s=2, alpha=0.4)
    ax.set_xlabel('Planarity')
    ax.set_ylabel('HTD')
    ax.set_title(f'Planarity ({args.planarity_window} frames) vs HTD. Trial {args.trial}.')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_planarity_vs_HTD'
            f'_w={args.planarity_window}'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_speed_vs_K()
    plot_speed_vs_HTD()
    plot_HTD_vs_K()
    plot_speed_vs_planarity()
    plot_planarity_vs_K()
    plot_planarity_vs_HTD()
