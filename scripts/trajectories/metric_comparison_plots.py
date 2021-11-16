import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_planarity_from_args
from wormlab3d.trajectories.util import calculate_speeds, calculate_htd

show_plots = True
save_plots = False
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = LOGS_PATH + '/' + START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'u', 'smoothing_window', 'directionality', 'projection']:
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
        elif k == 'u' and args.trajectory_point is not None:
            fn += f'_u={args.trajectory_point}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'projection' and args.projection is not None:
            fn += f'_p={args.projection}'

    # Add K-estimation params
    if method in ['speed_vs_K', 'HTD_vs_K', 'planarity_vs_K']:
        fn += f'_Knf={args.K_sample_frames}'
        fn += f'_K0={args.K0}'

    # Add planarity window parameter
    if method in ['speed_vs_planarity', 'planarity_vs_K', 'planarity_vs_HTD']:
        fn += f'_pw={args.planarity_window}'

    return fn + '.' + img_extension


def make_title(method: str, args: Namespace):
    speed_title = 'Speed'
    K_est_title = f'K estimate ({args.K_sample_frames} frames, K0={args.K0})'
    htd_title = 'HTD'
    planarity_title = f'Planarity ({args.planarity_window} frames)'

    if method == 'speed_vs_K':
        t = f'{speed_title} vs {K_est_title}.'
    elif method == 'speed_vs_HTD':
        t = f'{speed_title} vs {htd_title}.'
    elif method == 'HTD_vs_K':
        t = f'{htd_title} vs {K_est_title}.'
    elif method == 'speed_vs_planarity':
        t = f'{speed_title} vs {planarity_title}.'
    elif method == 'planarity_vs_K':
        t = f'{planarity_title} vs {K_est_title}.'
    elif method == 'planarity_vs_HTD':
        t = f'{planarity_title} vs {htd_title}.'
    else:
        raise RuntimeError(f'Unrecognised method: {method}.')

    t += f' Trial {args.trial}.'

    if args.smoothing_window is not None:
        t += f'\nSmoothing window = {args.smoothing_window} frames.'

    return t


def plot_speed_vs_K():
    """
    Plot speed against K estimate.
    """
    args = get_args()
    K_ests = get_K_estimates_from_args(args)
    X = get_trajectory_from_args(args)
    speeds = calculate_speeds(X, signed=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=speeds, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('Speed')
    ax.set_ylabel('K_est')
    ax.set_title(make_title('speed_vs_K', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('speed_vs_K', args))
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
    ax.set_title(make_title('speed_vs_HTD', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('speed_vs_HTD', args))
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

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=htd, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('HTD')
    ax.set_ylabel('K_est')
    ax.set_title(make_title('HTD_vs_K', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('HTD_vs_K', args))
    if show_plots:
        plt.show()


def plot_speed_vs_planarity():
    """
    Plot speeds against planarity.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    speeds = calculate_speeds(X, signed=True)
    planarity = get_planarity_from_args(args)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=speeds, y=planarity, s=2, alpha=0.4)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Planarity')
    ax.set_title(make_title('speed_vs_planarity', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('speed_vs_planarity', args))
    if show_plots:
        plt.show()


def plot_planarity_vs_K():
    """
    Plot planarity against K estimate.
    """
    args = get_args()
    planarity = get_planarity_from_args(args)
    K_ests = get_K_estimates_from_args(args)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=planarity, y=K_ests, s=2, alpha=0.4)
    ax.set_xlabel('Planarity')
    ax.set_ylabel('K_est')
    ax.set_title(make_title('planarity_vs_K', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('planarity_vs_K', args))
    if show_plots:
        plt.show()


def plot_planarity_vs_HTD():
    """
    Plot planarity against HTD.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    planarity = get_planarity_from_args(args)
    htd = calculate_htd(X)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=planarity, y=htd, s=2, alpha=0.4)
    ax.set_xlabel('Planarity')
    ax.set_ylabel('HTD')
    ax.set_title(make_title('planarity_vs_HTD', args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename('planarity_vs_HTD', args))
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_speed_vs_K()
    plot_speed_vs_HTD()
    plot_HTD_vs_K()
    plot_speed_vs_planarity()
    plot_planarity_vs_K()
    plot_planarity_vs_HTD()
