import os

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args

# tex_mode()

show_plots = True
save_plots = False


def plot_trajectory_2d():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    projections = ['xy', 'yz', 'xz']

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, u in enumerate(us):
        args.trajectory_point = u
        for j, p in enumerate(projections):
            args.projection = p
            trajectory = get_trajectory_from_args(args)

            axes[i, j].set_title(f'u={u}, p={p}')
            axes[i, j].plot(trajectory[:, 0], trajectory[:, 1])

    frames_str = ''
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else -1
        frames_str = f'_f={start_frame}-{end_frame}'

    fig.suptitle(f'trial={args.trial}{frames_str}_{args.midline3d_source}')
    fig.tight_layout()

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory'
            f'_trial={args.trial}' +
            frames_str +
            f'_{args.midline3d_source}'
            f'_u={",".join([str(u) for u in us])}'
            '.svg'
        )
    if show_plots:
        plt.show()


def plot_trajectory_2d_wt3d_vs_reconst():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    projections = ['xy', 'yz', 'xz']
    srcs = ['reconst', 'WT3D']
    src_files = ['039_shelf-2.hdf5', None]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for k, src in enumerate(srcs):
        args.midline3d_source = src
        args.midline3d_source_file = src_files[k]
        for i, u in enumerate(us):
            args.trajectory_point = u
            for j, p in enumerate(projections):
                args.projection = p
                trajectory = get_trajectory_from_args(args)
                axes[i, j].set_title(f'u={u}, p={p}')
                axes[i, j].plot(trajectory[:, 0], trajectory[:, 1], label=src, linewidth=0.6, alpha=0.7)
    axes[0, 0].legend()

    frames_str = ''
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else -1
        frames_str = f'_f={start_frame}-{end_frame}'

    fig.suptitle(f'trial={args.trial}{frames_str} WT3D vs reconst')
    fig.tight_layout()

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory'
            f'_trial={args.trial}' +
            frames_str +
            f'_WT3D_vs_reconst'
            f'_u={",".join([str(u) for u in us])}'
            '.svg'
        )
    if show_plots:
        plt.show()


def plot_brownian_trajectory_2d():
    D = 100
    n_steps = 1000
    total_time = 1
    p = BrownianParticle(D=D)
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
    projections = ['xy', 'yz', 'xz']

    fig, axes = plt.subplots(3, figsize=(6, 10))

    for i, p in enumerate(projections):
        if p == 'xy':
            X_ = np.delete(X, 2, 1)
        elif p == 'yz':
            X_ = np.delete(X, 0, 1)
        elif p == 'xz':
            X_ = np.delete(X, 1, 1)
        axes[i].set_title(p)
        axes[i].plot(X_[:, 0], X_[:, 1])

    fig.suptitle(f'Brownian trajectory. D={D}, n_steps={n_steps}, total_time={total_time}.')
    fig.tight_layout()

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_brownian_particle'
            f'_D={D}_n={n_steps}_T={total_time}'
            '.svg'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    plot_trajectory_2d()
    # plot_trajectory_2d_wt3d_vs_reconst()
    # plot_brownian_trajectory_2d()
