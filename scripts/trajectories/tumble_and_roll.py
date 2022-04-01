import os
from argparse import Namespace
from multiprocessing import Pool
from typing import List, Dict, Union

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np

from simple_worm.plot3d import interactive
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.cache import get_trajectory_from_args

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})

N_WORKERS = 8

show_plots = False
save_plots = False
show_animation = True
save_animation = True


def evaluate(trajectory: np.ndarray, deltas: Union[int, List[int]]) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Evaluate the trajectory - calculate angles for all deltas.
    """
    res = {}
    L = len(trajectory)
    return_dict = True
    if type(deltas) == int:
        return_dict = False
        deltas = [deltas]

    for delta in deltas:
        logger.info(f'Calculating angles for delta = {delta}.')
        s = np.zeros(L - 2 * delta)
        for i in range(L - 2 * delta):
            v1 = trajectory[i + delta] - trajectory[i]
            v2 = trajectory[i + 2 * delta] - trajectory[i + delta]
            angle = calculate_angle(v1, v2)
            s[i] = angle
        res[delta] = s

    if return_dict:
        return res
    else:
        return res[deltas[0]]


def plot_results(res: List[Dict[int, np.ndarray]], args: Namespace):
    """
    Plot histograms of the results.
    """
    fig, axes = plt.subplots(len(res), len(args.deltas), figsize=(4 + len(res), 4 + len(args.deltas)))
    for i, delta in enumerate(args.deltas):
        ax = axes[i]
        ax.hist(res[delta], bins=100, density=True, facecolor='green', alpha=0.75)
        ax.set_title(f'$\Delta={delta}$')
        ax.set_ylabel('$P(\\theta;\Delta)$')
        ax.set_xlabel('$\\theta$')
    suptitle = f'u={args.trajectory_point}'
    if args.projection is not None:
        suptitle += f', projection={args.projection}'
    plt.suptitle(suptitle)
    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_u={args.trajectory_point}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            (f'_projection={args.projection}' if args.projection is not None else '') +
            '.svg'
        )
    if show_plots:
        plt.show()


def evaluate_wrapper(args):
    return evaluate(*args)


def animate_results(args: Namespace):
    delta_init = 5
    max_delta = 1000
    delta_step = 5
    n_bins = 100
    fps = 5
    # projections = ['xy', 'yz', 'zx']
    projections = [None]  # 'xy', 'yz', 'zx']
    us = [0.2, 0.5, 0.8]
    # us = [0.5]

    hist_colours = {
        0.2: 'green',
        0.5: 'red',
        0.8: 'blue'
    }

    interactive()

    # Get deltas
    deltas = []
    delta = delta_init
    while delta <= max_delta:
        deltas.append(delta)
        delta += delta_step

    # Calculate angles
    results = {}
    for p in projections:
        # Get trajectory data for the different projections
        args.projection = p
        results[p] = {}

        for u in us:
            logger.info(f'Calculating angles for projection={p}, u={u}.')

            # Get trajectory data for the different body coordinates
            results[p][u] = {}
            args.trajectory_point = u
            trajectory = get_trajectory_from_args(args)

            # Calculate the angles for all deltas
            with Pool(processes=N_WORKERS) as pool:
                res = pool.map(
                    evaluate_wrapper,
                    [[trajectory, delta] for delta in deltas]
                )

            for i, delta in enumerate(deltas):
                assert not np.isnan(res[i]).any()
                results[p][u][delta] = res[i]

    # Set up plot
    fig, axes = plt.subplots(len(projections), figsize=(10, 12), squeeze=False)
    hists = {}
    for i in range(len(projections)):
        p = projections[i]
        hists[p] = {}
        ax = axes[i, 0]
        ax.set_title(p)
        ax.set_ylabel('$P(\\theta;\Delta)$')
        ax.set_xlabel('$\\theta$')
        for u in us:
            _, _, hist = ax.hist(
                results[p][u][delta_init],
                bins=n_bins,
                density=True,
                facecolor=hist_colours[u],
                alpha=0.3
            )
            hists[p][u] = hist
        if p is None:
            ax.set_xticks([0, np.pi])
            ax.set_xticklabels(['$0$', '$\pi$'])
            ax.set_ylim(bottom=0, top=1)
        else:
            ax.set_xticks([-np.pi, np.pi])
            ax.set_xticklabels(['$-\pi$', '$\pi$'])
            ax.set_ylim(bottom=0, top=0.6)

    title = fig.suptitle(f'$\Delta={delta_init} ({delta_init / 25:.2f}s)$')

    def update(k):
        delta = deltas[k]
        for p in projections:
            for u in us:
                n, _ = np.histogram(results[p][u][delta], n_bins, density=True)
                for count, rect in zip(n, hists[p][u].patches):
                    rect.set_height(count)
        title.set_text(f'$\Delta={delta} \quad ({delta / 25:.2f}s)$')
        return ()

    fig.tight_layout()

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=len(deltas),
        blit=True,
        interval=500
    )

    if save_animation:
        os.makedirs(LOGS_PATH, exist_ok=True)
        fn = f'trial={args.trial}' \
             f'_{args.midline3d_source}' \
             f'_d={delta_init}-{max_delta}_step={delta_step}_fps={fps}' \
             f'_u={",".join([str(u) for u in us])}'
        metadata = dict(
            title=fn,
            artist='WormLab Leeds'
        )
        save_path = LOGS_PATH + '/' + START_TIMESTAMP + '_' + fn + '.mp4'
        logger.info(f'Saving animation to {save_path}.')
        ani.save(save_path, writer='ffmpeg', fps=fps, metadata=metadata)

    if show_animation:
        plt.show()


def tumble_roll():
    """
    """
    parser = build_target_arguments_parser()
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                        help='Time lag sizes.')
    parser.add_argument('--trajectory-point', type=float, default=0.5, help='Number between 0 (head) and 1 (tail).')
    parser.add_argument('--n-frames', type=int, help='Number of frames to use.')
    parser.add_argument('--projection', type=str, choices=['xy', 'yz', 'zx'],
                        help='Use only two of the three coordinates.')
    args = parser.parse_args()
    assert not (args.trial is None and args.FS is None), 'Trial or FS must be specified.'
    assert 0 <= args.trajectory_point <= 1, 'Trajectory sample point should be between 0 and 1.'

    if show_animation or save_animation:
        animate_results(args)

    if show_plots or save_plots:
        for deltas in [
            # [1,5,10,20,50],
            # [100,200,500,1000,5000],
            # [1000,2000,3000,5000,10000],
            [10, 100, 500, 1000, 2000],
        ]:
            for projection in ['xy', 'yz', 'zx']:
                args.deltas = deltas
                args.projection = projection
                trajectory = get_trajectory_from_args(args)
                res = evaluate(trajectory, args.deltas)
                plot_results(res, args)

        # trajectory = get_trajectory(args)
        # res = evaluate(trajectory, args.deltas)
        # plot_results(res, args)


if __name__ == '__main__':
    tumble_roll()
