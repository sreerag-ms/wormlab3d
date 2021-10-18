import os
from argparse import Namespace
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectory.util import generate_or_load_trajectory_cache

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

N_WORKERS = 8

show_plots = True
save_plots = False
show_animation = False
save_animation = False


def get_trajectory(args: Namespace) -> np.ndarray:
    """
    Load the full 3D trajectory and take the centre of mass at each time point.
    """
    X, meta = generate_or_load_trajectory_cache(
        args.trial,
        args.midline3d_source,
        args.midline3d_source_file,
        args.rebuild_cache
    )
    X = X.mean(axis=1)
    return X


def calculate_displacement(trajectory: np.ndarray, deltas: Union[int, List[int]]) \
        -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Evaluate the trajectory - calculate displacement.
    """
    res = {}
    return_dict = True
    if type(deltas) == int:
        return_dict = False
        deltas = [deltas]

    for delta in deltas:
        logger.info(f'Calculating displacement for delta = {delta}.')
        N = len(trajectory) - 2 * delta
        displacements = np.zeros(N)
        for i in range(N):
            displacements[i] = np.linalg.norm(trajectory[i] - trajectory[i + delta])
        rel_displacements = displacements / displacements.mean()
        res[delta] = rel_displacements

    if return_dict:
        return res
    else:
        return res[deltas[0]]


def calculate_displacement_projections(trajectory: np.ndarray, deltas: Union[int, List[int]]) \
        -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Evaluate the trajectory - calculate displacement in x, y, z.
    """
    res = {}
    return_dict = True
    if type(deltas) == int:
        return_dict = False
        deltas = [deltas]

    for delta in deltas:
        logger.info(f'Calculating angles for delta = {delta}.')
        N = len(trajectory) - 2 * delta
        displacements = np.zeros((N, 3))
        for i in range(N):
            displacements[i] = trajectory[i] - trajectory[i + delta]
        res[delta] = displacements

    if return_dict:
        return res
    else:
        return res[deltas[0]]


def calculate_transitions_and_dwells(res: Dict[int, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    ret = {}
    for delta, displacements in res.items():
        avg = displacements.mean()
        states = (displacements > avg).astype(np.int8)
        on_dwells = []
        off_dwells = []
        start_t = 0
        active_state = states[0]
        t = 1
        while t < len(states):
            state_t = states[t]
            if active_state != state_t:
                if active_state == 1:
                    on_dwells.append([start_t, t])
                else:
                    off_dwells.append([start_t, t])
                start_t = t
                active_state = state_t
            t += 1
        if t != start_t:
            if active_state == 1:
                on_dwells.append([start_t, t])
            else:
                off_dwells.append([start_t, t])

        on_dwell_durations = np.array([
            dwell[1] - dwell[0]
            for dwell in on_dwells
        ])
        off_dwell_durations = np.array([
            dwell[1] - dwell[0]
            for dwell in off_dwells
        ])

        dwells = {
            'on': on_dwells,
            'on_durations': on_dwell_durations,
            'off': off_dwells,
            'off_durations': off_dwell_durations,
        }
        ret[delta] = dwells

    return ret


def plot_displacement_results(res: Dict[int, np.ndarray], args: Namespace):
    """
    Plot histograms of the results.
    """
    fig, axes = plt.subplots(len(args.deltas), figsize=(10, 4 + 2 * len(args.deltas)), sharex=True, sharey=True)
    for i, delta in enumerate(args.deltas):
        ax = axes[i]
        ax.hist(res[delta], bins=100, density=True, facecolor='green', alpha=0.75)
        ax.set_title(f'$\Delta={delta / 25:.2f}s$')
        ax.set_ylabel('$P(d;\Delta)$')
        ax.set_xlabel('$d=\\frac{|x(t+\Delta)-x(t)|}{(\\text{mean displacement}}$')
    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            '.svg'
        )
    if show_plots:
        plt.show()


def plot_displacement_projections(res: Dict[int, np.ndarray], args: Namespace):
    """
    Plot histograms of the results in x, y, z.
    """
    fig, axes = plt.subplots(len(args.deltas), 3, figsize=(10, 4 + 2 * len(args.deltas)), sharex=False, sharey=False)
    for i, delta in enumerate(args.deltas):
        for j in range(3):
            ax = axes[i, j]
            ax.hist(res[delta][:, j], bins=100, density=True, facecolor='green', alpha=0.75)
            ax.set_title(f'$\Delta={delta / 25:.2f}s$')
            ax.set_ylabel('$P(d;\Delta)$')
            x = ['x', 'y', 'z'][j]
            ax.set_xlabel(f'$d={x}(t+\Delta)-{x}(t)$')
    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            '.svg'
        )
    if show_plots:
        plt.show()


def plot_displacement_over_time(res: Dict[int, np.ndarray], dwells: Dict[int, dict], args: Namespace):
    """
    Plot time trace of the results.
    """
    fig, axes = plt.subplots(len(args.deltas), 2, figsize=(12, 4 + 2 * len(args.deltas)))
    for i, delta in enumerate(args.deltas):
        d = res[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(d, alpha=0.75)
        ax.set_title(f'$\Delta={delta / 25:.2f}s$')
        ax.set_ylabel('$|x(t+\Delta)-x(t)|$')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = d.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(d), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(d), color='orange', alpha=0.3, zorder=-1,
                            linewidth=0)

        # Plot histogram of dwell times
        ax = axes[i, 1]
        ax.hist(dwells[delta]['on_durations'], bins=50, density=True, alpha=0.5, color='green')
        ax.hist(dwells[delta]['off_durations'], bins=50, density=True, alpha=0.5, color='orange')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_traces'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            '.svg'
        )

    if show_plots:
        plt.show()


def get_args():
    parser = build_target_arguments_parser()
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--n-frames', type=int, help='Number of frames to use.')
    parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                        help='Initial value of K for the optimiser.')
    args = parser.parse_args()
    assert not (args.trial is None and args.frame_sequence is None), 'Trial or FS must be specified.'
    return args


def displacement():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_displacement(trajectory, args.deltas)
    plot_displacement_results(displacements, args)


def displacement_projections():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_displacement_projections(trajectory, args.deltas)
    plot_displacement_projections(displacements, args)


def displacement_over_time():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_displacement(trajectory, args.deltas)
    dwells = calculate_transitions_and_dwells(displacements)
    plot_displacement_over_time(displacements, dwells, args)


if __name__ == '__main__':
    # displacement()
    displacement_projections()
    # displacement_over_time()
