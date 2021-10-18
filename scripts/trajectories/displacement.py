import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.toolkit.plot_utils import tex_mode
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectories.displacement import calculate_squared_displacements, plot_squared_displacement_histograms, \
    calculate_displacement_projections, plot_displacement_projections_histograms, plot_squared_displacements_over_time, \
    calculate_transitions_and_dwells_multiple_deltas
from wormlab3d.trajectories.util import generate_or_load_trajectory_cache

tex_mode()

show_plots = True
save_plots = True


def get_args():
    parser = build_target_arguments_parser()
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                        help='Time lag sizes.')
    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 (default) to use centre of mass.')
    args = parser.parse_args()
    assert args.trial is not None, 'Trial must be specified.'
    assert args.trajectory_point == -1 or 0 <= args.trajectory_point <= 1, 'trajectory-point must be -1 for centre of mass or between 0 and 1.'
    return args


def get_trajectory(args: Namespace) -> np.ndarray:
    """
    Load the full 3D trajectory and then either take a slice at the required point or use the centre of mass.
    """
    X, meta = generate_or_load_trajectory_cache(
        args.trial,
        args.midline3d_source,
        args.midline3d_source_file,
        args.rebuild_cache
    )
    N = X.shape[1]

    if args.trajectory_point == -1:
        X = X.mean(axis=1)
    else:
        u = round(args.trajectory_point * N)
        assert 0 <= u <= N
        X = X[:, u]

    return X


def displacement():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_squared_displacements(trajectory, args.deltas)
    plot_squared_displacement_histograms(displacements)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            f'_u={args.trajectory_point}' +
            '.svg'
        )
    if show_plots:
        plt.show()


def displacement_projections():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_displacement_projections(trajectory, args.deltas)
    plot_displacement_projections_histograms(displacements)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            f'_u={args.trajectory_point}' +
            '.svg'
        )
    if show_plots:
        plt.show()


def displacement_over_time():
    args = get_args()
    trajectory = get_trajectory(args)
    displacements = calculate_squared_displacements(trajectory, args.deltas)
    dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)
    plot_squared_displacements_over_time(displacements, dwells)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_traces'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}' +
            f'_u={args.trajectory_point}' +
            '.svg'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    displacement()
    displacement_projections()
    displacement_over_time()
