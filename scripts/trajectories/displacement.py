import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.displacement import calculate_displacements, plot_displacement_histograms, \
    calculate_displacement_projections, plot_displacement_projections_histograms, plot_squared_displacements_over_time, \
    calculate_transitions_and_dwells_multiple_deltas

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = LOGS_PATH + '/' + START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'directionality', 'aggregation', 'deltas', 'u']:
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
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'aggregation':
            fn += f'_{args.aggregation}'
        elif k == 'deltas':
            fn += f'_d={",".join([str(d) for d in args.deltas])}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'projection':
            fn += f'_p={args.projection}'

    return fn + '.' + img_extension


def displacement():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacements(trajectory, args.deltas, args.aggregation)
    plot_displacement_histograms(displacements)
    if save_plots:
        plt.savefig(
            make_filename('histograms', args)
        )
    if show_plots:
        plt.show()


def displacement_projections():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacement_projections(trajectory, args.deltas)
    plot_displacement_projections_histograms(displacements)
    if save_plots:
        plt.savefig(
            make_filename('histograms_projections', args, excludes=['projection'])
        )
    if show_plots:
        plt.show()


def displacement_over_time():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacements(trajectory, args.deltas, args.aggregation)
    dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)
    plot_squared_displacements_over_time(displacements, dwells)
    if save_plots:
        plt.savefig(
            make_filename('traces', args)
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # displacement()
    # displacement_projections()
    displacement_over_time()
