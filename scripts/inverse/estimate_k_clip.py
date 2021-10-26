import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from simple_worm.material_parameters import MP_DEFAULT_K
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.simple_worm.estimate_k import generate_or_load_K_estimates_cache
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectories.util import SMOOTHING_WINDOW_TYPES, smooth_trajectory

show_plots = True
save_plots = True


def get_data(args: Namespace) -> np.ndarray:
    if args.n_frames is not None:
        end_frame = args.frame_num + args.n_frames
    else:
        end_frame = None

    K_ests, meta = generate_or_load_K_estimates_cache(
        trial_id=args.trial,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
        start_frame=args.frame_num,
        end_frame=end_frame,
        smoothing_window=args.smoothing_window,
        n_sample_frames=args.K_sample_frames,
        K0=args.K0,
        rebuild_cache=args.rebuild_cache
    )

    return K_ests


def plot_results(K_ests: List[float], args: Namespace):
    fig, ax = plt.subplots(1, figsize=(10, 8))

    start_frame = 0 if args.frame_num is None else args.frame_num
    frame_nums = np.arange(len(K_ests)) + start_frame

    ax.plot(frame_nums, K_ests)
    ax.set_title(
        f'Trial={args.trial}. '
        f'Sample duration={args.K_sample_frames} frames. '
        f'K0={args.K0}. '
        f'Smoothing window={args.smoothing_window} frames.'
    )
    ax.set_ylabel('K_est')
    ax.set_xlabel('frame')

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trial={args.trial}'
            f'_frames={frame_nums[0]}-{frame_nums[-1]}'
            f'_sample={args.K_sample_frames:.2f}s'
            f'_K0={args.K0}'
            f'_w={args.smoothing_window}'
            '.svg'
        )
    if show_plots:
        plt.show()


def plot_xyz(X: np.ndarray):
    fig, axes = plt.subplots(3, figsize=(10, 8))
    window_len = 5
    u = 0
    max_frame = 100

    for i in range(3):
        ax = axes[i]
        ax.plot(X[:max_frame, u, i])
        for w in SMOOTHING_WINDOW_TYPES:
            X_s = smooth_trajectory(X, window_len, w)
            ax.plot(X_s[:max_frame, u, i], label=w)
        ax.legend()
    plt.show()


def estimate_K():
    """
    Estimate the K parameter across a clip for various settings.
    """
    parser = build_target_arguments_parser()
    parser.add_argument('--n-frames', type=int, help='Number of frames from frame_num to process.')
    parser.add_argument('--K0', type=float, default=MP_DEFAULT_K, help='Initial value of K for the optimiser.')
    parser.add_argument('--rebuild-cache', type=str2bool, default=False, help='Rebuild the trajectory cache.')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Smooth trajectory with moving average.')
    parser.add_argument('--K-sample-frames', type=int, default=5,
                        help='Number of frames from which to calculate the K estimate.')
    args = parser.parse_args()
    assert args.trial is not None, 'Trial must be specified.'
    if args.frame_num is None:
        args.frame_num = 0
    K_ests = get_data(args)
    plot_results(K_ests, args)


if __name__ == '__main__':
    estimate_K()
