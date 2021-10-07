import math
import os
from argparse import Namespace
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.material_parameters import MP_DEFAULT_K
from simple_worm.util import estimate_K_from_x
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Trial
from wormlab3d.toolkit.util import build_target_arguments_parser, str2bool
from wormlab3d.trajectory.util import generate_or_load_trajectory_cache, SMOOTHING_WINDOW_TYPES, smooth_trajectory

N_WORKERS = 8

show_plots = True
save_plots = True


def est_K(X: np.ndarray, i: int, n_sample_frames: int, K0: float) -> float:
    X_sample = X[i:i + n_sample_frames]
    psi = np.zeros(X_sample.shape[:-1])
    FS = FrameSequenceNumpy(x=X_sample.transpose(0, 2, 1), psi=psi, calculate_components=True)
    K_est = estimate_K_from_x(FS, K0, verbosity=0)
    logger.info(f'Frame #{i}: {K_est:.3E}')
    return K_est


def est_K_wrapper(args) -> float:
    return est_K(*args)


def get_data(args: Namespace):
    X, meta = generate_or_load_trajectory_cache(
        trial_id=args.trial,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
        rebuild_cache=args.rebuild_cache
    )

    assert meta['start_frame'] <= args.frame_num <= meta['end_frame'], \
        f'Cache not available for frame {args.frame_num}.'
    slice_start = args.frame_num - meta['start_frame']

    if args.n_frames is not None:
        assert args.frame_num + args.n_frames <= meta['end_frame'], \
            f'Cache not available for frames > {meta["end_frame"]}.'
        slice_end = slice_start + args.n_frames
    else:
        slice_end = None

    X = X.copy()
    X = X[slice_start:slice_end]

    return X


def calculate_estimates_parallel(X: np.ndarray, n_sample_frames: int, K0: float) -> List[float]:
    end_frame = X.shape[0] - n_sample_frames

    with Pool(processes=N_WORKERS) as pool:
        K_ests = pool.map(
            est_K_wrapper,
            [[X, i, n_sample_frames, K0] for i in range(end_frame)]
        )

    return K_ests


def calculate_estimates(X: np.ndarray, n_sample_frames: int, K0: float) -> List[float]:
    K_ests = []
    end_frame = X.shape[0] - n_sample_frames

    i = 0
    while i < end_frame:
        K0 = K0 if len(K_ests) == 0 else K_ests[-1]
        K_est = est_K(X, i, n_sample_frames, K0)
        K_ests.append(K_est)
        i += 1

    return K_ests


def plot_results(K_ests: List[float], args: Namespace):
    fig, ax = plt.subplots(1, figsize=(10, 8))

    start_frame = 0 if args.frame_num is None else args.frame_num
    frame_nums = np.arange(len(K_ests)) + start_frame

    ax.plot(frame_nums, K_ests)
    ax.set_title(
        f'Trial={args.trial}. '
        f'Sample duration={args.duration:.2f}s. '
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
            f'_duration={args.duration:.2f}s'
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
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--smoothing-window', type=int, help='Smooth trajectory with moving average.', default=5)
    args = parser.parse_args()
    assert args.trial is not None, 'Trial must be specified.'
    assert args.duration is not None, 'Duration must be specified.'
    if args.frame_num is None:
        args.frame_num = 0

    X = get_data(args)
    trial = Trial.objects.get(id=args.trial)
    n_sample_frames = math.ceil(args.duration * trial.fps)
    if args.n_frames is not None:
        assert args.n_frames > n_sample_frames, 'Number of frames to process must be > number of frames per sample.'

    # plot_xyz(X)  # debug

    X_smoothed = smooth_trajectory(X, window_len=args.smoothing_window)
    if N_WORKERS > 1:
        K_ests = calculate_estimates_parallel(X_smoothed, n_sample_frames, args.K0)
    else:
        K_ests = calculate_estimates(X_smoothed, n_sample_frames, args.K0)
    plot_results(K_ests, args)


if __name__ == '__main__':
    estimate_K()
