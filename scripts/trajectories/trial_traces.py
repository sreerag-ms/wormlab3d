import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import fetch_annotations, calculate_speeds, calculate_htd

# tex_mode()

show_plots = True
save_plots = False
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = LOGS_PATH / f'{START_TIMESTAMP}_{method}'

    for k in ['trial', 'frames', 'src', 'smoothing_window', 'directionality', 'Knf', 'K0', 'pw']:
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
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'Knf':
            fn += f'_Knf={args.K_sample_frames}'
        elif k == 'K0':
            fn += f'_K0={args.K0}'
        elif k == 'pw':
            fn += f'_pw={args.planarity_window}'

    return fn + '.' + img_extension


def traces():
    args = get_args()
    X, meta = get_trajectory_from_args(args, return_meta=True)
    N = len(X)
    fps = 25
    ts = np.linspace(0, N / fps, N)

    logger.info('Calculating singular values.')
    pcas = get_pca_cache_from_args(args)
    ratios = pcas.explained_variance_ratio

    logger.info('Calculating K ests.')
    K_ests = get_K_estimates_from_args(args)

    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True)

    logger.info('Calculating HTDs.')
    htd = calculate_htd(X)

    logger.info('Fetching annotations.')
    tags, frame_idxs = fetch_annotations(trial_id=args.trial, frame_nums=meta['frame_nums'])

    fig, axes = plt.subplots(5, figsize=(12, 12), sharex=True, gridspec_kw={
        'height_ratios': [0.3, 1, 1, 1, 1],
        'left': 0.07,
        'right': 0.85,
        'top': 0.95,
        'bottom': 0.05,
        'hspace': 0.2
    })

    # Annotations axis
    ax = axes[0]
    ys = np.arange(len(tags))
    for i, tag in enumerate(tags):
        ts_tag = frame_idxs[i] / fps
        lbl = tag.name.replace(' ', '\n')
        ax.scatter(ts_tag, np.ones_like(ts_tag) * ys[-i - 1] / 10, label=lbl, s=5, marker='s')
    ax.legend(loc='upper left', bbox_to_anchor=(0.862, 0.97), bbox_transform=fig.transFigure, markerscale=3)
    ax.axis('off')
    ax.grid()

    # PCA axis
    ax = axes[1]
    ax.plot(ts, ratios[:, 0], label='$\lambda_1$')
    ax.plot(ts, ratios[:, 1], label='$\lambda_2$')
    ax.plot(ts, ratios[:, 2], label='$\lambda_3$')
    ax.legend()
    ax.set_ylabel('Explained variance ratio')
    ax.set_title('PCA singular values variance ratios.')
    ax.grid()

    # K estimates
    ax = axes[2]
    ax.plot(ts, K_ests)
    ax.set_ylabel('K')
    ax.set_title(f'K estimate ({args.K_sample_frames} frames, K0={args.K0}).')
    ax.grid()

    # Speeds
    ax = axes[3]
    ax.axhline(y=0, color='lightgrey')
    ax.plot(ts, speeds)
    ax.set_ylabel('Speed')
    ax.set_title('Speed.')
    ax.grid()

    # HTD
    ax = axes[4]
    ax.plot(ts, htd)
    ax.set_ylabel('HTD')
    ax.set_title('HTD.')
    ax.grid()

    ax.set_xlabel('Time (s)')

    # Make title
    frames_str_title = ''
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else -1
        frames_str_title = f' (frames={start_frame}-{end_frame})'
    title = f'Trial={args.trial}{frames_str_title}. Src={args.midline3d_source}. ' + \
            (f'Smoothing window={args.smoothing_window} frames. ' if args.smoothing_window is not None else '') + \
            (f'Pruned slowest={args.prune_slowest_ratio * 100:.1f}%. '
             if args.prune_slowest_ratio is not None else '') + \
            (f'Directionality={args.directionality}. ' if args.directionality is not None else '')
    fig.suptitle(title)

    fig.tight_layout()
    if save_plots:
        plt.savefig(make_filename('traces', args))
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    traces()
