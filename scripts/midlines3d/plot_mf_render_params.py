import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Dataset
from wormlab3d.data.model.dataset import DatasetMidline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'

RENDER_PARAMETER_KEYS = [
    'sigmas',
    'intensities',
    'exponents',
]

prop_cycle = plt.rcParams['axes.prop_cycle']
default_colours = prop_cycle.by_key()['color']
colours = {c: default_colours[c] for c in range(3)}
colours['highlight'] = 'darkviolet'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to plot the MF render parameters.')

    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--stats-window', type=int, default=5, help='Averaging window for the values.')
    parser.add_argument('--x-label', type=str, default='frame', help='Label x-axis with time or frame number.')
    parser.add_argument('--highlight-frames', type=lambda s: [int(item) for item in s.split(',')],
                        help='Highlight these frame numbers.')
    args = parser.parse_args()

    print_args(args)

    return args


def _rolling_stats(vals: List[np.ndarray], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rolling mean and standard deviations.
    """
    pl = np.ones(int((window_size - 1) / 2)) * vals[0]
    pr = np.ones(window_size - len(pl) - 1) * vals[-1]
    errs_padded = np.r_[pl, vals, pr]
    x = sliding_window_view(errs_padded, window_size)
    means = x.mean(axis=1)
    stds = x.std(axis=1)

    return means, stds


def plot_rec_render_params(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
):
    """
    Plot the render parameters for a reconstruction.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    logger.info(f'Plotting render parameters for trial {rec.trial.id} (rec={rec.id}).')

    ts = TrialState(
        reconstruction=rec,
        start_frame=rec.start_frame_valid,
        end_frame=rec.end_frame_valid + 1
    )

    N = len(ts) - 1
    if args.x_label == 'time':
        x = np.linspace(0, N / ts.trial.fps, N)
    else:
        x = np.arange(N) + ts.start_frame

    fig, axes = plt.subplots(3, figsize=(8, 10))
    for i, k in enumerate(RENDER_PARAMETER_KEYS):
        p = ts.get(k)[:, 0]
        cam_p = ts.get('camera_' + k)
        ax = axes[i]
        ax.set_title(k.capitalize())

        means, stds = _rolling_stats(p, args.stats_window)
        ax.plot(x, means, color='black', linewidth=2, linestyle='--', alpha=0.8)

        for c in range(3):
            means, stds = _rolling_stats(p * cam_p[:, c], args.stats_window)
            ax.plot(x, means, color=colours[c], label=f'Cam{c}', linewidth=1, alpha=0.8)
            ax.fill_between(x, means - 2 * stds, means + 2 * stds, color=colours[c],
                            alpha=0.2, linewidth=0.5)

        ax.legend()
        if args.x_label == 'time':
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('Frame #')

    fig.tight_layout()

    if save_plots:
        if save_dir is None:
            save_dir = LOGS_PATH
            path = save_dir / (f'{START_TIMESTAMP}'
                               f'_trial={rec.trial.id:03d}'
                               f'_{rec.id}'
                               f'_sw={args.stats_window}'
                               f'.{img_extension}')
        else:
            path = save_dir / (f'trial={rec.trial.id:03d}'
                               f'_{rec.id}.{img_extension}')

        if save_plots:
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_rec_render_params_simple(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
):
    """
    Plot the render parameters for a reconstruction.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    logger.info(f'Plotting render parameters for trial {rec.trial.id} (rec={rec.id}).')

    # Get frame range
    start_frame = args.start_frame if args.start_frame is not None else rec.start_frame_valid
    end_frame = args.end_frame if args.end_frame is not None else rec.end_frame_valid
    assert rec.start_frame_valid <= start_frame < end_frame <= rec.end_frame_valid, 'Invalid frame range!'

    ts = TrialState(
        reconstruction=rec,
        start_frame=start_frame,
        end_frame=end_frame + 1
    )

    N = len(ts) - 1
    if args.x_label == 'time':
        x = np.linspace(0, N / ts.trial.fps, N)
    else:
        x = np.arange(N)  # + ts.start_frame

    # Make plots
    plt.rc('axes', labelsize=6)  # fontsize of the X label
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=6)  # fontsize of the legend
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=2, size=2)

    fig, axes = plt.subplots(3, figsize=(3.6, 2.4), sharex=True, gridspec_kw={
        'left': 0.08,
        'right': 0.97,
        'top': 0.9,
        'bottom': 0.12,
        'hspace': 0.12,
    })

    for i, k in enumerate(RENDER_PARAMETER_KEYS):
        p = ts.get(k)[:, 0]
        cam_p = ts.get('camera_' + k)
        ax = axes[i]

        markers = []

        for c in range(3):
            means, stds = _rolling_stats(p * cam_p[:, c], args.stats_window)
            ax.plot(x, means, color=colours[c], label=f'Camera {c}', linewidth=1, alpha=0.8)
            ax.fill_between(x, means - 2 * stds, means + 2 * stds, color=colours[c],
                            alpha=0.2, linewidth=0.5)

            for frame_num in args.highlight_frames:
                n_adj = frame_num - ts.start_frame
                if n_adj < x[0] or n_adj > x[-1]:
                    continue
                markers.append((n_adj, means[n_adj]))

        # Add highlighted-frames markers
        ylim = ax.get_ylim()
        for frame_num in args.highlight_frames:
            n_adj = frame_num - ts.start_frame
            ax.vlines(x=n_adj, ymin=ylim[0] * 0.8, ymax=ylim[1] * 1.2, linestyle='-', linewidth=1,
                      alpha=0.6, color=colours['highlight'], zorder=3)
        if len(markers) > 0:
            markers = np.array(markers)
            ax.scatter(x=markers[:, 0], y=markers[:, 1], marker='o', s=50, alpha=0.6,
                       facecolors='none', edgecolors=colours['highlight'], linewidth=1, zorder=4)
        ax.set_ylim(ylim)

        if i == 0:
            ax.set_yticks([0.04, 0.06, 0.08])
        elif i == 1:
            ax.set_yticks([0.5, 0.7, 0.9])
        elif i == 2:
            ax.set_yticks([1, 1.75, 2.5])

        ax.set_xlim(left=0, right=N)
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        ax.grid()

        if i == 0:
            legend = ax.legend(loc='lower center', mode=None, ncol=3, bbox_to_anchor=(0.5, 1),
                               bbox_transform=ax.transAxes)
            for line in legend.get_lines():
                line.set_linewidth(2)

    if args.x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    if save_plots:
        if save_dir is None:
            save_dir = LOGS_PATH
            path = save_dir / (f'{START_TIMESTAMP}'
                               f'_trial={rec.trial.id:03d}'
                               f'_{rec.id}'
                               f'_f={ts.start_frame}-{ts.end_frame - 1}'
                               f'_sw={args.stats_window}'
                               f'.{img_extension}')
        else:
            path = save_dir / (f'trial={rec.trial.id:03d}'
                               f'_{rec.id}.{img_extension}')

        if save_plots:
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_ds_render_params():
    """
    Plot the render parameters for all reconstructions in a dataset.
    """
    args = get_args()
    assert args.dataset is not None, 'This script requires setting --dataset=id.'
    ds = Dataset.objects.get(id=args.dataset)
    assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'

    # Make save dir
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_ds={ds.id}_sw={args.stats_window}'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Loop over reconstructions
    logger.info(f'Plotting render parameters for all reconstructions in dataset {ds.id}.')
    for i, rec in enumerate(ds.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(ds.reconstructions)}.')
        args.reconstruction = rec.id
        plot_rec_render_params(args, save_dir)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    # plot_rec_render_params()
    plot_rec_render_params_simple()
    # plot_ds_render_params()
