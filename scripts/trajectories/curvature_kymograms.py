import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCES
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot helicity across a clip.')
    parser.add_argument('--trial', type=int, help='Trial by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--source', type=str, choices=M3D_SOURCES, help='Midline3D source.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--resample-points', type=int, default=-1, help='Resample the curve points.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')
    args = parser.parse_args()

    assert args.reconstruction is not None or args.trial is not None, 'Trial or reconstruction must be specified!'

    print_args(args)

    return args


def curvature_kymogram():
    """
    Plot a kymogram of the curvature over time.
    """
    args = parse_args()

    common_args = {
        'reconstruction_id': args.reconstruction,
        'trial_id': args.trial,
        'midline_source': args.source,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'resample_points': args.resample_points
    }
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    reconstruction = Reconstruction.objects.get(id=meta['reconstruction'])
    trial = reconstruction.trial
    K = np.abs(Z)
    N = len(Z)
    if args.x_label == 'time':
        ts = np.arange(args.start_frame, args.start_frame + N) / trial.fps
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Plot
    fig, axes = plt.subplots(1, figsize=(6, 3))
    ax = axes
    ax.set_title(f'Curvature\n'
                 f'Trial {trial.id} ({trial.date:%Y-%m-%d} #{trial.trial_num}).\n'
                 f'Reconstruction {reconstruction.id} ({reconstruction.source}).')
    im = ax.imshow(K.T, aspect='auto', cmap='Reds', origin='lower', extent=(ts[0], ts[-1], 0, 1))
    cax = ax.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax.transAxes)
    fig.colorbar(im, ax=ax, cax=cax)
    ht_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold')
    ax.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax.set_ylabel('$\kappa$', fontsize=12, labelpad=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_curvature_kymogram_' \
                           f't={trial.id:03d}_' \
                           f'f={meta["start_frame"]}-{meta["end_frame"]}' \
                           f'r={reconstruction.id}_{reconstruction.source}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    curvature_kymogram()
