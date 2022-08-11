import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.helicities import calculate_helicities, plot_helicities
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot helicity across a clip.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def helicity_trace(x_label: str = 'time'):
    """
    Plot a trace of the helicity over time.
    """
    args = parse_args()

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial
    X, meta = get_trajectory(**common_args)
    N = len(X)
    if x_label == 'time':
        ts = np.linspace(0, N / trial.fps, N)
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Calculate helicity
    logger.info('Calculating helicities.')
    H = calculate_helicities(X)

    # Plot
    fig, axes = plt.subplots(1, figsize=(12, 8))

    # Helicities
    ax = axes
    ax.set_title(f'Helicity\n'
                 f'Trial {trial.id} ({trial.date:%Y-%m-%d} #{trial.trial_num}).\n'
                 f'Reconstruction {reconstruction.id} ({reconstruction.source}).')
    ax.axhline(y=0, color='darkgrey')
    plot_helicities(
        ax=ax,
        helicities=H,
        xs=ts,
    )
    h_lim = np.abs(H).max() * 1.1
    ax.set_ylim(top=h_lim, bottom=-h_lim)

    label_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold', fontfamily='Symbol')
    ax.text(-0.02, 0.96, '↻', verticalalignment='top', **label_args)
    ax.text(-0.02, 0.03, '↺', verticalalignment='bottom', **label_args)
    ax.set_yticks([0, ])
    ax.set_yticklabels([])

    if x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_helicity_trace_' \
                           f'r={reconstruction.id}_' \
                           f'f={meta["start_frame"]}-{meta["end_frame"]}' \
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
    helicity_trace(x_label='frames')
