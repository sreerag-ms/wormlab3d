import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from simple_worm.plot3d import MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.helicities import calculate_helicities, plot_helicities
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

# tex_mode()

show_plots = True
save_plots = False
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot helicity across a clip.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--resample-points', type=int, default=-1, help='Resample the curve points.')
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
        ts = np.arange(args.start_frame, args.start_frame + N) / trial.fps
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


def helicity_kymogram(x_label: str = 'time'):
    """
    Plot a kymogram of the helicity over time.
    """
    args = parse_args()
    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25,
        'resample_points': args.resample_points
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    N = len(Z)
    if x_label == 'time':
        ts = np.arange(args.start_frame, args.start_frame + N) / trial.fps
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Calculate helicity
    logger.info('Calculating helicities.')
    kappa = np.abs(Z)
    psi = np.unwrap(np.angle(Z), axis=1)
    tau = np.gradient(psi, axis=1)
    H = kappa * tau

    # Plot
    # fig, axes = plt.subplots(1, figsize=(12, 8))
    fig, axes = plt.subplots(1, figsize=(6, 3))

    # Torsion
    ax = axes
    ax.set_title(f'Helicity\n'
                 f'Trial {trial.id} ({trial.date:%Y-%m-%d} #{trial.trial_num}).\n'
                 f'Reconstruction {reconstruction.id} ({reconstruction.source}).')
    im = ax.imshow(H.T, aspect='auto', cmap='PRGn', origin='lower', extent=(0, N / trial.fps, 0, 1),
                   # , extent=(ts[0], ts[-1], 0, 1),
                   norm=MidpointNormalize(midpoint=0))
    cax = ax.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax.transAxes)
    fig.colorbar(im, ax=ax, cax=cax)
    ht_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold')
    ax.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax.set_ylabel('$\zeta$', fontsize=12, labelpad=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    if x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_helicity_kymogram_' \
                           f'r={reconstruction.id}_' \
                           f'f={meta["start_frame"]}-{meta["end_frame"]}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def torsion_kymogram(x_label: str = 'time', threshold: float = np.pi / 2):
    """
    Plot a kymogram of the torsion over time.
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
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    N = len(Z)
    if x_label == 'time':
        ts = np.arange(args.start_frame, args.start_frame + N) / trial.fps
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Calculate torsions
    psi = np.unwrap(np.angle(Z), axis=1)
    torsion = np.gradient(psi, axis=1)

    # Threshold to ignore torsion at low-curvature. k = 1/r. k = 2pi => circle.
    if threshold is not None:
        kappa = np.abs(Z)
        excludes = kappa < threshold
        torsion[excludes] = 0

    # Plot
    fig, axes = plt.subplots(1, figsize=(12, 8))

    # Torsion
    ax = axes
    ax.set_title(f'Torsion\n'
                 f'Trial {trial.id} ({trial.date:%Y-%m-%d} #{trial.trial_num}).\n'
                 f'Reconstruction {reconstruction.id} ({reconstruction.source}).')
    im = ax.imshow(torsion.T, aspect='auto', cmap='PRGn', origin='lower', extent=(ts[0], ts[-1], 0, 1),
                   norm=MidpointNormalize(midpoint=0))
    cax = ax.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax.transAxes)
    fig.colorbar(im, ax=ax, cax=cax)
    ht_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold')
    ax.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax.set_ylabel('$\\tau$', fontsize=12, labelpad=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])

    if x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_torsion_kymogram_' \
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
    # helicity_trace(x_label='time')
    helicity_kymogram(x_label='time')
    # torsion_kymogram(x_label='time')
