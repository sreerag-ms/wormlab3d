import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.dataset import Dataset
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

show_plots = True
save_plots = True
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot an eigenworm basis.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--eigenworms', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1,2,3,4', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')
    args = parser.parse_args()

    assert args.dataset is not None, \
        'This script must be run with the --dataset=ID argument defined.'

    print_args(args)

    return args


def eigenworm_modulation():
    """
    Show how eigenworms vary across concentrations/reconstructions.
    """
    args = parse_args()
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    metas = ds.metas
    lambdas = {}

    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        dataset_id=args.dataset,
        n_components=args.n_components,
        regenerate=False
    )

    # Group by concentration and then by reconstruction
    for i, rid in enumerate(metas['reconstruction']):
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in lambdas:
            lambdas[c] = {}
        Z, _ = get_trajectory(reconstruction_id=rid, natural_frame=True)
        X_ew = ew.transform(Z)
        lambdas[c][rid] = np.abs(X_ew)

    # Sort by concentration
    lambdas = {k: v for k, v in sorted(list(lambdas.items()))}

    # Determine positions
    concs = [float(k) for k in lambdas.keys()]
    step = 0.25
    ticks = np.arange(min(concs), max(concs) + step, step)

    # Prepare output
    out = np.zeros((len(concs), ew.n_components))
    for i, (c, r_vals) in enumerate(lambdas.items()):
        lambdas_c = np.concatenate([v for v in r_vals.values()])
        out[i] = lambdas_c.sum(axis=0) / lambdas_c.sum()

    # Make plots
    fig, axes = plt.subplots(1, figsize=(8, 6), sharex=False, sharey=False)
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, len(args.plot_components)))

    ax = axes
    ax.set_title('Eigenworms contributions')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Relative contribution')
    ax.axhline(y=0, linestyle=':', color='lightgrey', zorder=-1)

    for i, idx in enumerate(args.plot_components):
        ax.plot(
            concs,
            out[:, idx],
            linewidth=2,
            marker='x',
            markersize=10,
            markeredgewidth=4,
            alpha=0.8,
            label=f'$\lambda_{i}$',
            color=colours[i]
        )

    ax.legend()

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_ew_modulation' \
                           f'_ds={ds.id}' \
                           f'_ew={ew.id}' \
                           f'_plot={",".join([str(c) for c in args.plot_components])}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    eigenworm_modulation()
