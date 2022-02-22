import os
from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Dataset, Reconstruction

show_plots = True
save_plots = True
img_extension = 'png'


def _get_dataset() -> Dataset:
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    args = parser.parse_args()
    if args.dataset is None:
        raise RuntimeError('This script must be run with the --dataset=ID argument defined.')
    ds = Dataset.objects.get(id=args.dataset)

    return ds


def plot_non_planarity_with_splits():
    """
    Multiple plots of posture planarity split by user, concentration and source.
    """
    split_keys = ['user', 'concentration', 'source']
    logger.info('Fetching dataset.')
    ds = _get_dataset()
    nonp = ds.get_nonplanarities()
    metas = ds.metas

    # Make plots
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=False, sharey=False)

    # Plot histogram of all results
    logger.info('Plotting.')
    for i in range(2):
        ax = axes[0, i]
        ax.set_title('Non-planarity of postures - all.')
        ax.hist(nonp, bins=100, density=True)
        ax.set_xlabel('Non-Planarity')
        if i == 0:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('log(Density)')
            ax.set_yscale('log')
    axes[0, 2].axis('off')

    # Plot overlapping histograms of splits
    for i in range(2):
        for j, ucs in enumerate(split_keys):
            ax = axes[j + 1, i]
            ax.set_title(f'Split by {ucs}.')
            for k, idxs in metas[ucs].items():
                ax.hist(nonp[idxs], bins=100, density=True, alpha=0.5, label=k)

            ax.set_xlabel('Non-Planarity')
            if i == 0:
                ax.set_ylabel('Density')
            else:
                ax.set_ylabel('log(Density)')
                ax.set_yscale('log')

            ax.legend()

    # Violin plots
    for j, ucs in enumerate(split_keys):
        ax = axes[j + 1, 2]
        ax.set_title(f'Split by {ucs}.')

        dists = OrderedDict()
        for k, idxs in metas[ucs].items():
            dists[k] = nonp[idxs]

        # sort
        dists = {k: v for k, v in sorted(list(dists.items()))}
        ax.set_xticks(list(range(1, len(dists) + 1)))
        ax.set_xticklabels(dists.keys())
        ax.set_xlabel(ucs)
        ax.set_ylabel('Non-Planarity')

        parts = ax.violinplot(dists.values(), widths=1, showmeans=True, showmedians=True)
        parts['cmedians'].set_color('green')
        parts['cmedians'].set_alpha(0.7)
        parts['cmedians'].set_linestyle(':')
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_alpha(0.7)
        parts['cmeans'].set_linestyle('--')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_nonp_splits_ds={ds.id}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def plot_non_planarity():
    """
    Posture planarity split by concentration and trials.
    """
    logger.info('Fetching dataset.')
    ds = _get_dataset()
    nonp = ds.get_nonplanarities()
    metas = ds.metas
    nps = {}

    # Group by concentration and then by reconstruction
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in nps:
            nps[c] = {}
        idxs = metas['reconstruction'][rid]
        nps[c][rid] = nonp[idxs]

    # Sort by concentration
    nps = {k: v for k, v in sorted(list(nps.items()))}

    # Find largest number of trials for any conc.
    max_num_reconstructions = max([len(v) for v in nps.values()])

    # Determine positions
    concs = [k for k in nps.keys()]
    step = 0.25
    ticks = np.arange(min(concs), max(concs) + step, step)

    # Make plots
    fig, axes = plt.subplots(1, figsize=(14, 12), sharex=False, sharey=False)

    ax = axes
    ax.set_title('Non-planarity.')

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Non-Planarity')

    plots = []
    labels = []

    for c in ticks:
        if c not in nps:
            continue
        values = []
        for v in nps[c].values():
            values.append(v)
        pos = np.linspace(c - step, c + step, len(nps[c]) + 4)[2:-2]
        positions = pos.tolist()

        parts = ax.violinplot(
            values,
            positions,
            widths=step / max_num_reconstructions,
            showmeans=True,
            showmedians=True,
        )
        plots.append(parts)
        labels.append(f'{c:.2f}%')

        parts['cmedians'].set_color('green')
        parts['cmedians'].set_alpha(0.7)
        parts['cmedians'].set_linestyle(':')
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_alpha(0.7)
        parts['cmeans'].set_linestyle('--')

    ax.legend([p['bodies'][0] for p in plots], labels)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_nonp_ds={ds.id}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_non_planarity_with_splits()
    plot_non_planarity()
