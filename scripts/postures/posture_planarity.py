import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Dataset

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
    Scatter plot of planarity against concentrations.
    """
    split_keys = ['user', 'concentration', 'source']
    logger.info('Fetching dataset.')
    ds = _get_dataset()
    nonp = ds.get_nonplanarities()
    metas = ds.metas

    # Make plots
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey=False)

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

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_nonp_splits_ds={ds.id}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_non_planarity_with_splits()
