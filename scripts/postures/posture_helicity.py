import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.dataset import DatasetMidline3D

show_plots = True
save_plots = True
img_extension = 'png'


def _get_dataset() -> DatasetMidline3D:
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    args = parser.parse_args()
    if args.dataset is None:
        raise RuntimeError('This script must be run with the --dataset=ID argument defined.')
    ds = DatasetMidline3D.objects.get(id=args.dataset)

    return ds


def plot_helicity():
    """
    Posture helicity split by concentration and trials.
    """
    logger.info('Fetching dataset.')
    ds = _get_dataset()
    H = ds.get_helicities(recalculate=False)
    metas = ds.metas
    hels = {}

    # Group by concentration and then by reconstruction
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in hels:
            hels[c] = {}
        idxs = metas['reconstruction'][rid]
        hels[c][rid] = H[idxs]

    # Sort by concentration
    hels = {k: v for k, v in sorted(list(hels.items()))}

    # Find largest number of trials for any conc.
    max_num_reconstructions = max([len(v) for v in hels.values()])

    # Determine positions
    concs = [k for k in hels.keys()]
    step = 0.25
    ticks = np.arange(min(concs), max(concs) + step, step)

    # Make plots
    fig, axes = plt.subplots(1, figsize=(10, 8), sharex=False, sharey=False)

    ax = axes
    ax.set_title('Posture helicities.')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Helicity')
    ax.axhline(y=0, linestyle=':', color='lightgrey', zorder=-1)

    plots = []
    labels = []

    for c in ticks:
        if c not in hels:
            continue
        values = []
        for v in hels[c].values():
            values.append(v)
        pos = np.linspace(c - step, c + step, len(hels[c]) + 4)[2:-2]
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
        path = LOGS_PATH / f'{START_TIMESTAMP}_helicity_ds={ds.id}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_helicity()
