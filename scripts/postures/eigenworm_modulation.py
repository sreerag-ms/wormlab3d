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
from wormlab3d.trajectories.util import calculate_speeds

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
    parser.add_argument('--min-speed', type=float,
                        help='Minimum speed to include.')
    parser.add_argument('--max-speed', type=float,
                        help='Maximum speed to include.')
    args = parser.parse_args()

    assert args.dataset is not None, \
        'This script must be run with the --dataset=ID argument defined.'

    print_args(args)

    return args


def eigenworm_modulation(
        by_concentration: bool = True,
        by_reconstruction: bool = True,
):
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

        if args.min_speed is not None or args.max_speed is not None:
            min_speed = args.min_speed if args.min_speed is not None else -np.inf
            max_speed = args.max_speed if args.max_speed is not None else np.inf
            X, _ = get_trajectory(reconstruction_id=rid, smoothing_window=25)
            speeds = calculate_speeds(X, signed=True) * reconstruction.trial.fps
            idxs = (speeds > min_speed) & (speeds < max_speed)
            Z = Z[idxs]

        X_ew = ew.transform(Z)
        lambdas[c][rid] = np.abs(X_ew)

    # Sort by concentration
    lambdas = {k: v for k, v in sorted(list(lambdas.items()))}

    # Determine positions
    concs = [float(k) for k in lambdas.keys()]
    step = 0.25
    ticks = np.arange(min(concs), max(concs) + step, step)

    # Prepare output
    out_conc = np.zeros((len(concs), ew.n_components))
    out_reconst = {c: [] for c in concs}
    out_reconst_stats = np.zeros((len(concs), ew.n_components, 2))
    for i, (c, r_vals) in enumerate(lambdas.items()):
        lambdas_c = np.concatenate([v for v in r_vals.values()])
        out_conc[i] = lambdas_c.sum(axis=0) / lambdas_c.sum()
        out_reconst[c] = np.array([rv.sum(axis=0) / rv.sum() for rv in r_vals.values()])
        out_reconst_stats[i] = np.array([out_reconst[c].mean(axis=0), out_reconst[c].std(axis=0)]).T

    # Make plots
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, len(args.plot_components)))

    def init_plot():
        fig_, axes = plt.subplots(1, figsize=(8, 6), sharex=False, sharey=False)
        ax_ = axes
        ax_.set_xticks(ticks)
        ax_.set_xticklabels(ticks)
        ax_.set_xlabel('Concentration')
        ax_.set_ylabel('Relative contribution')
        ax_.axhline(y=0, linestyle=':', color='lightgrey', zorder=-1)
        return fig_, ax_

    def make_filename(method: str):
        return LOGS_PATH / (f'{START_TIMESTAMP}' \
                            f'_ew_modulation_{method}' \
                            f'_ds={ds.id}' \
                            f'_ew={ew.id}' \
                            f'_plot={",".join([str(c_) for c_ in args.plot_components])}' + \
                            (f'_spmin={args.min_speed}' if args.min_speed is not None else '') + \
                            (f'_spmax={args.max_speed}' if args.max_speed is not None else '') + \
                            f'.{img_extension}')

    if by_concentration:
        fig, ax = init_plot()
        ax.set_title('Eigenworm contributions - by concentration')

        for i, idx in enumerate(args.plot_components):
            ax.plot(
                concs,
                out_conc[:, idx],
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
            path = make_filename('by_conc')
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

    if by_reconstruction:
        fig, ax = init_plot()
        ax.set_title('Eigenworm contributions - by reconstruction')
        offsets = np.linspace(0, 0.1, len(args.plot_components))

        for i, idx in enumerate(args.plot_components):
            for j, c in enumerate(concs):
                ij_vals = out_reconst[c][:, i]
                ax.scatter(
                    x=np.ones_like(ij_vals) * c + offsets[i],
                    y=ij_vals,
                    color=colours[i],
                    marker='o',
                    facecolor='none',
                    s=20,
                    alpha=0.6,
                )

            means = out_reconst_stats[:, idx, 0]
            stds = out_reconst_stats[:, idx, 1]
            ax.errorbar(
                concs + offsets[i],
                means,
                yerr=stds,
                capsize=5,
                color=colours[i],
                label=f'$\lambda_{i}$',
                alpha=0.7,
            )

        ax.legend()
        fig.tight_layout()

        if save_plots:
            path = make_filename('by_reconst')
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    eigenworm_modulation(
        by_concentration=True,
        by_reconstruction=True,
    )
