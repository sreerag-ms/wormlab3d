import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction, Eigenworms
from wormlab3d.data.model.dataset import Dataset
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.util import print_args, hash_data
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import calculate_speeds

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

show_plots = True
save_plots = True
img_extension = 'svg'


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


def _calculate_data(
        args: Namespace,
        ds: Dataset,
        ew: Eigenworms
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Fetch the data.
    """
    logger.info('Fetching dataset.')
    metas = ds.metas
    lambdas = {}

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

        if len(Z):
            X_ew = ew.transform(Z)
            lambdas[c][rid] = np.abs(X_ew)

    # Sort by concentration
    lambdas = {k: v for k, v in sorted(list(lambdas.items()))}

    return lambdas


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[Dataset, Eigenworms, Dict[float, Dict[str, np.ndarray]]]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        dataset_id=args.dataset,
        n_components=args.n_components,
        regenerate=False
    )
    additional = {
        'min_speed': args.min_speed,
        'max_speed': args.max_speed
    }

    cache_path = DATA_CACHE_PATH / f'ds={ds.id}_ew={ew.id}_{hash_data(additional)}'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    lambdas = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            lambdas = {}
            for name in data.files:
                c, rid = name.split('_')
                c = float(c)
                if c not in lambdas:
                    lambdas[c] = {}
                lambdas[c][rid] = data[name]
            lambdas = {k: v for k, v in sorted(list(lambdas.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            lambdas = None
            logger.warning(f'Could not load cache: {e}')

    if lambdas is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        lambdas = _calculate_data(args, ds, ew)
        data = {
            f'{c}_{rid}': r_vals
            for c, c_vals in lambdas.items()
            for rid, r_vals in c_vals.items()
        }
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, ew, lambdas


def eigenworm_modulation(
        by_concentration: bool = True,
        by_reconstruction: bool = True,
):
    """
    Show how eigenworms vary across concentrations/reconstructions.
    """
    args = parse_args()
    ds, ew, lambdas = _generate_or_load_data(args)

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
        return LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_ew_modulation_{method}'
                            f'_ds={ds.id}'
                            f'_ew={ew.id}'
                            f'_plot={",".join([str(c_) for c_ in args.plot_components])}' +
                            (f'_spmin={args.min_speed}' if args.min_speed is not None else '') +
                            (f'_spmax={args.max_speed}' if args.max_speed is not None else '') +
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


def eigenworm_modulation_by_conc():
    """
    Show how eigenworms vary across concentrations.
    Make a paper-ready bespoke plot.
    """
    args = parse_args()
    ds, ew, lambdas = _generate_or_load_data(args, rebuild_cache=False, cache_only=False)
    # exclude_concs = []
    # break_at = 100
    exclude_concs = [2.75, 4]
    break_at = 5

    # Determine positions
    concs = [float(k) for k in lambdas.keys() if k not in exclude_concs]
    ticks = np.arange(len(concs))

    # Prepare output
    out_conc = np.zeros((len(concs), ew.n_components))
    out_reconst = {c: [] for c in concs}
    out_reconst_stats = np.zeros((len(concs), ew.n_components, 2))
    i = 0
    for c, r_vals in lambdas.items():
        if c in exclude_concs:
            continue
        lambdas_c = np.concatenate([v for v in r_vals.values()])
        out_conc[i] = lambdas_c.sum(axis=0) / lambdas_c.sum()
        out_reconst[c] = np.array([rv.sum(axis=0) / rv.sum() for rv in r_vals.values()])
        out_reconst_stats[i] = np.array([out_reconst[c].mean(axis=0), out_reconst[c].std(axis=0)]).T
        i += 1

    # Make plots
    # cmap = plt.get_cmap('jet')
    # colours = cmap(np.linspace(0, 1, len(args.plot_components)))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = prop_cycle.by_key()['color']

    plt.rc('axes', labelsize=6)  # fontsize of the X label
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('ytick.major', pad=1, size=2)

    fig, ax = plt.subplots(1, figsize=(2.15, 1.9), gridspec_kw={
        'left': 0.16,
        'right': 0.99,
        'top': 0.97,
        'bottom': 0.16,
    })
    ax.set_xticks(ticks)
    ax.set_xticklabels(concs)
    ax.set_xlabel('Concentration (% gelatin)')
    ax.set_ylabel('Relative contribution', labelpad=1)

    plot_args = dict(
        linewidth=2,
        marker='x',
        markersize=7,
        markeredgewidth=1.5,
        alpha=0.8,
    )

    for i, idx in enumerate(args.plot_components):
        ax.plot(
            ticks[:break_at],
            out_conc[:break_at, idx],
            label=f'$\lambda_{i + 1}$',
            color=colours[i],
            **plot_args
        )
        if break_at < len(ticks):
            ax.plot(
                ticks[break_at:],
                out_conc[break_at:, idx],
                color=colours[i],
                **plot_args
            )

    if break_at < len(ticks):
        dx = .1
        dy = .02
        dist = 0.2
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        break_line_args = dict(transform=trans, color='k', clip_on=False)
        x_div = break_at - 0.5
        ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (-dy, dy), **break_line_args)
        ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (-dy, dy), **break_line_args)
        ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)
        ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)
        # ax.axvline(x=x_div, linestyle=':', color='k')

    ax.set_yticks([0.1, 0.15, 0.2, 0.25])
    # ax.legend(loc='upper left',  markerscale=0.8, handlelength=1, handletextpad=0.8, labelspacing=1,
    #           borderpad=0.8, bbox_to_anchor=(1.01, 0.9), bbox_transform=ax.transAxes)
    ax.legend(loc='upper right', markerscale=0.8, handlelength=1, handletextpad=0.6, labelspacing=0,
              borderpad=0.7, ncol=5, columnspacing=0.8, bbox_to_anchor=(0.99, 0.98), bbox_transform=ax.transAxes)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_ew_modulation_conc'
                            f'_ds={ds.id}'
                            f'_ew={ew.id}'
                            f'_plot={",".join([str(c_) for c_ in args.plot_components])}' +
                            f'_exc={",".join([str(c_) for c_ in exclude_concs])}' +
                            (f'_spmin={args.min_speed}' if args.min_speed is not None else '') +
                            (f'_spmax={args.max_speed}' if args.max_speed is not None else '') +
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def eigenworm_modulation_by_rec(
        layout: str = 'paper'
):
    """
    Show how eigenworms vary across reconstruction.
    Make a paper-ready bespoke plot.
    """
    args = parse_args()
    ds, ew, lambdas = _generate_or_load_data(args, rebuild_cache=False, cache_only=True)
    # exclude_concs = []
    # break_at = 100
    exclude_concs = [2.75, 4]
    break_at = 5

    # Determine positions
    concs = [float(k) for k in lambdas.keys() if k not in exclude_concs]
    ticks = np.arange(len(concs))

    # Prepare output
    out_conc = np.zeros((len(concs), ew.n_components))
    out_reconst = {c: [] for c in concs}
    out_reconst_stats = np.zeros((len(concs), ew.n_components, 2))
    i = 0
    for c, r_vals in lambdas.items():
        if c in exclude_concs:
            continue
        lambdas_c = np.concatenate([v for v in r_vals.values()])
        out_conc[i] = lambdas_c.sum(axis=0) / lambdas_c.sum()
        out_reconst[c] = np.array([rv.sum(axis=0) / rv.sum() for rv in r_vals.values()])
        out_reconst_stats[i] = np.array([out_reconst[c].mean(axis=0), out_reconst[c].std(axis=0)]).T
        i += 1

    # Make plots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = prop_cycle.by_key()['color']

    if layout == 'paper':
        plt.rc('axes', labelsize=6)  # fontsize of the X label
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=2)
        plt.rc('ytick.major', pad=1, size=2)
        fig, ax = plt.subplots(1, figsize=(2.15, 1.9), gridspec_kw={
            'left': 0.14,
            'right': 0.99,
            'top': 0.97,
            'bottom': 0.15,
        })
        ax.set_xlabel('Concentration (% gelatin)')
        ax.set_ylabel('Relative contribution', labelpad=1)
    else:
        plt.rc('axes', labelsize=9)  # fontsize of the X label
        plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=8)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)
        fig, ax = plt.subplots(1, figsize=(4.5, 2.6), gridspec_kw={
            'left': 0.14,
            'right': 0.84,
            'top': 0.97,
            'bottom': 0.18,
        })
        ax.set_xlabel('Concentration (% gelatin)', labelpad=5)
        ax.set_ylabel('Relative contribution', labelpad=8)

    ax.set_xticks(ticks)
    ax.set_xticklabels(concs)
    offsets = np.linspace(-0.1, 0.1, len(args.plot_components))

    for i, idx in enumerate(args.plot_components):
        for j, c in enumerate(concs):
            ij_vals = out_reconst[c][:, i]
            ax.scatter(
                x=np.ones_like(ij_vals) * ticks[j] + offsets[i],
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
            ticks[:break_at] + offsets[i],
            means[:break_at],
            yerr=stds[:break_at],
            capsize=5,
            color=colours[i],
            label=f'$\lambda_{i + (1 if layout == "paper" else 0)}$',
            alpha=0.7,
        )

        if break_at < len(ticks):
            ax.errorbar(
                ticks[break_at:] + offsets[i],
                means[break_at:],
                yerr=stds[break_at:],
                capsize=5,
                color=colours[i],
                alpha=0.7,
            )

    if break_at < len(ticks):
        if layout == 'paper':
            dx = .1
            dy = .02
            dist = 0.2
        else:
            dx = .08
            dy = .03
            dist = 0.16

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        break_line_args = dict(transform=trans, color='k', clip_on=False)
        x_div = break_at - 0.5
        ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (-dy, dy), **break_line_args)
        ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (-dy, dy), **break_line_args)
        ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)
        ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)

    ax.set_yticks([0.1, 0.15, 0.2, 0.25])

    if layout == 'paper':
        # Remove the errorbars from the legend handles
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        ax.legend(handles, labels, loc='upper right', markerscale=0.8, handlelength=1, handletextpad=0.6, labelspacing=0,
                  borderpad=0.5, ncol=5, columnspacing=0.8, bbox_to_anchor=(0.99, 0.98), bbox_transform=ax.transAxes)

    else:
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1.01, 0.95), bbox_transform=ax.transAxes)

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_ew_modulation_recs'
                            f'_ds={ds.id}'
                            f'_ew={ew.id}'
                            f'_plot={",".join([str(c_) for c_ in args.plot_components])}' +
                            f'_exc={",".join([str(c_) for c_ in exclude_concs])}' +
                            (f'_spmin={args.min_speed}' if args.min_speed is not None else '') +
                            (f'_spmax={args.max_speed}' if args.max_speed is not None else '') +
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # eigenworm_modulation(
    #     by_concentration=True,
    #     by_reconstruction=True,
    # )
    # eigenworm_modulation_by_conc()
    eigenworm_modulation_by_rec(layout='thesis')
