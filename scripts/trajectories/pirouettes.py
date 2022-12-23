import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.manoeuvres import get_manoeuvres
from wormlab3d.trajectories.util import smooth_trajectory

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot pirouettes.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--min-reversal-frames', type=int, default=25,
                        help='Minimum number of reversal frames to use to identify a manoeuvre.')
    parser.add_argument('--min-reversal-distance', type=float, default=0.,
                        help='Minimum reversal distance to use to identify a manoeuvre.')
    parser.add_argument('--manoeuvre-window', type=int, default=100,
                        help='Number of frames to include either side of a detected manoeuvre.')

    parser.add_argument('--max-reversal-interval', type=int, default=125,
                        help='Maximum number of frames allowed between reversals of the same cluster.')
    parser.add_argument('--min-manoeuvres-in-cluster', type=int, default=3,
                        help='Minimum manoeuvres required to qualify as a cluster.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_rf={args.min_reversal_frames}' \
           f'_rd={args.min_reversal_distance}' \
           f'_mw={args.manoeuvre_window}' \
           f'_ri={args.max_reversal_interval}' \
           f'_s={args.min_manoeuvres_in_cluster}'


def _nonp(pca: PCA) -> np.ndarray:
    r = pca.explained_variance_ratio_.T
    return r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    metas = ds.metas
    durations = {}
    n_turns = {}
    nonps = {}

    # Group by concentration and then by reconstruction
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in n_turns:
            n_turns[c] = []
            durations[c] = []
            nonps[c] = []

        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        # Fetch reconstruction and centre
        X, _ = get_trajectory(reconstruction_id=rid, smoothing_window=args.smoothing_window)
        X = X - X.mean(axis=(0, 1), keepdims=True)
        N = X.shape[1]

        # Pick trajectory point
        if args.trajectory_point == -1:
            Xt = X.mean(axis=1)
        else:
            u = round(args.trajectory_point * N)
            if u == N:
                u -= 1
            assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
            Xt = X[:, u]

        # Get the manoeuvres
        manoeuvres = get_manoeuvres(
            X,
            Xt,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=False,
        )

        # Cluster the manoeuvres
        clusters = []
        active_cluster = []
        for i, m in enumerate(manoeuvres):
            if len(active_cluster) > 0:
                if m['rev_start_idx'] - active_cluster[-1]['rev_end_idx'] < args.max_reversal_interval:
                    active_cluster.append(m)
                else:
                    if len(active_cluster) >= args.min_manoeuvres_in_cluster:
                        clusters.append(active_cluster)
                    active_cluster = [m]

            else:
                active_cluster = [m]

        # Add any final cluster
        if len(active_cluster) >= args.min_manoeuvres_in_cluster:
            clusters.append(active_cluster)
        logger.info(f'Found {len(clusters)} clusters.')

        # Build the pirouette statistics
        for cluster in clusters:
            start_idx = cluster[0]['start_idx']
            end_idx = cluster[-1]['end_idx']

            # Calculate PCA on entire pirouette
            Xp = Xt[start_idx:end_idx]
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            pca.fit(Xp)

            nonpc = np.ptp(Xp @ pca.components_, axis=0)
            vol = np.product(nonpc)

            # Collate
            durations[c].append((end_idx - start_idx) / fps)
            n_turns[c].append(len(cluster))
            # nonps[c].append(_nonp(pca))
            nonps[c].append(vol)

    # Sort by concentration
    durations = {k: np.array(v) for k, v in sorted(list(durations.items()))}
    n_turns = {k: np.array(v) for k, v in sorted(list(n_turns.items()))}
    nonps = {k: np.array(v) for k, v in sorted(list(nonps.items()))}

    return durations, n_turns, nonps


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[Dataset, Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    durations = None
    n_turns = None
    nonps = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            durations = {}
            n_turns = {}
            nonps = {}
            for name in data.files:
                c, d_or_a = name.split('_')
                if d_or_a != 'durations':
                    continue
                c = float(c)
                durations[c] = data[name]
                n_turns[c] = data[f'{c}_n-turns']
                nonps[c] = data[f'{c}_nonps']
            durations = {k: v for k, v in sorted(list(durations.items()))}
            n_turns = {k: v for k, v in sorted(list(n_turns.items()))}
            nonps = {k: v for k, v in sorted(list(nonps.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            durations = None
            logger.warning(f'Could not load cache: {e}')

    if durations is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        durations, n_turns, nonps = _calculate_data(args, ds)
        data = {}
        for c, d_vals in durations.items():
            data[f'{c}_durations'] = d_vals
            data[f'{c}_n-turns'] = n_turns[c]
            data[f'{c}_nonps'] = nonps[c]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, durations, n_turns, nonps


def plot_pirouettes():
    """
    Plot the pauses, durations against activity.
    """
    args = parse_args()
    ds, durations, n_turns, nonps = _generate_or_load_data(args, rebuild_cache=True,
                                                           cache_only=False)

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    d_vals = np.concatenate([d for c, d in durations.items() if _is_included(c)])
    nt_vals = np.concatenate([a for c, a in n_turns.items() if _is_included(c)])
    np_vals = np.concatenate([ad for c, ad in nonps.items() if _is_included(c)])
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Set up plot
    plt.rc('axes', labelsize=9)  # fontsize of the X label
    plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=8)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=3)
    plt.rc('ytick.major', pad=2, size=3)

    gs = GridSpec(
        nrows=3,
        ncols=2,
        width_ratios=(4, 2.5),
        height_ratios=(2.2, 3, 3),
        wspace=0,
        hspace=0,
        top=0.97,
        bottom=0.14,
        left=0.1,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, 4))

    # Get colours
    cm = plt.get_cmap('jet')
    concs = [float(k) for k in durations.keys() if _is_included(k)]
    ticks = np.arange(min(concs), max(concs) + 0.25, 0.25)
    all_colours = cm(np.linspace(0, 1, len(ticks)))
    colours = {c: all_colours[i] for i, c in enumerate(ticks) if c in concs}

    # Number of turns scatter plot
    ax_scat_nt = fig.add_subplot(gs[1, 0])
    ax_scat_nt.set_ylabel('Number of turns', labelpad=8)
    for c, durations_c in durations.items():
        if not _is_included(c):
            continue
        ax_scat_nt.scatter(durations_c, n_turns[c], label=f'{c:.2f}% ({len(durations_c):,d})', s=20, marker='o',
                        facecolors='none', edgecolors=colours[c], alpha=0.7)
    legend = ax_scat_nt.legend()
    ax_scat_nt.set_yticks([3,5,10])
    ax_scat_nt.spines['top'].set_visible(False)
    ax_scat_nt.spines['right'].set_visible(False)
    ax_scat_nt.spines['bottom'].set(linestyle='--', color='grey')
    ax_scat_nt.tick_params(axis='x', bottom=False, labelbottom=False)

    # Non-planarity scatter plot
    ax_scat_np = fig.add_subplot(gs[2, 0])
    ax_scat_np.set_xlabel('Duration (s)', labelpad=5)
    ax_scat_np.set_ylabel('Non-planarity', labelpad=8)
    for c, durations_c in durations.items():
        if not _is_included(c):
            continue
        ax_scat_np.scatter(durations_c, nonps[c], s=20, marker='o',
                        facecolors='none', edgecolors=colours[c], alpha=0.7)
    ax_scat_np.set_yticks([0, 0.2, 0.4])
    ax_scat_np.spines['top'].set_visible(False)
    ax_scat_np.spines['right'].set_visible(False)

    # Duration histogram
    ax_hist_duration = fig.add_subplot(gs[0, 0])
    ax_hist_duration.hist(d_vals, orientation='vertical', color='green', **hist_args)
    ax_hist_duration.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_hist_duration.spines['bottom'].set(linestyle='--', color='grey')
    ax_hist_duration.set_ylabel('Density')
    # ax_hist_duration.set_yticks([0.5])

    # Number of turns histogram
    ax_hist_nt = fig.add_subplot(gs[1, 1])
    ax_hist_nt.hist(nt_vals, orientation='horizontal', color='tab:cyan', **hist_args)
    ax_hist_nt.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_nt.spines['left'].set(linestyle='--', color='grey')
    ax_hist_nt.set_xticks([0.5])

    # Non-planarity histogram
    ax_hist_np = fig.add_subplot(gs[2, 1])
    ax_hist_np.hist(np_vals, orientation='horizontal', color='plum', **hist_args)
    ax_hist_np.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_np.spines['left'].set(linestyle='--', color='grey')
    ax_hist_np.set_xlabel('Density')
    ax_hist_np.set_xticks([10])

    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_hist_duration.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), bbox_transform=ax_hist_duration.transAxes, ncol=2, columnspacing=0.7, handletextpad=0.2)
    legend.remove()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_pirouettes'
                            f'_{_identifiers(args)}'
                            f'_n={len(d_vals)}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    plot_pirouettes()
