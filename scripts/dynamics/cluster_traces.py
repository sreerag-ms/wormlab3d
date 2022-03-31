import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.dynamics.clusterer import DynamicsClusterer
from wormlab3d.toolkit.plot_utils import fancy_dendrogram, plot_reordered_distances
from wormlab3d.toolkit.util import print_args, str2bool
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

# tex_mode()

show_plots = False
save_plots = True
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot cluster traces.')
    parser.add_argument('--checkpoint', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--distance-metric', type=str, help='Distance metric.')
    parser.add_argument('--linkage-method', type=str, help='Linkage method to use.')
    parser.add_argument('--min-clusters', type=int, help='Minimum number of clusters to plot.')
    parser.add_argument('--max-clusters', type=int, help='Maximum number of clusters to plot.')
    parser.add_argument('--step', type=int, help='Window step size.')
    parser.add_argument('--plot-traces', type=str2bool, help='Plot the traces.')
    parser.add_argument('--plot-matrices', type=str2bool, help='Plot the matrices.')

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    assert args.step > 0, 'Step size must be > 0.'
    assert args.plot_traces or args.plot_matrices, 'One of the plot options must be set.'

    print_args(args)

    return args


def make_cluster_plots():
    """
    Generate some cluster plots.
    """
    args = parse_args()

    DC = DynamicsClusterer(
        checkpoint_id=args.checkpoint,
        reconstruction_id=args.reconstruction,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        step=args.step
    )

    L, distances = DC.cluster(
        distance_metric=args.distance_metric,
        linkage_method=args.linkage_method,
    )

    if args.plot_traces:
        _plot_cluster_trace(args, DC, L)

    if args.plot_matrices:
        _plot_matrices(args, DC, L, squareform(distances))


def _plot_cluster_trace(
        args: Namespace,
        DC: DynamicsClusterer,
        L: np.ndarray,
):
    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': DC.dataset_args.smoothing_window,
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    X_traj, meta = get_trajectory(**common_args)
    N = len(X_traj)
    ts = np.linspace(0, N / reconstruction.trial.fps, N)
    ws = DC.dataset_args.sample_duration

    # Speed
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X_traj, signed=True)

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    r = pcas.explained_variance_ratio.T
    nonp = r[2] / np.sqrt(r[1] * r[0])

    # Plot
    n_cluster_plots = args.max_clusters - args.min_clusters + 1
    fig, axes = plt.subplots(2 + n_cluster_plots, figsize=(16, (2 + n_cluster_plots) * 3), sharex=True)

    # Speeds
    ax = axes[0]
    ax.axhline(y=0, color='lightgrey')
    ax.plot(ts, speeds)
    ax.set_ylabel('Speed')
    ax.grid()

    # Planarities
    ax = axes[1]
    ax.plot(ts, nonp)
    ax.set_ylabel('Non-Planarity')
    ax.grid()

    # Cluster plots
    cmap = plt.get_cmap('jet')
    cluster_nums = list(range(args.min_clusters, args.max_clusters + 1))
    for i, n_clusters in enumerate(cluster_nums):
        logger.info(f'Clustering into {n_clusters} clusters.')
        clusters = fcluster(L, n_clusters, criterion='maxclust')
        y_locs = np.arange(1, n_clusters + 1)
        colours = cmap(np.linspace(0, 1, n_clusters))

        ax = axes[2 + i]
        ax.set_title(f'Clusters = {n_clusters}.')
        ax.set_ylabel('Cluster index')

        for j in range(n_clusters):
            cluster_idx = j + 1
            zs = np.argwhere(clusters == cluster_idx)[:, 0]
            xs = []
            for z in zs:
                xs.extend(np.arange(z * args.step, (z + 1) * args.step) + int((ws - args.step) / 2))
            ys = np.ones_like(xs) * y_locs[j]
            ax.scatter(ts[xs], ys, label=cluster_idx, color=colours[j])

        ax.set_yticks(y_locs)
        ax.grid()

    ax.set_xlabel('Time (s)')

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}. Checkpoint={DC.checkpoint.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    title += f'\nLinkage method: {args.linkage_method}. Distance metric: {args.distance_metric}.'
    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_cluster_traces' \
                           f'_cp={DC.checkpoint.id}' \
                           f'_d={args.distance_metric}' \
                           f'_l={args.linkage_method}' \
                           f'_c={args.min_clusters}-{args.max_clusters}' \
                           f'_r={reconstruction.id}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def _plot_matrices(
        args: Namespace,
        DC: DynamicsClusterer,
        L: np.ndarray,
        distances_sf: np.ndarray
):
    # Set up plots
    n_cluster_plots = args.max_clusters - args.min_clusters + 1
    n_cluster_plot_rows = int(np.ceil(n_cluster_plots / 3))
    n_rows = 1 + n_cluster_plot_rows
    n_cols = 3
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, fig)

    # Show original data
    ax = plt.subplot(gs[0, 0])
    ax.matshow(distances_sf, cmap=plt.cm.Blues)
    ax.set_title('Distances between encodings')

    # Calculate and plot full dendrogram
    ax = plt.subplot(gs[0, 1:])
    fancy_dendrogram(
        ax,
        L,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=True,  # otherwise numbers in brackets are counts
        leaf_rotation=0.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        annotate_above=1,  # useful in small plots so annotations don't overlap,
    )
    cluster_nums = list(range(args.min_clusters, args.max_clusters + 1))
    cluster_idx = 0
    for row_idx in range(n_cluster_plot_rows):
        for col_idx in range(3):
            if cluster_idx >= len(cluster_nums):
                break
            n_clusters = cluster_nums[cluster_idx]
            clusters = fcluster(L, n_clusters, criterion='maxclust')
            ax = plt.subplot(gs[row_idx + 1, col_idx])
            plot_reordered_distances(ax, distances_sf, clusters)
            ax.axis('off')
            cluster_idx += 1

    fig.suptitle(f'Reconstruction: {args.reconstruction}. '
                 f'Linkage method: {args.linkage_method}. '
                 f'Distance metric: {args.distance_metric}.')
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_distances' \
                           f'_cp={DC.checkpoint.id}' \
                           f'_d={args.distance_metric}' \
                           f'_l={args.linkage_method}' \
                           f'_c={args.min_clusters}-{args.max_clusters}' \
                           f'_r={args.reconstruction}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
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
    make_cluster_plots()
