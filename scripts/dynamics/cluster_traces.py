import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastcluster import linkage
from matplotlib import gridspec
from scipy.cluster.hierarchy import fcluster, cophenet
from scipy.spatial.distance import squareform, pdist

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction, Checkpoint
from wormlab3d.dynamics.args import DynamicsNetworkArgs, DynamicsRuntimeArgs, DynamicsDatasetArgs, DynamicsOptimiserArgs
from wormlab3d.dynamics.manager import Manager
from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.plot_utils import fancy_dendrogram, plot_reordered_distances
from wormlab3d.toolkit.util import print_args, to_numpy, str2bool
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


def find_best_linkage_settings(X):
    # Find best linkage method using cophenetic correlation coefficient
    linkage_methods = ['single', 'complete', 'average', 'centroid', 'median', 'ward', 'weighted']
    metrics = ['canberra', 'cityblock', 'euclidean', 'hamming', 'matching', 'sqeuclidean', 'seuclidean', 'cosine',
               'correlation']
    best_score = 0
    best_method = 'average'
    best_metric = 'euclidean'
    for method in linkage_methods:
        for metric in metrics:
            try:
                Z = linkage(X, method, metric)
                c, coph_dists = cophenet(Z, pdist(X, metric))
            except Exception:
                continue
            print('Method = {}. Metric = {}. Score = {}'.format(method, metric, c))
            if c > best_score:
                best_score = c
                best_method = method
                best_metric = metric
    print('Best combination: Method = {}. Metric = {}. Score = {}'.format(best_method, best_metric, best_score))
    return best_method, best_metric


def make_cluster_plots():
    """
    Generate some cluster plots.
    """
    args = parse_args()

    # Load the checkpoint
    checkpoint = Checkpoint.objects.get(id=args.checkpoint)
    runtime_args = DynamicsRuntimeArgs(
        resume=True,
        resume_from=checkpoint.id,
        cpu_only=True
    )
    dataset_args = DynamicsDatasetArgs(**{
        **checkpoint.dataset_args,
        **{'load_dataset': True, 'dataset_id': checkpoint.dataset.id}
    })
    net_args = DynamicsNetworkArgs(
        load=True,
        net_id=checkpoint.network_params.id,
        latent_size=checkpoint.network_params.latent_size,
        args_classifier=NetworkArgs(net_id=checkpoint.network_params.classifier_net.id),
        args_dynamics=NetworkArgs(net_id=checkpoint.network_params.dynamics_net.id)
    )
    optimiser_args = DynamicsOptimiserArgs(**checkpoint.optimiser_args)

    # Construct manager
    manager = Manager(
        runtime_args=runtime_args,
        dataset_args=dataset_args,
        net_args=net_args,
        optimiser_args=optimiser_args,
    )

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': dataset_args.smoothing_window,
    }

    # Get natural frame trajectory
    X_nf, meta = get_trajectory(**common_args, natural_frame=True)

    # Convert to eigenworms
    X = manager.ew.transform(np.array(X_nf))

    # Restrict to the components we need
    X = X[:, :dataset_args.n_components]

    # Stack real and imaginary
    X2 = np.stack([np.real(X), np.imag(X)], axis=-1)
    X = np.zeros((len(X), dataset_args.n_components * 2))
    X[:, ::2] = X2[..., 0]
    X[:, 1::2] = X2[..., 1]
    logger.info(f'Trajectory trace shape: {X.shape}.')

    # Standardize data if required
    if dataset_args.standardise:
        X_mean = X.mean(axis=0)
        X -= X_mean
        X_std = X.std(axis=0)
        X /= X_std

    # Include nonplanarity
    if dataset_args.include_np:
        logger.info('Fetching planarities.')
        pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
        r = pcas.explained_variance_ratio.T
        nonp = r[2] / np.sqrt(r[1] * r[0])

        # Nonplanarity goes from [0,1] so standardise with data-independent scaling to [-1,1]
        if dataset_args.standardise:
            nonp = nonp * 2 - 1
        X = np.concatenate([nonp[:, None], X], axis=1)

    # Include speed
    if dataset_args.include_speed:
        logger.info('Calculating speeds.')
        X_traj, meta = get_trajectory(**common_args)
        speeds = calculate_speeds(X_traj, signed=True)

        # Standardise with data-independent scaling factor of 100
        if dataset_args.standardise:
            speeds *= 100
        X = np.concatenate([speeds[:, None], X], axis=1)

    # Slide along sequence with 3/4 overlap collecting all the data into batches.
    start = 0
    ws = manager.dataset_args.sample_duration
    Xs = []
    while start + ws < len(X):
        Xs.append(torch.from_numpy(X[start:start + ws].T))
        start += args.step
    Xs = torch.stack(Xs).to(torch.float32).to(manager.device)
    batches = Xs.split(manager.runtime_args.batch_size)
    logger.info(f'{len(batches)} batches generated from reconstruction.')

    # Run the batches through the classifier network to get the latent encodings.
    logger.info(f'Generating encodings.')
    manager.net.eval()
    Zs = []
    for batch in batches:
        Z = manager.net.classifier_net.forward(batch)
        Zs.append(Z)
    Zs = to_numpy(torch.cat(Zs, dim=0))

    # Calculate pairwise distances
    # find_best_linkage_settings(Zs)
    logger.info(f'Calculating pairwise distances using metric "{args.distance_metric}".')
    distances = pdist(Zs, args.distance_metric)
    distances = distances

    # Calculate linkage
    logger.info(f'Calculating linkage using method "{args.linkage_method}".')
    L = linkage(distances, args.linkage_method)

    if args.plot_traces:
        _plot_cluster_trace(args, common_args, checkpoint, L)

    if args.plot_matrices:
        _plot_matrices(args, checkpoint, L, squareform(distances))


def _plot_cluster_trace(
        args: Namespace,
        common_args: dict,
        checkpoint: Checkpoint,
        L: np.ndarray,
):
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    X_traj, meta = get_trajectory(**common_args)
    N = len(X_traj)
    ts = np.linspace(0, N / reconstruction.trial.fps, N)

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
                xs.extend(np.arange(z * args.step, (z + 1) * args.step))
            ys = np.ones_like(xs) * y_locs[j]
            ax.scatter(ts[xs], ys, label=cluster_idx, color=colours[j])

        ax.set_yticks(y_locs)
        ax.grid()

    ax.set_xlabel('Time (s)')

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}. Checkpoint={checkpoint.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    title += f'\nLinkage method: {args.linkage_method}. Distance metric: {args.distance_metric}.'
    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_cluster_traces' \
                           f'_cp={checkpoint.id}' \
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
        checkpoint: Checkpoint,
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
                           f'_cp={checkpoint.id}' \
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
