import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.posture_clusters import get_posture_clusters
from wormlab3d.postures.posture_distances import get_posture_distances
from wormlab3d.toolkit.plot_utils import fancy_dendrogram
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache

# tex_mode()

show_plots = False
save_plots = True
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to cluster postures.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--eigenworms', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def _reorder_distance_correlations(
        distances: np.ndarray,
        clusters: np.ndarray
) -> Tuple[np.ndarray, List[Rectangle], np.ndarray]:
    """
    Reorder the distance correlations.
    """
    block_idx = 0
    cluster_nums = np.unique(clusters)
    reordered_distances = np.zeros_like(distances)
    all_idxs = []
    squares = []
    for cluster_num in cluster_nums:
        idxs = (clusters == cluster_num).nonzero()[0]
        cluster_size = len(idxs)
        reordered_distances[block_idx:(block_idx + cluster_size)] = distances[idxs].copy()
        all_idxs.append(idxs)
        sqr = Rectangle((block_idx - 0.5, block_idx - 0.5), cluster_size, cluster_size,
                        linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        squares.append(sqr)
        block_idx += cluster_size

    # Reorder columns
    all_idxs = np.concatenate(all_idxs)
    reordered_distances[:, :] = reordered_distances[:, all_idxs]

    return reordered_distances, squares, all_idxs


def _plot_reordered_distances(ax: Axes, distances: np.ndarray, clusters: np.ndarray):
    """
    Reorder and plot the distances.
    """
    cluster_nums = np.unique(clusters)
    reordered_distances, squares, all_idxs = _reorder_distance_correlations(distances, clusters)

    # Plot reordered distances
    ax.matshow(reordered_distances, cmap=plt.cm.Blues)
    ax.set_title(f'Number of clusters = {len(cluster_nums)}.')

    # Show clusters on plot
    for sqr in squares:
        ax.add_patch(sqr)


def cluster_postures(
        use_ews: bool = True,
        linkage_method: str = 'ward',
        min_clusters: int = 3,
        max_clusters: int = 14
):
    """
    Plot clustered posture matrices.
    """
    args = parse_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Fetch distances
    distances, _ = get_posture_distances(
        reconstruction_id=reconstruction.id,
        use_eigenworms=use_ews,
        eigenworms_id=args.eigenworms,
        eigenworms_n_components=args.n_components,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        return_squareform=True,
        rebuild_cache=False
    )

    # Fetch clusters
    L, _ = get_posture_clusters(
        reconstruction_id=reconstruction.id,
        use_eigenworms=use_ews,
        eigenworms_id=args.eigenworms,
        eigenworms_n_components=args.n_components,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        linkage_method=linkage_method,
        rebuild_cache=False
    )

    # Set up plots
    n_cluster_plots = max_clusters - min_clusters + 1
    n_cluster_plot_rows = int(np.ceil(n_cluster_plots / 3))
    n_rows = 1 + n_cluster_plot_rows
    n_cols = 3
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, n_cols, fig)

    # Show original data
    ax = plt.subplot(gs[0, 0])
    ax.matshow(distances, cmap=plt.cm.Blues)
    ax.set_title('Distances between postures')

    # Calculate and plot full dendrogram
    ax = plt.subplot(gs[0, 1:])
    fancy_dendrogram(
        ax,
        L,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        # show_leaf_counts=True,  # otherwise numbers in brackets are counts
        leaf_rotation=0.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        annotate_above=1,  # useful in small plots so annotations don't overlap,
        # max_d=70,  # plot a horizontal cut-off line
    )
    # clusters = fcluster(L, max_d, criterion='distance')
    # clusters = fcluster(L, 1.8, depth=1000)

    cluster_nums = list(range(min_clusters, max_clusters + 1))
    cluster_idx = 0
    for row_idx in range(n_cluster_plot_rows):
        for col_idx in range(3):
            if cluster_idx >= len(cluster_nums):
                break
            n_clusters = cluster_nums[cluster_idx]
            logger.info(f'Clustering into {n_clusters} clusters.')
            clusters = fcluster(L, n_clusters, criterion='maxclust')
            ax = plt.subplot(gs[row_idx + 1, col_idx])
            _plot_reordered_distances(ax, distances, clusters)
            ax.axis('off')
            cluster_idx += 1

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}.'
    if args.start_frame is not None:
        title += f'Frames={args.start_frame}-{args.end_frame}.'
    title += f'\nLinkage method: {linkage_method}. ' \
             f'Distances calculated in {"eigenspace" if use_ews else "bishop-frame"}.'
    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_clusters' \
                           f'_{"ew" if use_ews else "nf"}' \
                           f'_l={linkage_method}' \
                           f'_c={min_clusters}-{max_clusters}' \
                           f'_r={reconstruction.id}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def tsne_postures(
        embedding_dim: int,
        nonp_threshold: float = None
):
    """
    Plot t-SNE embeddings of postures.
    """
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )
    assert embedding_dim in [2, 3]

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25
    }
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    r = pcas.explained_variance_ratio.T
    nonp = r[2] / np.sqrt(r[1] * r[0])

    # Natural frame
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)

    # Filter data
    if nonp_threshold is not None:
        Z = Z[nonp > nonp_threshold]
        nonp = nonp[nonp > nonp_threshold]

    # Eigenworms embeddings
    Q = ew.transform(np.array(Z))
    Q2 = np.concatenate([np.real(Q), np.imag(Q)], axis=1)

    # tSNE projections
    logger.info(f'Calculating {embedding_dim}-component tSNE embedding.')
    tsne = TSNE(n_components=embedding_dim)
    T = tsne.fit_transform(Q2).T

    fig = plt.figure(figsize=(16, 16))
    if embedding_dim == 3:
        ax = fig.add_subplot(projection='3d')
        s = ax.scatter(T[0], T[1], T[2], c=nonp)
    else:
        ax = fig.add_subplot()
        s = ax.scatter(T[0], T[1], c=nonp)

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}.'
    if nonp_threshold is not None:
        title += f'\nNon-planarity threshold={nonp_threshold:.2f}.'
    ax.set_title(title)

    fig.colorbar(s)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_tsne_{embedding_dim}d' \
                           f'_r={reconstruction.id}' \
                           f'_t={nonp_threshold:.2f}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # interactive()
    # tsne_postures(embedding_dim=2, nonp_threshold=0.1)
    # tsne_postures(embedding_dim=3, nonp_threshold=0.1)
    # tsne_postures(embedding_dim=2, nonp_threshold=0.2)
    # tsne_postures(embedding_dim=3, nonp_threshold=0.2)
    # tsne_postures(embedding_dim=2, nonp_threshold=0.3)
    # tsne_postures(embedding_dim=3, nonp_threshold=0.3)

    cluster_postures(use_ews=True, linkage_method='ward', min_clusters=3, max_clusters=14)
    # for ews in [True, False]:
    #     for l_method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
    #         cluster_postures(use_ews=ews, linkage_method=l_method)
