from collections import OrderedDict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform

from wormlab3d import logger
from wormlab3d.data.model import SwCheckpoint
from wormlab3d.toolkit.util import parse_target_arguments


def get_checkpoint() -> SwCheckpoint:
    """
    Find a checkpoint by id.
    """
    args = parse_target_arguments()
    if args.sw_checkpoint is None:
        raise RuntimeError('This script must be run with the --sw-checkpoint=ID argument defined.')
    return SwCheckpoint.objects.get(id=args.sw_checkpoint)


def find_best_linkage_settings(X) -> Tuple[str, str]:
    # Find best linkage method using cophenetic correlation coefficient
    linkage_methods = ['single', 'complete', 'average', 'centroid', 'median', 'ward', 'weighted']
    metrics = ['canberra', 'cityblock', 'euclidean', 'hamming', 'matching', 'sqeuclidean']
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
            logger.debug(f'Method = {method}. Metric = {metric}. Score = {c}')
            if c > best_score:
                best_score = c
                best_method = method
                best_metric = metric
    logger.debug(f'Best combination: Method = {best_method}. Metric = {best_metric}. Score = {best_score}')
    return best_method, best_metric


def cluster_matrix(M, Z, threshold):
    clusters = fcluster(Z, threshold, criterion='distance')
    # clusters = fcluster(Z, 4, criterion='maxclust')
    cluster_nums = np.unique(clusters)
    m_clustered = np.zeros_like(M)
    all_idxs = []
    squares = []
    groups = OrderedDict()
    block_idx = 0
    for cluster_num in cluster_nums:
        idxs = (clusters == cluster_num).nonzero()[0]
        cluster_size = len(idxs)
        m_clustered[block_idx:(block_idx + cluster_size)] = M[idxs].copy()
        all_idxs.append(idxs)
        groups[cluster_num] = idxs
        sqr = patches.Rectangle((block_idx - 0.5, block_idx - 0.5), cluster_size, cluster_size,
                                linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        squares.append(sqr)
        block_idx += cluster_size

    # Reorder columns
    all_idxs = np.concatenate(all_idxs)
    m_clustered[:, :] = m_clustered[:, all_idxs]

    # Sort groups
    groups = OrderedDict(sorted(list(groups.items()), key=lambda x: len(x[1]), reverse=True))

    return groups, m_clustered, squares


def solution_space_clustering(
        canonical: bool=True,
        include_gamma: bool=False,
        centroid_mode: str = 'mean'
):
    """
    Loads a checkpoint and does some clustering analysis on the results found across the different runs.
    """
    checkpoint = get_checkpoint()
    runs = checkpoint.get_runs()
    bs = len(runs)

    # Extract and flatten controls from the runs
    psi0 = np.array([run.F0.psi for run in runs])
    alpha = np.array([run.CS.alpha for run in runs])
    beta = np.array([run.CS.beta for run in runs])
    gamma = np.array([run.CS.gamma for run in runs])
    alpha_vec = alpha.reshape((bs, -1))
    beta_vec = beta.reshape((bs, -1))
    gamma_vec = gamma.reshape((bs, -1))
    T = alpha.shape[1]
    N = alpha.shape[2]

    # Generate rotated "equivalent" solutions
    if include_gamma:
        solutions0 = np.concatenate([psi0, alpha_vec, beta_vec, gamma_vec], axis=1)  # default (0-rotation)
        solutions1 = np.concatenate([psi0, beta_vec, -alpha_vec, gamma_vec], axis=1)  # +pi/2 rotation
        solutions2 = np.concatenate([psi0, -alpha_vec, -beta_vec, gamma_vec], axis=1)  # +pi rotation
        solutions3 = np.concatenate([psi0, -beta_vec, alpha_vec, gamma_vec], axis=1)  # +3pi/2 rotation
    else:
        solutions0 = np.concatenate([psi0, alpha_vec, beta_vec], axis=1)  # default (0-rotation)
        solutions1 = np.concatenate([psi0, beta_vec, -alpha_vec], axis=1)  # +pi/2 rotation
        solutions2 = np.concatenate([psi0, -alpha_vec, -beta_vec], axis=1)  # +pi rotation
        solutions3 = np.concatenate([psi0, -beta_vec, alpha_vec], axis=1)  # +3pi/2 rotation

    # Combine solutions
    solutions = np.zeros((4 * bs, solutions0.shape[-1]))
    solutions[0::4] = solutions0
    if canonical:
        solutions[1::4] = solutions1
        solutions[2::4] = solutions2
        solutions[3::4] = solutions3
    else:
        solutions[1::4] = solutions0
        solutions[2::4] = solutions0
        solutions[3::4] = solutions0

    # Determine distances between solutions and take the minimum for each canonical solution
    d_all_cond = pdist(solutions)
    d_all = squareform(d_all_cond)
    d = np.zeros((bs, bs))
    for i in range(bs - 1):
        for j in range(i + 1, bs):
            dij = d_all[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4].min()
            d[i, j] = dij
            d[j, i] = dij
    d_cond = squareform(d)

    # find_best_linkage_settings(d)
    # methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    methods = ['complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    fig, axes = plt.subplots(len(methods), 6, figsize=(18, 18))
    fig.suptitle(f'including gamma = {include_gamma}, canonical={canonical}')
    for i, method in enumerate(methods):
        Z = linkage(d_cond, method=method, optimal_ordering=True)
        threshold = max(Z[:, 2]) * 0.6
        ax = axes[i, 0]
        dn = dendrogram(Z, color_threshold=threshold, ax=ax)
        ax.set_title(method)

        groups, d_clust, squares = cluster_matrix(d, Z, threshold)

        # Clustered distances
        ax = axes[i, 1]
        ax.axis('off')
        ax.matshow(d_clust)
        for sqr in squares:
            ax.add_patch(sqr)

        # Pick centroid for each group
        for j, (cluster_num, idxs) in enumerate(groups.items()):
            if j > 3:
                break

            ax = axes[i, 2 + j]
            ax.axis('off')

            if centroid_mode == 'mean':
                controls = np.concatenate([
                    alpha[idxs].mean(axis=0),
                    beta[idxs].mean(axis=0),
                    np.c_[gamma[idxs].mean(axis=0), np.zeros(T)],
                ])
                ax.set_title(f'cluster={cluster_num} size={len(idxs)}\nmean')
            else:
                subgraph = d[idxs[:, None], idxs]
                subgraph_dists = subgraph.sum(axis=0)
                candidate = idxs[subgraph_dists.argmin()]
                controls = np.concatenate([
                    alpha[candidate],
                    beta[candidate],
                    np.c_[gamma[candidate], np.zeros(T)],
                ])
                ax.set_title(f'cluster={cluster_num} size={len(idxs)}\neg idx={candidate}')
            ax.matshow(controls.T)
        while j <= 3:
            axes[i, 2+j].axis('off')
            j+=1


    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # solution_space_clustering(canonical=True, include_gamma=True, centroid_mode='mean')
    # solution_space_clustering(canonical=True, include_gamma=True, centroid_mode='exemplar')
    # solution_space_clustering(canonical=False, include_gamma=True, centroid_mode='mean')
    solution_space_clustering(canonical=False, include_gamma=True, centroid_mode='exemplar')
