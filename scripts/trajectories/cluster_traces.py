import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.posture_clusters import get_posture_clusters
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

# tex_mode()

show_plots = False
save_plots = True
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot an eigenworm basis.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--eigenworms', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def traces(
        smoothing_window: int = 25,
        use_ews: bool = True,
        linkage_method: str = 'ward',
        min_clusters: int = 3,
        max_clusters: int = 7
):
    args = parse_args()
    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    X, meta = get_trajectory(**common_args, smoothing_window=smoothing_window)
    N = len(X)
    ts = np.linspace(0, N / reconstruction.trial.fps, N)

    # Speed
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True)

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1, smoothing_window=smoothing_window)
    nonp = pcas.nonp

    # Clusters
    L, _ = get_posture_clusters(
        **common_args,
        use_eigenworms=use_ews,
        eigenworms_id=args.eigenworms,
        eigenworms_n_components=args.n_components,
        linkage_method=linkage_method,
        rebuild_cache=False
    )

    # Plot
    n_cluster_plots = max_clusters - min_clusters + 1
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
    cluster_nums = list(range(min_clusters, max_clusters + 1))
    for i, n_clusters in enumerate(cluster_nums):
        logger.info(f'Clustering into {n_clusters} clusters.')
        clusters = fcluster(L, n_clusters, criterion='maxclust')
        y_locs = np.arange(1, n_clusters + 1)

        ax = axes[2 + i]
        ax.set_title(f'Clusters = {n_clusters}.')
        ax.set_ylabel('Cluster index')

        for j in range(n_clusters):
            cluster_idx = j + 1
            xs = np.argwhere(clusters == cluster_idx)[:, 0]
            ys = np.ones_like(xs) * y_locs[j]
            ax.scatter(ts[xs], ys, label=cluster_idx)

        ax.set_yticks(y_locs)
        ax.grid()

    ax.set_xlabel('Time (s)')

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    title += f'\nLinkage method: {linkage_method}. ' \
             f'Distances calculated in {"eigenspace" if use_ews else "bishop-frame"}.'
    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_cluster_traces' \
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


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    traces(
        use_ews=True,
        linkage_method='ward',
        min_clusters=3,
        max_clusters=8
    )
