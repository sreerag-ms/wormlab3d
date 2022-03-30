import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction, Checkpoint
from wormlab3d.dynamics.args import DynamicsNetworkArgs, DynamicsRuntimeArgs, DynamicsDatasetArgs, DynamicsOptimiserArgs
from wormlab3d.dynamics.manager import Manager
from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import print_args, to_numpy
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

# tex_mode()

show_plots = True
save_plots = False
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot cluster traces.')
    parser.add_argument('--checkpoint', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--linkage-method', type=str, help='Linkage method to use.')
    parser.add_argument('--min-cluster', type=str, help='Minimum number of clusters to plot.')
    parser.add_argument('--max-cluster', type=str, help='Maximum number of clusters to plot.')

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def traces(
        linkage_method: str = 'ward',
        min_clusters: int = 3,
        max_clusters: int = 7
):
    args = parse_args()

    # Load the checkpoint
    checkpoint = Checkpoint.objects.get(id=args.checkpoint)
    runtime_args = DynamicsRuntimeArgs(
        resume=True,
        resume_from=checkpoint.id,
        cpu_only=True
    )
    dataset_args = DynamicsDatasetArgs(**checkpoint.dataset_args)
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

    # Standardize data
    X_mean = X.mean(axis=0)
    X -= X_mean
    X_std = X.std(axis=0)
    X /= X_std

    # Slide along sequence with 3/4 overlap collecting all the data into batches.
    start = 0
    ws = manager.dataset_args.sample_duration
    step = int(ws / 4)
    Xs = []
    while start + ws < len(X):
        Xs.append(torch.from_numpy(X[start:start + ws].T))
        start += step
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
    Zs = torch.cat(Zs, dim=0)

    # Calculate pairwise distances
    logger.info('Calculating pairwise distances.')
    distances = torch.pdist(Zs, p=2)
    distances = to_numpy(distances)
    # distances_sf = squareform(distances)

    # Calculate linkage
    logger.info(f'Calculating linkage using method "{linkage_method}".')
    L = linkage(distances, linkage_method)

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
            zs = np.argwhere(clusters == cluster_idx)[:, 0]
            xs = []
            for z in zs:
                xs.extend(np.arange(z * step, (z + 1) * step))
            ys = np.ones_like(xs) * y_locs[j]
            ax.scatter(ts[xs], ys, label=cluster_idx)

        ax.set_yticks(y_locs)
        ax.grid()

    ax.set_xlabel('Time (s)')

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}. Checkpoint={checkpoint.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    title += f'\nLinkage method: {linkage_method}.'
    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_cluster_traces' \
                           f'_cp={checkpoint.id}' \
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
    traces()
