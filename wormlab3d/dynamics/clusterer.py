from typing import Tuple

import numpy as np
import torch
from fastcluster import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from wormlab3d import logger
from wormlab3d.data.model import Checkpoint
from wormlab3d.dynamics.args import DynamicsNetworkArgs, DynamicsRuntimeArgs, DynamicsDatasetArgs, DynamicsOptimiserArgs
from wormlab3d.dynamics.manager import Manager
from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import to_numpy
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

linkage_methods = ['single', 'complete', 'average', 'centroid', 'median', 'ward', 'weighted']
metrics = ['canberra', 'cityblock', 'euclidean', 'hamming', 'matching', 'sqeuclidean', 'seuclidean', 'cosine',
           'correlation']


class DynamicsClusterer:
    def __init__(
            self,
            checkpoint_id: str,
            reconstruction_id: str,
            start_frame: int = None,
            end_frame: int = None,
            step: int = 25
    ):
        self.checkpoint_id = checkpoint_id
        self.reconstruction_id = reconstruction_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self._init_checkpoint()
        self._init_args()
        self._init_manager()
        self._calculate_reconstruction_embeddings()

    def _init_checkpoint(self):
        """
        Load the checkpoint.
        """
        self.checkpoint = Checkpoint.objects.get(id=self.checkpoint_id)

    def _init_args(self):
        """
        Construct args.
        """
        self.runtime_args = DynamicsRuntimeArgs(
            resume=True,
            resume_from=self.checkpoint.id,
            cpu_only=True
        )
        self.dataset_args = DynamicsDatasetArgs(**{
            **self.checkpoint.dataset_args,
            **{'load_dataset': True, 'dataset_id': self.checkpoint.dataset.id}
        })
        self.net_args = DynamicsNetworkArgs(
            load=True,
            net_id=self.checkpoint.network_params.id,
            latent_size=self.checkpoint.network_params.latent_size,
            args_classifier=NetworkArgs(net_id=self.checkpoint.network_params.classifier_net.id),
            args_dynamics=NetworkArgs(net_id=self.checkpoint.network_params.dynamics_net.id)
        )
        self.optimiser_args = DynamicsOptimiserArgs(**self.checkpoint.optimiser_args)

    def _init_manager(self):
        """
        Construct manager.
        """
        self.manager = Manager(
            runtime_args=self.runtime_args,
            dataset_args=self.dataset_args,
            net_args=self.net_args,
            optimiser_args=self.optimiser_args,
        )

    def _calculate_reconstruction_embeddings(self):
        """
        Calculate reconstruction embeddings.
        """
        common_args = {
            'reconstruction_id': self.reconstruction_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'smoothing_window': self.dataset_args.smoothing_window,
        }

        # Get natural frame trajectory
        X_nf, meta = get_trajectory(**common_args, natural_frame=True)

        # Convert to eigenworms
        X = self.manager.ew.transform(np.array(X_nf))

        # Restrict to the components we need
        X = X[:, :self.dataset_args.n_components]

        # Stack real and imaginary
        X2 = np.stack([np.real(X), np.imag(X)], axis=-1)
        X = np.zeros((len(X), self.dataset_args.n_components * 2))
        X[:, ::2] = X2[..., 0]
        X[:, 1::2] = X2[..., 1]
        logger.info(f'Trajectory trace shape: {X.shape}.')

        # Standardize data if required
        if self.dataset_args.standardise:
            X_mean = X.mean(axis=0)
            X -= X_mean
            X_std = X.std(axis=0)
            X /= X_std

        # Include nonplanarity
        if self.dataset_args.include_np:
            logger.info('Fetching planarities.')
            pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
            r = pcas.explained_variance_ratio.T
            nonp = r[2] / np.sqrt(r[1] * r[0])

            # Nonplanarity goes from [0,1] so standardise with data-independent scaling to [-1,1]
            if self.dataset_args.standardise:
                nonp = nonp * 2 - 1
            X = np.concatenate([nonp[:, None], X], axis=1)

        # Include speed
        if self.dataset_args.include_speed:
            logger.info('Calculating speeds.')
            X_traj, meta = get_trajectory(**common_args)
            speeds = calculate_speeds(X_traj, signed=True)

            # Standardise with data-independent scaling factor of 100
            if self.dataset_args.standardise:
                speeds *= 100
            X = np.concatenate([speeds[:, None], X], axis=1)

        # Slide along sequence with 3/4 overlap collecting all the data into batches.
        start = 0
        ws = self.manager.dataset_args.sample_duration
        Xs = []
        while start + ws < len(X):
            Xs.append(torch.from_numpy(X[start:start + ws].T))
            start += self.step
        Xs = torch.stack(Xs).to(torch.float32).to(self.manager.device)
        batches = Xs.split(self.manager.runtime_args.batch_size)
        logger.info(f'{len(batches)} batches generated from reconstruction.')

        # Run the batches through the classifier network to get the latent encodings.
        logger.info(f'Generating encodings.')
        self.manager.net.eval()
        Zs = []
        for batch in batches:
            Z = self.manager.net.classifier_net.forward(batch)
            Zs.append(Z)
        self.Zs = to_numpy(torch.cat(Zs, dim=0))

    def cluster(self, distance_metric: str, linkage_method: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pairwise distances and linkage.
        """
        logger.info(f'Calculating pairwise distances using metric "{distance_metric}".')
        distances = pdist(self.Zs, distance_metric)
        distances = distances

        # Calculate linkage
        logger.info(f'Calculating linkage using method "{linkage_method}".')
        L = linkage(distances, linkage_method)

        return L, distances

    def find_best_linkage_settings(self):
        """
        Find best linkage method using cophenetic correlation coefficient.
        """
        best_score = 0
        best_method = 'average'
        best_metric = 'euclidean'
        for method in linkage_methods:
            for metric in metrics:
                try:
                    Z = linkage(self.Zs, method, metric)
                    c, coph_dists = cophenet(Z, pdist(self.Zs, metric))
                except Exception:
                    continue
                logger.info('Method = {}. Metric = {}. Score = {}'.format(method, metric, c))
                if c > best_score:
                    best_score = c
                    best_method = method
                    best_metric = metric
        logger.info(
            'Best combination: Method = {}. Metric = {}. Score = {}'.format(best_method, best_metric, best_score))
        return best_method, best_metric
