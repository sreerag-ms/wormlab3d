import os
from typing import Dict, List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastcluster import linkage
from matplotlib import gridspec
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from torch.utils.data import DataLoader

from wormlab3d import logger
from wormlab3d.data.model import Dataset, NetworkParameters, Reconstruction
from wormlab3d.data.model.dataset import DatasetEigentraces
from wormlab3d.dynamics.args import DynamicsDatasetArgs, DynamicsNetworkArgs, DynamicsOptimiserArgs, DynamicsRuntimeArgs
from wormlab3d.dynamics.data_loader import get_data_loader
from wormlab3d.dynamics.dynamics_clusterer_net import DynamicsClustererNet
from wormlab3d.nn.manager import Manager as BaseManager, NetworkParametersDynamicsClusterer
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.plot_utils import fancy_dendrogram, plot_reordered_distances
from wormlab3d.toolkit.util import is_bad, to_numpy
from wormlab3d.trajectories.cache import get_trajectory


# torch.autograd.set_detect_anomaly(True)


class Manager(BaseManager):
    runtime_args: DynamicsRuntimeArgs
    dataset_args: DynamicsDatasetArgs
    net_args: DynamicsNetworkArgs
    optimiser_args: DynamicsOptimiserArgs

    def __init__(
            self,
            runtime_args: DynamicsRuntimeArgs,
            dataset_args: DynamicsDatasetArgs,
            net_args: DynamicsNetworkArgs,
            optimiser_args: DynamicsOptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

    @property
    def input_shape(self) -> Tuple[int]:
        """
        Number of components * 2 (real and imaginary) x Sample duration.
        """
        return self.dataset_args.n_components * 2, self.dataset_args.sample_duration

    @property
    def output_shape(self) -> Tuple[int]:
        """
        Number of components * 2 (real and imaginary) x Sample duration.
        """
        return self.dataset_args.n_components * 2, self.dataset_args.sample_duration

    def _init_dataset(self) -> DatasetEigentraces:
        """
        Load the dataset.
        """
        logger.info('Initialising dataset.')
        args = self.dataset_args

        # Load the eigenworms
        self.ew = generate_or_load_eigenworms(
            eigenworms_id=args.eigenworms,
            dataset_id=args.dataset_m3d,
            reconstruction_id=args.reconstruction,
            regenerate=False
        )
        args.eigenworms = self.ew.id  # Update the args value so it gets set in the database.

        return super()._init_dataset()

    def _generate_dataset(self) -> DatasetEigentraces:
        """
        Generate a new dataset.
        """
        logger.info('Generating dataset.')
        args = self.dataset_args

        # Fetch reconstruction ids
        if args.reconstruction is not None:
            assert args.dataset_m3d is None, 'Can only specify one of reconstruction or dataset-m3d.'
            reconstruction_ids = [args.reconstruction, ]
        else:
            assert args.dataset_m3d is not None, 'Must specify one of reconstruction or dataset-m3d.'
            ds = Dataset.objects.get(id=args.dataset_m3d)
            reconstruction_ids = ds.reconstructions

        # Convert to eigen-traces
        X_train = []
        X_test = []
        for reconstruction_id in reconstruction_ids:
            logger.info(f'Generating eigentrace for reconstruction = {reconstruction_id}.')

            # Get natural frame trajectory
            Z, meta = get_trajectory(
                reconstruction_id=reconstruction_id,
                smoothing_window=args.smoothing_window,
                natural_frame=True,
                rebuild_cache=False
            )

            # Convert to eigenworms
            X = self.ew.transform(np.array(Z))

            # Get split points
            T = X.shape[0]
            T_train = int(T * args.train_test_split)

            # Restrict to the components we need
            X = X[:, :args.n_components]

            # Stack real and imaginary
            X2 = np.stack([np.real(X), np.imag(X)], axis=-1)
            X = np.zeros((T, args.n_components * 2))
            X[:, ::2] = X2[..., 0]
            X[:, 1::2] = X2[..., 1]
            logger.info(f'Trajectory trace shape: {X.shape}.')

            # Standardize data
            X_mean = X[0:T_train, :].mean(axis=0)
            X -= X_mean
            X_std = X[0:T_train, :].std(axis=0)
            X /= X_std

            # Split
            X_train.append(X[:T_train])
            X_test.append(X[T_train:])

        logger.debug('Saving dataset.')
        ds = DatasetEigentraces.from_args(args)
        ds.set_data(train=X_train, test=X_test, metas={'reconstruction_ids': reconstruction_ids})
        ds.save()

        return ds

    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        return get_data_loader(
            ds=self.ds,
            ds_args=self.dataset_args,
            train_or_test=train_or_test,
            batch_size=self.runtime_args.batch_size
        )

    def _init_network(self) -> Tuple[DynamicsClustererNet, NetworkParametersDynamicsClusterer]:
        """
        Initialise the network which consists of 2 subnetworks.
        """
        latent_size = self.net_args.latent_size
        sample_shape = (self.dataset_args.n_components * 2, self.dataset_args.sample_duration)
        latent_shape = (latent_size,)
        output_shape = (latent_size,) + sample_shape

        # Initialise classifier network
        classifier_net, classifier_net_params = super()._init_network(
            net_args=self.net_args.args_classifier,
            input_shape=sample_shape,
            output_shape=latent_shape,
            prefix='classifier',
            build_model=False
        )

        # Initialise the dynamics networks - 1 per class
        dynamics_net, dynamics_net_params = super()._init_network(
            net_args=self.net_args.args_dynamics,
            input_shape=(self.dataset_args.n_components * 2, latent_size),
            output_shape=sample_shape,
            prefix='dynamics',
            build_model=False
        )

        params = {**{
            'network_type': 'dynamics_clusterer',
            'input_shape': sample_shape,
            'output_shape': (output_shape, latent_shape),
            'classifier_net': classifier_net_params,
            'dynamics_net': dynamics_net_params,
            'latent_size': latent_size
        }}

        # Try to load an existing network
        net_params = None
        if self.net_args.load:
            # If we have a net id then load this from the database
            if self.net_args.net_id is not None:
                net_params = NetworkParameters.objects.get(id=self.net_args.net_id)
            else:
                # Otherwise, try to find one matching the same parameters
                net_params_matching = NetworkParameters.objects(**params)
                if net_params_matching.count() > 0:
                    net_params = net_params_matching[0]
                    logger.info(f'Found {len(net_params_matching)} suitable networks in database, using most recent.')
                else:
                    logger.info(f'No suitable networks found in database.')
            if net_params is not None:
                logger.info(f'Loaded network (id={net_params.id}, created={net_params.created}).')

        # Not loaded network, so create one
        if net_params is None:
            net_params = NetworkParametersDynamicsClusterer(**params)
            net_params.save()
            logger.info(f'Saved network id={net_params.id}.')

        # Instantiate the network
        full_net = DynamicsClustererNet(
            input_shape=sample_shape,
            output_shape=output_shape,
            classifier_net=classifier_net,
            dynamics_net=dynamics_net,
            X0_duration=self.dataset_args.X0_duration,
            build_model=True
        )
        full_net = torch.jit.script(full_net)
        logger.info(f'Instantiated full network with {full_net.get_n_params() / 1e6:.4f}M parameters.')
        # logger.debug(f'----------- Network --------------\n\n{full_net}\n\n')

        return full_net, net_params

    def _init_metrics(self) -> Tuple[Dict[str, callable], List[str]]:
        return {}, []

    def _train_epoch(self, final_epoch: int):
        """
        Cluster the latent representations every n epochs.
        """
        super()._train_epoch(final_epoch)

        # Checkpoint every n epochs
        if self.runtime_args.cluster_every_n_epochs > 0 \
                and self.checkpoint.epoch > 0 \
                and self.checkpoint.epoch % self.runtime_args.cluster_every_n_epochs == 0:
            self._cluster()

    def _process_batch(self, data: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """
        # Put batch on the right device
        X = data.to(self.device)

        # Push input data through net
        Y, Z = self.predict(X)
        loss, metrics = self.calculate_losses(X, Y, Z)

        return (Y, Z), loss, metrics

    def calculate_losses(self, X: torch.Tensor, Y: torch.tensor, Z: torch.tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate losses.
        """
        stats = {}

        # Calculate dynamics prediction losses between input samples X and output samples Y.
        loss_dyn = torch.sum((X - Y)**2, dim=(1, 2))

        # Add a L2 loss to the latent vector to encourage sparsity
        loss_Z = torch.sum(Z**2, dim=-1)

        # Log losses
        stats[f'loss_dyn/mean'] = loss_dyn.mean()
        stats[f'loss_dyn/var'] = loss_dyn.var()
        stats[f'loss_Z/mean'] = loss_Z.mean()
        stats[f'loss_Z/var'] = loss_Z.var()

        # Sum losses and take the batch mean.
        loss = loss_dyn.mean() + loss_Z.mean()
        stats['loss'] = loss.item()

        assert not is_bad(loss), 'Bad loss!'
        return loss, stats

    def _cluster(self):
        """
        Fetch some latent space representations and cluster.
        """
        logger.info('Clustering.')
        ra = self.runtime_args

        # Pick a random sequence
        reconstruction_ids = self.ds.metas['reconstruction_ids']
        idx = np.random.randint(len(reconstruction_ids))
        sequence = np.concatenate([self.ds.X_train[idx], self.ds.X_test[idx]])
        reconstruction = Reconstruction.objects.get(id=reconstruction_ids[idx])

        # Slide along sequence with 3/4 overlap collecting all the data into batches.
        start = 0
        ws = self.dataset_args.sample_duration
        step = int(ws / 4)
        Xs = []
        while start + ws < len(sequence):
            Xs.append(torch.from_numpy(sequence[start:start + ws].T))
            start += step
        Xs = torch.stack(Xs).to(torch.float32).to(self.device)
        batches = Xs.split(ra.batch_size)
        logger.info(f'{len(batches)} batches generated from reconstruction {reconstruction.id}.')

        # Run the batches through the classifier network to get the latent encodings.
        logger.info(f'Generating encodings.')
        self.net.eval()
        Zs = []
        for X in batches:
            Z = self.net.classifier_net.forward(X)
            Zs.append(Z)
        Zs = torch.cat(Zs, dim=0)

        # Calculate pairwise distances
        logger.info('Calculating pairwise distances.')
        distances = torch.pdist(Zs, p=2)
        distances = to_numpy(distances)
        distances_sf = squareform(distances)

        # Generate the clusters.
        for linkage_method in ra.linkage_methods:
            logger.info(f'Calculating linkage using method "{linkage_method}".')
            L = linkage(distances, linkage_method)

            # Set up plots
            n_cluster_plots = ra.max_clusters - ra.min_clusters + 1
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
            cluster_nums = list(range(ra.min_clusters, ra.max_clusters + 1))
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

            fig.suptitle(f'Reconstruction: {reconstruction.id}. Linkage method: {linkage_method}.')
            fig.tight_layout()
            self._save_plot(fig, f'clustering_{linkage_method}')

    def _make_plots(
            self,
            data: torch.Tensor,
            outputs: Tuple[torch.Tensor, torch.Tensor],
            train_or_test: str,
            end_of_epoch: bool = False
    ):
        """
        Generate some example plots.
        """
        if self.runtime_args.plot_n_examples > 0 and (
                end_of_epoch or
                (self.runtime_args.plot_every_n_batches > -1
                 and self.checkpoint.step > 0
                 and self.checkpoint.step % self.runtime_args.plot_every_n_batches == 0)
        ):
            logger.info('Plotting.')
            self._plot_samples(data, outputs, train_or_test)

    def _plot_samples(
            self,
            data: torch.Tensor,
            outputs: Tuple[torch.Tensor, torch.Tensor],
            train_or_test: str
    ):
        """
        Plot some samples.
        """
        bs = self.runtime_args.batch_size
        n_components = self.dataset_args.n_components
        n_rows = n_components * 2
        n_examples = min(self.runtime_args.plot_n_examples, bs)
        idxs = np.random.choice(bs, n_examples, replace=False)
        ts = np.arange(self.dataset_args.sample_duration)

        X = data
        Y, Z = outputs

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_examples,
            figsize=(n_examples * 5, 1 + n_rows)
        )

        for i, idx in enumerate(idxs):
            Xi = to_numpy(X[idx])
            Yi = to_numpy(Y[idx])

            # Plot data and dynamics outputs
            for j in range(n_components * 2):
                ax = axes[j, i]
                ax.plot(ts, Xi[j], alpha=0.8, color='black', linewidth=2, linestyle='--', zorder=100)
                ax.plot(ts, Yi[j], alpha=0.8, color='green')

                if i == 0:
                    ax.set_ylabel(f'${"Re" if j % 2 == 0 else "Im"}(\lambda_{int(j / 2)})$')
                if j == n_components * 2 - 1:
                    ax.set_xlabel('Frame')
                else:
                    ax.set_xticklabels([])

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}'
        )
        fig.tight_layout()
        self._save_plot(fig, 'samples', train_or_test)

    def _save_plot(self, fig: Figure, plot_type: str, train_or_test: str = None):
        """
        Either log the figure to the tensorboard logger or save it to disk.
        """
        suffix = f'_{train_or_test}' if train_or_test is not None else ''
        if self.runtime_args.save_plots:
            save_dir = self.logs_path / 'plots' / f'{plot_type}{suffix}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir / f'{self.checkpoint.step:06d}.png'
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(f'{plot_type}{suffix}', fig, self.checkpoint.step)
            self.tb_logger.flush()

        plt.close(fig)
