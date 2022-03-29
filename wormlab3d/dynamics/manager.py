import copy
import os
from typing import Dict, List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from wormlab3d import logger
from wormlab3d.data.model import Dataset, NetworkParameters
from wormlab3d.data.model.dataset import DatasetEigentraces
from wormlab3d.dynamics.args import DynamicsDatasetArgs, DynamicsNetworkArgs, DynamicsOptimiserArgs, DynamicsRuntimeArgs
from wormlab3d.dynamics.data_loader import get_data_loader
from wormlab3d.dynamics.dynamics_clusterer_net import DynamicsClustererNet
from wormlab3d.nn.manager import Manager as BaseManager, NetworkParametersDynamicsClusterer
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
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
        self.loss_dyn = torch.zeros(self.runtime_args.batch_size, self.net_args.n_classes)
        self.loss_com = torch.zeros(self.runtime_args.batch_size, self.net_args.n_classes)

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
        N = self.net_args.n_classes
        sample_shape = (self.dataset_args.n_components * 2, self.dataset_args.sample_duration)
        latent_shape = (N,)
        output_shape = (N,) + sample_shape

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
            input_shape=sample_shape,
            output_shape=sample_shape,
            prefix='dynamics',
            build_model=False
        )
        dynamics_nets = [dynamics_net, ]
        for i in range(1, N):
            dynamics_nets.append(copy.deepcopy(dynamics_net))
        dynamics_nets = nn.ModuleList(dynamics_nets)

        params = {**{
            'network_type': 'dynamics_clusterer',
            'input_shape': sample_shape,
            'output_shape': (output_shape, latent_shape),
            'classifier_net': classifier_net_params,
            'dynamics_net': dynamics_net_params,
            'n_classes': N
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
            dynamics_nets=dynamics_nets,
            X0_duration=self.dataset_args.X0_duration,
            build_model=True
        )
        full_net = torch.jit.script(full_net)
        logger.info(f'Instantiated full network with {full_net.get_n_params() / 1e6:.4f}M parameters.')
        # logger.debug(f'----------- Network --------------\n\n{full_net}\n\n')

        return full_net, net_params

    def _init_metrics(self) -> Tuple[Dict[str, callable], List[str]]:
        return {}, []

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

        # Calculate dynamics prediction losses between input samples X and output samples Y
        loss_dyn = torch.sum((X.unsqueeze(1) - Y)**2, dim=(2, 3))

        # Weight the errors by the latent class vector Z
        loss_com = Z * loss_dyn

        # Log losses
        for i in range(self.net_args.n_classes):
            stats[f'loss_dyn/{i}'] = loss_dyn[:, i].mean()
            stats[f'loss_com/{i}'] = loss_com[:, i].mean()
        stats[f'loss_dyn/mean'] = loss_dyn.mean()
        stats[f'loss_dyn/var'] = loss_dyn.var()

        # Sum losses over the classes and take the batch mean.
        loss = loss_com.sum(dim=-1).mean()
        stats['loss'] = loss.item()
        stats['loss/var'] = loss_com.var(dim=-1).mean()

        # Store the full losses as class variables for use in plotting
        self.loss_dyn = loss_dyn
        self.loss_com = loss_com

        assert not is_bad(loss), 'Bad loss!'
        return loss, stats

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
        Nc = self.net_args.n_classes
        n_components = self.dataset_args.n_components
        n_rows = 1 + n_components * 2
        n_examples = min(self.runtime_args.plot_n_examples, bs)
        idxs = np.random.choice(bs, n_examples, replace=False)
        ts = np.arange(self.dataset_args.sample_duration)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colours = prop_cycle.by_key()['color']
        colours = [default_colours[i] for i in range(Nc)]

        X = data
        Y, Z = outputs
        y_min = float(min(X.amin(), Y.amin()))
        y_max = float(max(X.amax(), Y.amax()))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_examples,
            figsize=(n_examples * 5, 1 + n_rows)
        )

        for i, idx in enumerate(idxs):
            Xi = to_numpy(X[idx])
            Yi = to_numpy(Y[idx])
            Zi = to_numpy(Z[idx])
            alphas = [0.3 + Zi[i] * 7 / 10 for i in range(Nc)]

            # Plot classification as bar chart
            ax = axes[0, i]
            ax.bar(np.arange(Nc), Zi, color=colours)
            ax.set_ylim(bottom=0, top=1)
            if i == 0:
                ax.set_yticks([0, 0.5])
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])

            cell_text = []
            for j in range(Nc):
                cell_text.append([
                    f'{Zi[j]:.3f}',
                    f'{float(self.loss_dyn[idx, j]):.3f}',
                    f'{float(self.loss_com[idx, j]):.3f}'
                ])

            ax.table(
                cellText=cell_text,
                rowLabels=np.arange(Nc) if i == 0 else None,
                rowColours=colours if i == 0 else None,
                colLabels=['$P(class)$', '$L_{dyn}$', '$L_{com}$'],
                loc='top',
                edges='closed'
            )

            # Plot data and dynamics outputs
            for j in range(n_components * 2):
                ax = axes[j + 1, i]
                ax.plot(ts, Xi[j], alpha=0.8, color='black', zorder=100)
                for k in range(Nc):
                    ax.plot(ts, Yi[k, j], alpha=alphas[k], color=colours[k])
                ax.set_ylim(bottom=y_min, top=y_max)

                if i == 0:
                    ax.set_ylabel(f'${"Re" if j % 2 == 0 else "Im"}(\lambda_{int(j / 2)})$')
                else:
                    ax.set_yticklabels([])

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

    def _save_plot(self, fig: Figure, plot_type: str, train_or_test: str):
        """
        Either log the figure to the tensorboard logger or save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path / 'plots' / f'{plot_type}_{train_or_test}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir / f'{self.checkpoint.step:06d}.png'
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(f'{plot_type}_{train_or_test}', fig, self.checkpoint.step)
            self.tb_logger.flush()

        plt.close(fig)
