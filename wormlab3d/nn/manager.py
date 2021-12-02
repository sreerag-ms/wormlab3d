import gc
import os
import shutil
import time
from collections import OrderedDict
from datetime import timedelta
from typing import Dict, List
from typing import Tuple

import torch
import torch.nn.functional as F
from bson.errors import InvalidId
from bson.objectid import ObjectId
from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wormlab3d import logger, LOGS_PATH
from wormlab3d.data.model.checkpoint import Checkpoint
from wormlab3d.data.model.network_parameters import *
from wormlab3d.nn.args import DatasetArgs, NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.nn.args.optimiser_args import LOSS_MSE, LOSS_KL, LOSS_BCE
from wormlab3d.nn.data_loader import load_dataset
from wormlab3d.nn.wrapped_data_parallel import WrappedDataParallel
from wormlab3d.toolkit.util import to_dict, is_bad

LOG_EVERY_N_BATCHES = 1
START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')


class Manager:
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetArgs,
            net_args: NetworkArgs,
            optimiser_args: OptimiserArgs,
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.dataset_args = dataset_args
        self.net_args = net_args
        self.optimiser_args = optimiser_args

        # Dataset and data loaders
        self.ds = self._init_dataset()
        self.train_loader, self.test_loader = self._init_data_loaders()

        # Network
        self.net, self.net_params = self._init_network()

        # Optimiser
        self.optimiser = self._init_optimiser()

        # Metrics
        self.metrics, self.metric_keys = self._init_metrics()

        # Runtime params
        self.device = self._init_devices()

        # Checkpoints
        self.checkpoint = self._init_checkpoint()

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int]:
        pass

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int]:
        pass

    @property
    def logs_path(self) -> str:
        return self.get_logs_path(self.checkpoint)

    @staticmethod
    def get_logs_path(checkpoint: Checkpoint) -> str:
        return LOGS_PATH \
               + f'/{checkpoint.dataset.created:%Y%m%d_%H:%M}_{checkpoint.dataset.id}' \
               + f'/{checkpoint.network_params.created:%Y%m%d_%H:%M}_{checkpoint.network_params.id}'

    def _init_dataset(self):
        """
        Load or create the dataset.
        """
        ds = None

        # Try to load an existing dataset
        if self.dataset_args.load:
            try:
                ds = load_dataset(self.dataset_args)
            except DoesNotExist:
                logger.info('No suitable datasets found in database.')

        # Not loaded dataset, so create one
        if ds is None:
            ds = self._generate_dataset()  # persists dataset to the database

        return ds

    @abstractmethod
    def _generate_dataset(self):
        pass

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get the data loaders.
        """
        logger.info('Initialising data loaders.')
        loaders = {}
        for tt in ['train', 'test']:
            loaders[tt] = self._get_data_loader(train_or_test=tt)

        return loaders['train'], loaders['test']

    @abstractmethod
    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        pass

    def _init_network(
            self,
            net_args: NetworkArgs = None,
            input_shape: tuple = None,
            output_shape: tuple = None,
            prefix: str = None
    ) -> Tuple[BaseNet, NetworkParameters]:
        """
        Build the network using the given parameters, defaulting to the instance attributes if not provided.
        """
        if net_args is None:
            net_args = self.net_args
        if input_shape is None:
            input_shape = self.input_shape
        if output_shape is None:
            output_shape = self.output_shape
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '-'
        logger.info(f'Initialising {prefix}network')

        net_params = None
        params = {**{
            'network_type': net_args.base_net,
            'input_shape': input_shape,
            'output_shape': output_shape,
        }, **net_args.hyperparameters}

        # Try to load an existing network
        if net_args.load:
            # If we have a net id then load this from the database
            if net_args.net_id is not None:
                net_params = NetworkParameters.objects.get(id=net_args.net_id)
            else:
                # Otherwise, try to find one matching the same parameters
                net_params_matching = NetworkParameters.objects(**params)
                if net_params_matching.count() > 0:
                    net_params = net_params_matching[0]
                    logger.info(
                        f'Found {len(net_params_matching)} suitable {prefix}networks in database, using most recent.')
                else:
                    logger.info(f'No suitable {prefix}networks found in database.')
            if net_params is not None:
                logger.info(f'Loaded {prefix}network (id={net_params.id}, created={net_params.created}).')

        # Not loaded network, so create one
        if net_params is None:
            # Separate classes are used to validate the different available hyperparameters
            if net_args.base_net == 'fcnet':
                net_params = NetworkParametersFC(**params)
            elif net_args.base_net == 'aenet':
                net_params = NetworkParametersAE(**params)
            elif net_args.base_net == 'resnet':
                net_params = NetworkParametersResNet(**params)
            elif net_args.base_net == 'resnet1d':
                net_params = NetworkParametersResNet1d(**params)
            elif net_args.base_net == 'densenet':
                net_params = NetworkParametersDenseNet(**params)
            elif net_args.base_net == 'pyramidnet':
                net_params = NetworkParametersPyramidNet(**params)
            elif net_args.base_net == 'nunet':
                net_params = NetworkParametersNuNet(**params)
            elif net_args.base_net == 'rdn':
                net_params = NetworkParametersRDN(**params)
            elif net_args.base_net == 'red':
                net_params = NetworkParametersRED(**params)
            else:
                raise ValueError(f'Unrecognised base net: {net_args.base_net}')

            # Save the network parameters to the database
            net_params.save()
            logger.info(f'Saved {prefix}net parameters to database (id={net_params.id})')

        # Instantiate the network
        net = net_params.instantiate_network()
        logger.info(f'Instantiated {prefix}network with {net.get_n_params() / 1e6:.4f}M parameters.')
        logger.debug(f'----------- {prefix}Network --------------\n\n{net}\n\n')

        return net, net_params

    def _init_optimiser(self) -> Optimizer:
        """
        Set up the optimiser.
        """
        logger.info('Initialising optimiser.')

        # Build the optimiser
        cls: Optimizer = getattr(torch.optim, self.optimiser_args.algorithm)
        optimiser = cls(
            params=self.net.parameters(),
            lr=self.optimiser_args.lr_init,
            weight_decay=self.optimiser_args.weight_decay
        )

        return optimiser

    def _init_metrics(self) -> Tuple[Dict[str, callable], List[str]]:
        """
        Set up the loss functions and any other metrics to track.
        """

        # Ensure the overall loss is included in the metrics to track
        track_metrics = self.runtime_args.track_metrics.copy()
        if self.optimiser_args.loss not in track_metrics:
            track_metrics.append(self.optimiser_args.loss)

        metrics = {}
        for loss_type in track_metrics:

            # Mean squared error / L2 loss
            if loss_type == LOSS_MSE:
                def mse(pred, target):
                    loss = F.mse_loss(pred, target, reduction='sum')
                    loss = loss / len(pred)  # return loss per-datum so different batch sizes can be compared
                    return loss

                metrics[loss_type] = mse

            # KL divergence
            elif loss_type == LOSS_KL:
                def kl(pred, target):
                    # pred = pred.clamp(max=1)
                    if pred.max() > 1:
                        pred = pred / pred.max()
                    assert pred.min() >= 0 and pred.max() <= 1, 'KL divergence requires predictions to be in (0,1) range.'
                    pred = pred.clamp(min=1e-3, max=1 - 1e-3)
                    pred_prob = torch.cat([pred, 1 - pred], dim=1)
                    pred_logprob = torch.log(pred_prob)
                    assert not is_bad(pred_logprob)
                    target = torch.cat([target, 1 - target], dim=1)
                    loss = F.kl_div(pred_logprob, target, reduction='sum')
                    loss = loss / len(pred)  # return loss per-datum so different batch sizes can be compared
                    return loss

                metrics[loss_type] = kl

            # BCE
            elif loss_type == LOSS_BCE:
                metrics[loss_type] = nn.BCELoss(reduction='sum')

            else:
                raise RuntimeError(f'Loss type: {loss_type} not recognised.')

        return metrics, list(metrics.keys())

    def _init_devices(self):
        """
        Find available devices and try to use what we want.
        """
        if self.runtime_args.gpu_only:
            cpu_or_gpu = 'gpu'
        elif self.runtime_args.cpu_only:
            cpu_or_gpu = 'cpu'
        else:
            cpu_or_gpu = None

        if cpu_or_gpu == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count()
        if device.type == 'cuda':
            if n_gpus > 1:
                logger.info(f'Using {n_gpus} GPUs!')
                self.net = WrappedDataParallel(self.net)
            else:
                logger.info('Using GPU')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
        else:
            if cpu_or_gpu == 'gpu':
                raise RuntimeError('GPU requested but not available. Aborting.')
            logger.info('Using CPU')

        return device

    def _init_checkpoint(self):
        """
        The current checkpoint instance contains the most up to date instance of the model.
        This is not persisted to the database until we actually want to checkpoint it, so should
        be thought of more as a checkpoint-buffer.
        """

        # Load previous checkpoint
        prev_checkpoint: Checkpoint = None
        if self.runtime_args.resume:
            try:
                if self.runtime_args.resume_from in ['latest', 'best']:
                    order_by = '-created' if self.runtime_args.resume_from == 'latest' else '+loss_test'
                    prev_checkpoints = Checkpoint.objects(
                        dataset=self.ds,
                        network_params=self.net_params
                    ).order_by(order_by)
                    if prev_checkpoints.count() > 0:
                        logger.info(
                            f'Found {prev_checkpoints.count()} previous checkpoints. '
                            f'Using {self.runtime_args.resume_from}.'
                        )
                        prev_checkpoint = prev_checkpoints[0]
                    else:
                        logger.error(
                            f'Found no checkpoints with dataset={self.ds.id} '
                            f'and network_params={self.net_params.id}'
                        )
                        raise DoesNotExist()
                else:
                    prev_checkpoint = None
                    try:
                        oid = ObjectId(self.runtime_args.resume_from)
                        prev_checkpoint = Checkpoint.objects.get(id=oid)
                    except (InvalidId, DoesNotExist):
                        pass

                    if prev_checkpoint is None:
                        prev_checkpoints = Checkpoint.objects(
                            dataset=self.ds,
                            network_params=self.net_params,
                            epoch=self.runtime_args.resume_from
                        ).order_by('-created')

                        if prev_checkpoints.count() == 0:
                            logger.error(f'Found no checkpoints matching id or epoch.')
                            raise DoesNotExist()
                        if len(prev_checkpoints) > 1:
                            logger.warning(f'Found multiple checkpoints matching epoch, using most recent.')
                        prev_checkpoint = prev_checkpoints[0]

                logger.info(f'Loaded checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}')
                logger.info(f'Test loss = {prev_checkpoint.loss_test:.6f}')
                for key, val in prev_checkpoint.metrics_test.items():
                    logger.info(f'\t{key}: {val:.4E}')
            except DoesNotExist:
                raise RuntimeError(f'Could not load checkpoint={self.runtime_args.resume_from}')

        # Either clone the previous checkpoint to use as the starting point
        if prev_checkpoint is not None:
            checkpoint = prev_checkpoint.clone()

            # Update the dataset and network params references to the ones now in use
            if checkpoint.dataset.id != self.ds.id:
                logger.warning('Dataset has changed! This may result in training occurring on test data!')
                checkpoint.dataset = self.ds

            # If the hyperparameters have changed, the model file might now be incompatible, but try anyway
            if checkpoint.network_params.id != self.net_params.id:
                logger.warning('Network parameters have changed! This may result in a broken network!')
                checkpoint.network_params = self.net_params

            # Args are stored against the checkpoint, so just override them
            checkpoint.dataset_args = to_dict(self.dataset_args)
            checkpoint.optimiser_args = to_dict(self.optimiser_args)
            checkpoint.runtime_args = to_dict(self.runtime_args)

            # Load the network and optimiser parameter states
            path = f'{self.get_logs_path(prev_checkpoint)}/checkpoints/{prev_checkpoint.id}.chkpt'
            state = torch.load(path, map_location=self.device)
            self.net.load_state_dict(self._fix_state(state['model_state_dict']), strict=False)
            if self.optimiser_args.algorithm != prev_checkpoint.optimiser_args['algorithm']:
                self.optimiser.step()
            self.optimiser.load_state_dict(state['optimiser_state_dict'])
            self.net.eval()
            logger.info(f'Loaded state from "{path}"')

        # ..or start a new checkpoint
        else:
            checkpoint = Checkpoint(
                dataset=self.ds,
                network_params=self.net_params,
                loss_type=self.optimiser_args.loss,
                dataset_args=to_dict(self.dataset_args),
                optimiser_args=to_dict(self.optimiser_args),
                runtime_args=to_dict(self.runtime_args),
            )

        return checkpoint

    def _fix_state(self, state):
        if isinstance(self.net, WrappedDataParallel):
            return state
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        return new_state

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.logs_path + '/events/' + START_TIMESTAMP, flush_secs=5)

    def configure_paths(self, renew_logs: bool = False):
        """Create the directories."""
        if renew_logs:
            logger.warning('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path + '/checkpoints', exist_ok=True)
        os.makedirs(self.logs_path + '/events', exist_ok=True)
        os.makedirs(self.logs_path + '/plots', exist_ok=True)

    def save_checkpoint(self):
        """
        Save the checkpoint information to the database and the network model parameters to file.
        """
        logger.info('Saving model checkpoint...')
        self.checkpoint.save()
        path = f'{self.logs_path}/checkpoints/{self.checkpoint.id}.chkpt'
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }, path)

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint
        self.checkpoint = self.checkpoint.clone()

    def log_graph(self):
        """
        Log the graph to tensorboard.
        """
        with torch.no_grad():
            dummy_input = torch.rand((self.train_loader.batch_size,) + tuple(self.net_params.input_shape))
            dummy_input = dummy_input.to(self.device)
            if not hasattr(self, 'tb_logger'):
                self._init_tb_logger()
            self.tb_logger.add_graph(self.net, [dummy_input, ], verbose=False)
            self.tb_logger.flush()

    def train(self, n_epochs):
        """
        Train the network for a number of epochs.
        """
        self.configure_paths()
        self._init_tb_logger()
        starting_epoch = self.checkpoint.epoch + 1
        final_epoch = starting_epoch + n_epochs - 1

        # Reset learning rates (todo: this properly)
        for group in self.optimiser.param_groups:
            group['lr'] = self.optimiser_args.lr_init

        if starting_epoch > 1:
            for group in self.optimiser.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = self.optimiser_args.lr_init

        # todo: lr scheduler
        milestones = [n_epochs // 2 + starting_epoch, n_epochs // (4 / 3) + starting_epoch]
        lr_scheduler = MultiStepLR(
            optimizer=self.optimiser,
            milestones=milestones,
            gamma=self.optimiser_args.lr_gamma,
            last_epoch=starting_epoch - 1 if starting_epoch > 1 else -1
        )
        lr_scheduler.last_epoch = starting_epoch - 1 if starting_epoch > 1 else -1

        for epoch in range(starting_epoch, final_epoch + 1):
            logger.info('{:-^80}'.format(' Train epoch: {} '.format(epoch)))
            self.checkpoint.epoch = epoch
            start_time = time.time()
            self.tb_logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

            # Train for an epoch
            self._train_epoch(final_epoch)
            time_per_epoch = time.time() - start_time
            seconds_left = float((final_epoch - epoch) * time_per_epoch)
            logger.info('Time per epoch: {}, Est. complete in: {}'.format(
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            lr_scheduler.step()

            # Test every n epochs
            if self.runtime_args.test_every_n_epochs > 0 \
                    and (epoch + 1) % self.runtime_args.test_every_n_epochs == 0:
                self.test()
                self.tb_logger.add_scalar(f'epoch/test/total', self.checkpoint.loss_test, epoch)
                for key, val in self.checkpoint.metrics_test.items():
                    self.tb_logger.add_scalar(f'epoch/test/{key}', val, epoch)
                    logger.info(f'Test {key}: {val:.4E}')

            # Checkpoint every n epochs
            if self.runtime_args.checkpoint_every_n_epochs > 0 \
                    and (epoch + 1) % self.runtime_args.checkpoint_every_n_epochs == 0:
                self.save_checkpoint()

    def _train_epoch(self, final_epoch):
        """
        Train for a single epoch
        """
        num_batches_per_epoch = len(self.train_loader)
        running_loss = 0.
        running_metrics = {k: 0. for k in self.metric_keys}
        epoch_loss = 0.
        epoch_metrics = {k: 0. for k in self.metric_keys}

        for i, data in enumerate(self.train_loader, 0):
            batch_outputs, batch_loss, batch_stats = self._train_batch(data)

            running_loss += batch_loss
            epoch_loss += batch_loss
            for k in self.metric_keys:
                running_metrics[k] += batch_stats[k]
                epoch_metrics[k] += batch_stats[k]

            # Log statistics every X mini-batches
            if (i + 1) % LOG_EVERY_N_BATCHES == 0:
                batches_loss_avg = running_loss / LOG_EVERY_N_BATCHES
                log_msg = f'[{self.checkpoint.epoch}/{final_epoch}][{i + 1}/{num_batches_per_epoch}]' \
                          f'\tLoss: {batches_loss_avg:.7f}'
                for k in self.metric_keys:
                    avg = running_metrics[k] / LOG_EVERY_N_BATCHES
                    log_msg += f'\t{k}: {avg:.4E}'
                logger.info(log_msg)
                running_loss = 0.
                running_metrics = {k: 0. for k in self.metric_keys}

            # Plots and checkpoints
            self._make_plots(data, batch_outputs, train_or_test='train')
            if self.runtime_args.checkpoint_every_n_batches > 0 \
                    and (i + 1) % self.runtime_args.checkpoint_every_n_batches == 0:
                self.save_checkpoint()

        gc.collect()

        # Update stats and write debug
        self.checkpoint.loss_train = float(epoch_loss) / num_batches_per_epoch
        self.checkpoint.metrics_train = epoch_metrics
        self.tb_logger.add_scalar('epoch/train/total_loss', epoch_loss, self.checkpoint.epoch)
        for key, val in epoch_metrics.items():
            self.tb_logger.add_scalar(f'epoch/train/{key}', val, self.checkpoint.epoch)
            logger.info(f'Train {key}: {val:.4E}')

        # End-of-epoch plots
        self._make_plots(data, batch_outputs, train_or_test='train', end_of_epoch=True)

    def _train_batch(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Train on a single batch of data.
        """
        self.net.train()
        outputs, loss, stats = self._process_batch(data)

        # Calculate gradients and do optimisation step
        self.optimiser.zero_grad()
        loss.backward()

        # Clip gradients
        # nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=100)

        self.optimiser.step()

        # Log losses
        self.tb_logger.add_scalar('batch/train/total_loss', loss, self.checkpoint.step)
        for key, val in stats.items():
            self.tb_logger.add_scalar(f'batch/train/{key}', val, self.checkpoint.step)

        # Calculate L2 loss
        norms = self.net.calc_norms()
        weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
        for _, norm in norms.items():
            weights_cumulative_norm += norm
        assert not is_bad(weights_cumulative_norm)
        self.tb_logger.add_scalar('batch/train/w_norm', weights_cumulative_norm.item(), self.checkpoint.step)

        # Increment global step counter
        self.checkpoint.step += 1
        self.checkpoint.examples_count += self.runtime_args.batch_size

        return outputs, loss, stats

    def test(self) -> Tuple[float, Dict]:
        """
        Test across the whole test dataset.
        """
        if not len(self.test_loader):
            raise RuntimeError('No test data available, cannot test!')
        logger.info('Testing')
        self.net.eval()
        cumulative_loss = 0.
        cumulative_stats = {k: 0. for k in self.metric_keys}

        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                batch_outputs, batch_loss, batch_stats = self._process_batch(data)
                cumulative_loss += batch_loss
                for k in self.metric_keys:
                    cumulative_stats[k] += batch_stats[k]

        test_loss = cumulative_loss / len(self.test_loader)
        test_stats = {k: cumulative_stats[k] / len(self.test_loader) for k in self.metric_keys}

        self.checkpoint.loss_test = float(test_loss)
        self.checkpoint.metrics_test = test_stats

        self._make_plots(data, batch_outputs, train_or_test='test', end_of_epoch=True)

        return test_loss, test_stats

    @abstractmethod
    def _process_batch(self, data):
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """
        pass

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Take a batch of input data and return network output.
        """
        X = X.to(self.device)
        Y_pred = self.net(X)
        return Y_pred

    def calculate_losses(self, Y_pred: torch.Tensor, Y_target: torch.tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate losses
        """
        stats = {}
        for metric, fn in self.metrics.items():
            stats[metric] = fn(Y_pred, Y_target)
            if metric == self.optimiser_args.loss:
                loss = stats[metric]
        assert not is_bad(loss), 'Bad loss!'
        return loss, stats

    @abstractmethod
    def _make_plots(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, list],
            outputs: torch.Tensor,
            train_or_test: str,
            end_of_epoch: bool = False
    ):
        """
        Generate some example plots.
        """
        pass
