import os
import shutil
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from wormlab3d import logger, LOGS_PATH
from wormlab3d.data.data_helpers import get_data_loaders
from wormlab3d.nn.models.basenet import BaseNet

START_TIMESTAMP = time.strftime('%Y-%m-%d_%H%M%S')
LOG_EVERY_N_BATCHES = 1
CHECKPOINT_EVERY_N_BATCHES = 1
PLOT_EVERY_N_BATCHES = 1


class Manager(ABC):
    def __init__(
            self,
            net: BaseNet,
            data_type: str,
            batch_size: int,
            rebuild_dataset=False,
            n_frames=10,
            frame_shift=5,
            n_cpca_components=2,
            restrict_classes=None,
            include_mirrors=False,
            inv_opt_params={},
            train_test_split=0.8,
            augment=False,
            n_dataloader_workers=4,
            optimiser_params=None,
            # init_learning_rate=0.1,
            # learning_rate_update=None,
            # lr_gamma=0.5,
            # sgd_momentum=0.9,
            # weight_decay=1e-4,
            cpu_or_gpu=None,
    ):
        self.net = net
        self.data_type = data_type
        self.augment = augment
        # self.init_learning_rate = init_learning_rate
        # self.learning_rate_update = learning_rate_update
        # self.lr_gamma = lr_gamma
        # self.sgd_momentum = sgd_momentum
        # self.weight_decay = weight_decay
        self.n_frames = n_frames
        self.batch_size = batch_size

        self.ds, self.train_loader, self.test_loader = get_data_loaders(
            data_type,
            batch_size,
            rebuild_dataset=rebuild_dataset,
            n_frames=n_frames,
            frame_shift=frame_shift,
            n_cpca_components=n_cpca_components,
            restrict_classes=restrict_classes,
            include_mirrors=include_mirrors,
            inv_opt_params=inv_opt_params,
            train_test_split=train_test_split,
            augment=augment,
            n_workers=n_dataloader_workers,
        )

        self.starting_epoch = 1
        self.global_step = 0
        self.best_loss = 1.e10
        self.best_stats = {}

        self._build_optimiser()
        self._determine_devices(cpu_or_gpu=cpu_or_gpu)

    @property
    def logs_path(self) -> str:
        return LOGS_PATH + f'/{self.net.id}/{self.ds.id}'

    @property
    @abstractmethod
    def stat_keys(self) -> List[str]:
        """Define the loss keys to track."""
        pass

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.logs_path + '/events', flush_secs=5)

    def _determine_devices(self, cpu_or_gpu: str = None):
        """Find available devices and use what is desired"""
        if cpu_or_gpu == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count()
        if self.device.type == 'cuda':
            if n_gpus > 1:
                logger.info('Using {} GPUs!'.format(n_gpus))
                self.net.multi_gpu_mode()
            else:
                logger.info('Using GPU')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, nn.Module):
                    v.to(self.device)
        else:
            if cpu_or_gpu == 'gpu':
                raise RuntimeError('GPU requested but not available. Aborting.')
            logger.info('Using CPU')

    @abstractmethod
    def _build_optimiser(self):
        """Set up the losses and optimiser."""
        pass

    def configure_paths(self, renew_logs: bool = False):
        """Create the directories."""
        if renew_logs:
            logger.info('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path + '/checkpoints', exist_ok=True)
        os.makedirs(self.logs_path + '/events', exist_ok=True)
        os.makedirs(self.logs_path + '/plots', exist_ok=True)

    def save_checkpoint(self, epoch: int, loss: float, stats: dict, filename: str = 'model'):
        """Save the network model to a checkpoint file."""
        logger.info('Saving model checkpoint...')
        checkpoints_path = self.logs_path + f'/checkpoints/{filename}.chkpt'
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'stats': stats,
            'model_state_dict': self.net.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }, checkpoints_path)

    def restore_checkpoint(self):
        """Restore from a checkpoint."""
        checkpoints_path = self.logs_path + '/checkpoints/model.chkpt'
        checkpoint = torch.load(checkpoints_path, map_location=self.device)
        self.starting_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step'] + 1
        self.best_loss = checkpoint['loss']
        self.best_stats = checkpoint['stats']
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.net.eval()
        logger.info(f'Loaded checkpoint file "{checkpoints_path}"')
        logger.info(f'Starting epoch = {self.starting_epoch}')
        if self.best_loss is not None:
            logger.info(f'Current loss = {self.best_loss:.5f}')
        for key, val in self.best_stats.items():
            logger.info(f'\t{key}: {val:.4E}')

    def train(self, n_epochs):
        """Train the network for a number of epochs."""
        self._init_tb_logger()  # need to call this here in case paths have changed
        final_epoch = self.starting_epoch + n_epochs - 1

        # todo: lr scheduler
        milestones = [n_epochs // 2, n_epochs // (4 / 3)]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=milestones, gamma=self.lr_gamma)

        for epoch in range(self.starting_epoch, final_epoch + 1):
            logger.info('{:-^80}'.format(' Train epoch: {} '.format(epoch)))
            start_time = time.time()
            self.tb_logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

            self._train_epoch(epoch, final_epoch)
            time_per_epoch = time.time() - start_time
            seconds_left = float((final_epoch - epoch) * time_per_epoch)
            logger.info('Time per epoch: {}, Est. complete in: {}'.format(
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            lr_scheduler.step()

            # Test every epoch
            test_loss, test_stats = self.test(epoch)
            self.tb_logger.add_scalar(f'epoch/test/total', test_loss, epoch)
            for key, val in test_stats.items():
                self.tb_logger.add_scalar(f'epoch/test/{key}', val, epoch)
                logger.info(f'Test {key}: {val:.4E}')

            if test_loss < self.best_loss:
                self.save_checkpoint(epoch, test_stats=test_stats)
                self.best_stats = test_stats

    def _train_epoch(self, epoch, final_epoch):
        """Train for a single epoch"""
        num_batches_per_epoch = len(self.train_loader)
        running_loss = 0.
        running_stats = {k: 0. for k in self.stat_keys}
        epoch_loss = 0.
        epoch_stats = {k: 0. for k in self.stat_keys}

        for i, data in enumerate(self.train_loader, 0):
            batch_loss, batch_stats = self._train_batch(data)

            running_loss += batch_loss
            epoch_loss += batch_loss
            for k in self.stat_keys:
                running_stats[k] += batch_stats[k]
                epoch_stats[k] += batch_stats[k]

            # Log statistics every X mini-batches
            if (i + 1) % LOG_EVERY_N_BATCHES == 0:
                batches_loss_avg = running_loss / LOG_EVERY_N_BATCHES
                log_msg = f'[{epoch}/{final_epoch}][{i + 1}/{num_batches_per_epoch}]' \
                          f'\t\tLoss: {batches_loss_avg:.7f}'
                for k in self.stat_keys:
                    avg = running_stats[k] / LOG_EVERY_N_BATCHES
                    log_msg += f'\t\t{k}: {avg:.4E}'
                logger.info(log_msg)
                running_loss = 0.
                running_stats = {k: 0. for k in self.stat_keys}

            # Plot
            self._make_plots(
                data,
                train_or_test='train',
                epoch=epoch,
                batch_idx=i
            )

            # Checkpoint
            if (i + 1) % CHECKPOINT_EVERY_N_BATCHES == 0:
                self.save_checkpoint(epoch, loss=batch_loss, stats=batch_stats)

        # Write debug
        self.tb_logger.add_scalar('epoch/train/total_loss', epoch_loss, epoch)
        for key, val in epoch_stats.items():
            self.tb_logger.add_scalar(f'epoch/train/{key}', val, epoch)
            logger.info(f'Train {key}: {val:.4E}')

    def _train_batch(self, data) -> Tuple[float, Dict]:
        """Train for a single batch of data."""
        self.net.train()
        batch_loss, batch_stats = self._process_batch(data, training=True)

        # Calculate gradients and do optimisation step
        self.optimiser.zero_grad()
        batch_loss.backward()
        self.optimiser.step()

        # Log losses
        self.tb_logger.add_scalar('batch/train/total_loss', batch_loss, self.global_step)
        for key, val in batch_stats.items():
            self.tb_logger.add_scalar(f'batch/train/{key}', val, self.global_step)

        # Calculate L2 loss
        norms = self.net.calc_norms()
        weights_cumulative_norm = 0.
        for _, norm in norms.items():
            weights_cumulative_norm += norm
        self.logger.add_scalar('batch/train/w_norm', weights_cumulative_norm.item(), self.global_step)

        # Increment global step counter
        self.global_step += 1

        return batch_loss, batch_stats

    def test(self) -> Tuple[float, Dict]:
        """Test across the whole test dataset"""
        if not len(self.test_loader):
            raise RuntimeError('No test data available, cannot test!')
        self.net.eval()
        cumulative_loss = 0.
        cumulative_stats = {k: 0. for k in self.stat_keys}

        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                batch_loss, batch_stats = self._process_batch(data, training=False)
                cumulative_loss += batch_loss
                for k in self.stat_keys:
                    cumulative_stats[k] += batch_stats[k]

        test_loss = cumulative_loss / len(self.test_loader)
        test_stats = {cumulative_stats[k] / len(self.test_loader) for k in self.stat_keys}

        return test_loss, test_stats

    @abstractmethod
    def _process_batch(self, data, training: bool=False, **kwargs):
        pass
