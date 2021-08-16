from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from bson import ObjectId
from matplotlib import gridspec
from mongoengine import DoesNotExist
from torch import Tensor, autograd
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE, CAMERA_IDXS, logger
from wormlab3d.data.model import SegmentationMasks, NetworkParameters
from wormlab3d.data.model.network_parameters import NetworkParametersRotAE
from wormlab3d.midlines2d.args import DatasetMidline2DCoordsArgs
from wormlab3d.midlines2d.data_loader_coords import get_data_loader as get_data_loader_coords
from wormlab3d.midlines2d.generate_midline2d_dataset import generate_midline2d_dataset
from wormlab3d.midlines3d.args import *
from wormlab3d.midlines3d.data_loader import get_data_loader as get_data_loader_masks
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.midlines3d.rotae_net import RotAENet
from wormlab3d.nn.args import RuntimeArgs
from wormlab3d.nn.data_loader import load_dataset
from wormlab3d.nn.ema import EMA
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.nn.wrapped_data_parallel import WrappedDataParallel
from wormlab3d.toolkit.util import to_numpy

D0_EMA_RATE = 0.9
D2D_EMA_RATE = 0.9
D3D_EMA_RATE = 0.9


class ManagerRotAE(BaseManager):
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetRotAECoordsArgs,
            net_args: RotAENetworkArgs,
            optimiser_args: RotAEOptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

        # Register exponential moving averages
        ema = EMA()
        ema.register(f'd0/acc.{D0_EMA_RATE}', decay=D0_EMA_RATE, val=0.5)
        ema.register(f'd0/acc_diff.{D0_EMA_RATE}', decay=D0_EMA_RATE, val=1.)
        ema.register(f'd0/bias2.{D0_EMA_RATE}', decay=D0_EMA_RATE, val=0.5)
        ema.register(f'd2d/acc.{D2D_EMA_RATE}', decay=D2D_EMA_RATE, val=0.5)
        ema.register(f'd2d/acc_diff.{D2D_EMA_RATE}', decay=D2D_EMA_RATE, val=1.)
        ema.register(f'd2d/bias2.{D2D_EMA_RATE}', decay=D2D_EMA_RATE, val=0.5)
        ema.register(f'd3d/acc.{D3D_EMA_RATE}', decay=D3D_EMA_RATE, val=0.5)
        ema.register(f'd3d/acc_diff.{D3D_EMA_RATE}', decay=D3D_EMA_RATE, val=1.)
        ema.register(f'd3d/bias2.{D3D_EMA_RATE}', decay=D3D_EMA_RATE, val=0.5)
        self.ema = ema

    @property
    def input_shape(self) -> Tuple[int]:
        """A triplet of segmentation masks, one from each camera."""
        return (3,) + PREPARED_IMAGE_SIZE

    @property
    def output_shape(self) -> Tuple[int]:
        """A triplet of projected coordinate renderings, one from each camera."""
        return (3,) + PREPARED_IMAGE_SIZE

    def _init_dataset(self):
        """
        Load or create the dataset.
        """
        ds = super()._init_dataset()

        # 2D and 3D midlines datasets
        self.dataset_2d_args = DatasetMidline2DCoordsArgs(**{**vars(self.dataset_args), 'centre_3d_max_error': 0.})
        # ds_3d_args = DatasetMidline2DCoordsArgs(**vars(self.dataset_args))
        ds_2d = None
        ds_3d = None
        if self.dataset_args.load:
            try:
                ds_2d = load_dataset(self.dataset_2d_args)
            except DoesNotExist:
                logger.info('No suitable 2D midline datasets found in database.')

        # Not loaded 2D midline dataset, so create one
        if ds_2d is None:
            ds_2d = generate_midline2d_dataset(self.dataset_2d_args)

        # todo: 3D midline dataset

        self.ds_2d = ds_2d
        self.ds_3d = ds_3d

        return ds

    def _generate_dataset(self):
        return generate_masks_dataset(self.dataset_args)

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get the data loaders.
        """
        logger.info('Initialising segmentation masks data loaders.')
        loaders = {}
        for tt in ['train', 'test']:
            loaders[tt] = get_data_loader_masks(
                ds=self.ds,
                ds_args=self.dataset_args,
                train_or_test=tt,
                batch_size=self.runtime_args.batch_size
            )

        logger.info('Initialising 2D midlines data loaders.')
        self.loader_2d = get_data_loader_coords(
            ds=self.ds_2d,
            ds_args=self.dataset_2d_args,
            train_or_test='train',
            batch_size=self.runtime_args.batch_size * 3,
            include_images=False
        )
        self.loader_2d_iterator = iter(self.loader_2d)

        # todo: loader3d

        return loaders['train'], loaders['test']

    def _init_network(self) -> Tuple[BaseNet, NetworkParameters]:
        """
        Initialise the network which includes 2-4 subnetworks.
        """
        masks_shape = (3,) + PREPARED_IMAGE_SIZE
        n_supplements = N_CAM_COEFFICIENTS * 3 + 3 + 2 * 3
        c2d_input_shape = (3 + n_supplements,) + PREPARED_IMAGE_SIZE
        c2d_output_shape = (3 + n_supplements, 2, self.dataset_args.n_worm_points)
        c3d_input_shape = (3 * 2 + n_supplements, self.dataset_args.n_worm_points)
        c3d_output_shape = (3, self.dataset_args.n_worm_points)
        d0_input_shape = (6,) + PREPARED_IMAGE_SIZE
        d2d_input_shape = (2, self.dataset_args.n_worm_points)
        d_output_shape = (1,)
        d3d_input_shape = (self.dataset_args.n_worm_points, 2)

        # Initialise networks
        c2d_net, c2d_net_params = super()._init_network(
            net_args=self.net_args.args_c2d,
            input_shape=c2d_input_shape,
            output_shape=c2d_output_shape,
            prefix='c2d'
        )
        c3d_net, c3d_net_params = super()._init_network(
            net_args=self.net_args.args_c3d,
            input_shape=c3d_input_shape,
            output_shape=c3d_output_shape,
            prefix='c3d'
        )

        if self.net_args.use_d0:
            d0_net, d0_net_params = super()._init_network(
                net_args=self.net_args.args_d0,
                input_shape=d0_input_shape,
                output_shape=d_output_shape,
                prefix='d0'
            )
        else:
            d0_net, d0_net_params = None, None

        if self.net_args.use_d2d:
            d2d_net, d2d_net_params = super()._init_network(
                net_args=self.net_args.args_d2d,
                input_shape=d2d_input_shape,
                output_shape=d_output_shape,
                prefix='d2d'
            )
        else:
            d2d_net, d2d_net_params = None, None

        if self.net_args.use_d3d:
            d3d_net, d3d_net_params = super()._init_network(
                net_args=self.net_args.args_d3d,
                input_shape=d3d_input_shape,
                output_shape=d_output_shape,
                prefix='d3d'
            )
        else:
            d3d_net, d3d_net_params = None, None

        params = {**{
            'network_type': 'rotae',
            'c2d_net': c2d_net_params,
            'c3d_net': c3d_net_params,
            'd0_net': d0_net_params,
            'd2d_net': d2d_net_params,
            'd3d_net': d3d_net_params,
            'input_shape': (masks_shape, n_supplements),
            'output_shape': (masks_shape, n_supplements, c3d_output_shape),
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
            net_params = NetworkParametersRotAE(**params)
            net_params.save()

        # Fetch the camera coefficients mean and range for normalisation
        cam_coeffs_mean, cam_coeffs_range = self.ds.get_camera_coeffs_range()
        p3d_mean, p3d_range = self.ds.get_points_3d_range()
        p2d_mean, p2d_range = self.ds.get_points_2d_range()

        # Instantiate the network
        full_net = RotAENet(
            c2d_net=c2d_net,
            c3d_net=c3d_net,
            n_worm_points=self.dataset_args.n_worm_points,
            cam_coeffs_mean=torch.from_numpy(cam_coeffs_mean).to(torch.float32),
            cam_coeffs_range=torch.from_numpy(cam_coeffs_range).to(torch.float32),
            p3d_mean=torch.from_numpy(p3d_mean).to(torch.float32),
            p3d_range=torch.from_numpy(p3d_range).to(torch.float32),
            p2d_mean=torch.from_numpy(p2d_mean).to(torch.float32),
            p2d_range=torch.from_numpy(p2d_range).to(torch.float32),
            blur_sigma=self.optimiser_args.renderer_blur_sigma,
            max_rotation=self.optimiser_args.max_rotation,
            distorted_cameras=True,
        )

        # Discriminators
        self.d0_net = d0_net
        self.d2d_net = d2d_net
        self.d3d_net = d3d_net

        return full_net, net_params

    def _init_optimiser(self) -> Optimizer:
        """
        Set up the optimiser.
        """
        logger.info('Initialising optimisers.')

        # Build the optimiser for main network
        cls: Optimizer = getattr(torch.optim, self.optimiser_args.algorithm)
        optimiser = cls(
            params=self.net.parameters(),
            lr=self.optimiser_args.lr_init,
            weight_decay=self.optimiser_args.weight_decay
        )

        # Build the optimisers for the discriminators
        if self.net_args.use_d0:
            self.optimiser_d0 = cls(
                params=self.d0_net.parameters(),
                lr=self.optimiser_args.lr_init,
                weight_decay=self.optimiser_args.weight_decay
            )
        if self.net_args.use_d2d:
            self.optimiser_d2d = cls(
                params=self.d2d_net.parameters(),
                lr=self.optimiser_args.lr_init,
                weight_decay=self.optimiser_args.weight_decay
            )
        if self.net_args.use_d3d:
            self.optimiser_d3d = cls(
                params=self.d3d_net.parameters(),
                lr=self.optimiser_args.lr_init,
                weight_decay=self.optimiser_args.weight_decay
            )

        return optimiser

    def _init_metrics(self) -> Tuple[Dict[str, callable], List[str]]:
        return {}, []

    def _init_devices(self):
        """
        Find available devices and try to use what we want.
        """
        device = super()._init_devices()
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            if self.net_args.use_d0:
                self.d0_net = WrappedDataParallel(self.d0_net)
            if self.net_args.use_d2d:
                self.d2d_net = WrappedDataParallel(self.d2d_net)
            if self.net_args.use_d3d:
                self.d3d_net = WrappedDataParallel(self.d3d_net)

        return device

    def _init_checkpoint(self):
        checkpoint = super()._init_checkpoint()
        if not self.net_args.use_d2d:
            return checkpoint

        # If checkpoint has been restored, load the discriminator bits too
        if checkpoint.cloned_from is not None:
            prev_checkpoint = checkpoint.cloned_from

            # Load the network and optimiser parameter states
            path = f'{self.get_logs_path(prev_checkpoint)}/checkpoints/{prev_checkpoint.id}.chkpt'
            state = torch.load(path, map_location=self.device)
            self.net.eval()
            if self.net_args.use_d0:
                self.d0_net.load_state_dict(state['d0_state_dict'])
                self.optimiser_d0.load_state_dict(state['optimiser_d0_state_dict'])
                self.d0_net.eval()
            if self.net_args.use_d2d:
                self.d2d_net.load_state_dict(state['d2d_state_dict'])
                self.optimiser_d2d.load_state_dict(state['optimiser_d2d_state_dict'])
                self.d2d_net.eval()
            if self.net_args.use_d3d:
                self.d3d_net.load_state_dict(state['d3d_state_dict'])
                self.optimiser_d3d.load_state_dict(state['optimiser_d3d_state_dict'])
                self.d3d_net.eval()
            logger.info(f'Loaded state from "{path}"')

        return checkpoint

    def save_checkpoint(self):
        """
        Save the checkpoint information to the database and the network model parameters to file.
        """
        if not (self.net_args.use_d0 or self.net_args.use_d2d or self.net_args.use_d3d):
            return super().save_checkpoint()

        logger.info('Saving model checkpoint...')
        self.checkpoint.save()

        data = {
            'model_state_dict': self.net.state_dict(),
            'd0_state_dict': self.d0_net.state_dict(),
            'd2d_state_dict': self.d2d_net.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'optimiser_d2d_state_dict': self.optimiser_d2d.state_dict(),
        }

        if self.net_args.use_d0:
            data['d0_state_dict'] = self.d0_net.state_dict()
            data['optimiser_d0_state_dict'] = self.optimiser_d0.state_dict()
        if self.net_args.use_d2d:
            data['d2d_state_dict'] = self.d2d_net.state_dict()
            data['optimiser_d2d_state_dict'] = self.optimiser_d2d.state_dict()
        if self.net_args.use_d3d:
            data['d3d_state_dict'] = self.d3d_net.state_dict()
            data['optimiser_d3d_state_dict'] = self.optimiser_d3d.state_dict()

        path = f'{self.logs_path}/checkpoints/{self.checkpoint.id}.chkpt'
        torch.save(data, path)

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint
        self.checkpoint = self.checkpoint.clone()

    def _train_batch(self, data) -> Tuple[Tensor, Tensor, Dict]:
        """
        Train on a single batch of data.
        """

        # Autoencoder training
        outputs, loss, stats = super()._train_batch(data)

        # Discriminator training
        images, X0, W0, X1, X2, Y0, Y1, Y2 = outputs
        bs = X0.shape[0]

        if self.net_args.use_d0:
            self.d0_net.train()

            # Stitch images and masks together
            X0_real = torch.cat([images.detach(), X0.detach()], dim=1)
            Y0_fake = torch.cat([images.detach(), Y0.detach()], dim=1)
            self.optimiser_d0.zero_grad()
            X0_real.requires_grad = True
            Y0_fake.requires_grad = True
            real_pred = self.d0_net(X0_real)
            fake_pred = self.d0_net(Y0_fake)
            real_acc = (real_pred > 0).to(torch.float32).mean()
            fake_acc = (fake_pred < 0).to(torch.float32).mean()
            mean_acc = (real_acc + fake_acc) / 2
            acc_diff = (real_acc - fake_acc)**2
            self.tb_logger.add_scalar('d0/acc/real', real_acc.item(), self.checkpoint.step)
            self.tb_logger.add_scalar('d0/acc/fake', fake_acc.item(), self.checkpoint.step)
            self.tb_logger.add_scalar('d0/acc/mean', mean_acc.item(), self.checkpoint.step)
            d_acc_ema = self.ema(f'd0/acc.{D0_EMA_RATE}', mean_acc.item())
            self.tb_logger.add_scalar(f'd0/acc.{D0_EMA_RATE}', d_acc_ema, self.checkpoint.step)
            d_acc_diff_ema = self.ema(f'd0/acc_diff.{D0_EMA_RATE}', acc_diff.item())
            self.tb_logger.add_scalar(f'd0/acc_diff.{D0_EMA_RATE}', d_acc_diff_ema, self.checkpoint.step)

            # Balance using a bias estimate
            real_pred_mean = torch.mean(real_pred)
            fake_pred_mean = torch.mean(fake_pred)
            bias = real_pred_mean + fake_pred_mean
            self.tb_logger.add_scalar(f'd0_w/bias', bias.item(), self.checkpoint.step)
            bias2 = bias**2
            self.tb_logger.add_scalar(f'd0_w/bias2', bias2.item(), self.checkpoint.step)
            bias2_ema = self.ema(f'd0/bias2.{D2D_EMA_RATE}', bias2.item())
            self.tb_logger.add_scalar(f'd0_w/bias2.{D2D_EMA_RATE}', bias2_ema, self.checkpoint.step)

            # Adversarial loss
            d_loss = -real_pred_mean + fake_pred_mean

            # Gate the loss
            d_loss = torch.sigmoid(d_loss) + bias2
            self.tb_logger.add_scalar('d0/loss', d_loss.item(), self.checkpoint.step)

            # Train
            logger.debug(f'Training d0. Loss: {d_loss:.5f} EMA: {d_acc_ema:.5f}')
            d_loss.backward()
            self.optimiser_d0.step()

        if self.net_args.use_d2d:
            self.d2d_net.train()

            # Use samples from autoencoder
            X1_fake = X1.detach().reshape(bs * 3, 2, X1.shape[2])
            Y1_fake = Y1.detach().reshape(bs * 3, 2, Y1.shape[2])
            bias2_ema = self.ema[f'd2d/bias2.{D2D_EMA_RATE}']

            step = 0
            threshold = 1
            max_steps = 10
            while (step == 0 or bias2_ema > threshold) and step < max_steps:
                try:
                    X1_real = next(self.loader_2d_iterator)[0]
                except StopIteration:
                    logger.debug('Resetting 2D loader.')
                    self.loader_2d_iterator = iter(self.loader_2d)
                    X1_real = next(self.loader_2d_iterator)[0]
                X1_real = (X1_real - torch.tensor([100, 100])).to(self.device)
                X1_real = X1_real.permute(0, 2, 1)
                self.optimiser_d2d.zero_grad()
                X1_real.requires_grad = True
                X1_fake.requires_grad = True
                Y1_fake.requires_grad = True
                real_pred = self.d2d_net(X1_real)
                fake_pred_X1 = self.d2d_net(X1_fake)
                fake_pred_Y1 = self.d2d_net(Y1_fake)

                real_acc = (real_pred > 0).sum() / (bs * 3)
                fake_acc = ((fake_pred_X1 < 0).sum() + (fake_pred_Y1 < 0).sum()) / (bs * 3 * 2)
                mean_acc = (real_acc + fake_acc) / 2
                acc_diff = (real_acc - fake_acc)**2
                self.tb_logger.add_scalar('d2d/acc/real', real_acc, self.checkpoint.step)
                self.tb_logger.add_scalar('d2d/acc/fake', fake_acc, self.checkpoint.step)
                self.tb_logger.add_scalar('d2d/acc/mean', mean_acc, self.checkpoint.step)
                d_acc_ema = self.ema(f'd2d/acc.{D2D_EMA_RATE}', mean_acc)
                self.tb_logger.add_scalar(f'd2d/acc.{D2D_EMA_RATE}', d_acc_ema, self.checkpoint.step)
                d_acc_diff_ema = self.ema(f'd2d/acc_diff.{D2D_EMA_RATE}', acc_diff)
                self.tb_logger.add_scalar(f'd2d/acc_diff.{D2D_EMA_RATE}', d_acc_diff_ema, self.checkpoint.step)
                self.tb_logger.add_scalar(f'd2d/T.{D2D_EMA_RATE}', d_acc_ema - d_acc_diff_ema - threshold,
                                          self.checkpoint.step)

                k = 2
                p = 6

                # Compute W-div gradient penalty
                real_grad_out = torch.ones((bs * 3, 1), device=self.device)
                real_grad = autograd.grad(
                    real_pred, X1_real, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.view(bs * 3, -1).pow(2).sum(dim=1)**(p / 2)
                self.tb_logger.add_scalar(f'd2d_w/real_grad_norm', real_grad_norm.mean(), self.checkpoint.step)

                fake_grad_out = torch.ones((bs * 3, 1), device=self.device)
                fake_grad = autograd.grad(
                    fake_pred_X1, X1_fake, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad_norm = fake_grad.view(bs * 3, -1).pow(2).sum(dim=1)**(p / 2)
                self.tb_logger.add_scalar(f'd2d_w/fake_grad_norm', fake_grad_norm.mean(), self.checkpoint.step)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
                self.tb_logger.add_scalar(f'd2d_w/div_gp', div_gp, self.checkpoint.step)

                # Balance using a bias estimate
                real_pred_mean = torch.mean(real_pred)
                fake_pred_mean = torch.mean(fake_pred_X1 + fake_pred_Y1)
                bias = real_pred_mean + fake_pred_mean
                self.tb_logger.add_scalar(f'd2d_w/bias', bias, self.checkpoint.step)
                bias2 = bias**2
                self.tb_logger.add_scalar(f'd2d_w/bias2', bias2, self.checkpoint.step)
                bias2_ema = self.ema(f'd2d/bias2.{D2D_EMA_RATE}', bias2)
                self.tb_logger.add_scalar(f'd2d_w/bias2.{D2D_EMA_RATE}', bias2_ema, self.checkpoint.step)

                # Adversarial loss
                d_loss = -real_pred_mean + fake_pred_mean + div_gp

                # Gate the loss
                d_loss = torch.sigmoid(d_loss) + bias2
                self.tb_logger.add_scalar('d2d/loss', d_loss, self.checkpoint.step)

                # Only train discriminator if it is worse than threshold
                if step > 0 and bias2_ema < threshold:
                    break
                logger.debug(f'Training d2d step {step} Loss: {d_loss:.5f} EMA: {d_acc_ema:.5f}')
                d_loss.backward()
                self.optimiser_d2d.step()
                self.checkpoint.step += 1
                step += 1
            self.tb_logger.add_scalar('d2d/train_steps', step, self.checkpoint.step)

        return outputs, loss, stats

    def _process_batch(
            self,
            data: Tuple[
                Tensor,
                Tensor,
                List[ObjectId],
                Tensor,
                Tensor,
                Tensor,
            ]
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Split up data from loader and put on correct device
        images, X0, mask_ids, cam_coeffs_base, points_3d_base, points_2d_base = data
        images = images.to(self.device)
        X0 = X0.to(self.device)
        cam_coeffs_base = cam_coeffs_base.to(self.device)
        points_3d_base = points_3d_base.to(self.device)
        points_2d_base = points_2d_base.to(self.device)

        # Run through the network
        W0, X1, X2, Y0, Y1, Y2 = self.net(X0, cam_coeffs_base, points_3d_base, points_2d_base)
        outputs = [images, X0, W0, X1, X2, Y0, Y1, Y2]

        # Calculate losses
        loss, metrics = self.calculate_losses(*outputs)

        return outputs, loss, metrics

    def calculate_losses(
            self,
            images: Tensor,
            X0: Tensor,
            W0: Tensor,
            X1: Tensor,
            X2: Tensor,
            Y0: Tensor,
            Y1: Tensor,
            Y2: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Calculate losses
        """
        stats = {}

        def mse(pred, target):
            loss = F.mse_loss(pred, target, reduction='sum')
            loss = loss / len(pred)  # return loss per-datum so different batch sizes can be compared
            return loss

        total_loss = 0

        def _avg_pool_2d(X_, oob_val=0.):
            # Average pooling with overlap and boundary values
            padded_grad = F.pad(X_, (1, 1, 1, 1), mode='constant', value=oob_val)
            ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
            return ag

        def _avg_surface_2d(X_: torch.Tensor):
            G = -X_
            GS = [G]
            g = G
            while g.shape[-1] > 1:
                g2 = _avg_pool_2d(g, oob_val=0)
                GS.append(g2)
                g = g2

            # Got all the grad averages, now add them together and average
            G_sum = torch.zeros_like(G)
            for i, g in enumerate(GS):
                G_sum += F.interpolate(GS[i], PREPARED_IMAGE_SIZE, mode='bilinear', align_corners=False)
            G_avg = G_sum  # / len(GS)

            return G_avg

        # X0a = _avg_surface_2d(X0)
        # W0a = _avg_surface_2d(W0)
        # Y0a = _avg_surface_2d(Y0)

        for k, (pred, target) in {'masks': [X0, Y0], 'masks_coords': [X0, W0], 'c2d': [X1, Y1],
                                  'c3d': [X2, Y2]}.items():
            # for k, (pred, target) in {'masks': [X0a, Y0], 'masks_coords': [X0a, W0], 'c2d': [X1, Y1], 'c3d': [X2, Y2]}.items():
            # for k, (pred, target) in {'masks': [X0a, Y0a], 'masks_coords': [X0a, W0a], 'c2d': [X1, Y1], 'c3d': [X2, Y2]}.items():
            loss = mse(pred, target)
            stats[f'MSE_{k}'] = loss
            w = getattr(self.optimiser_args, f'w_{k}')
            weighted_loss = loss * w
            stats[f'MSE_{k}/weighted'] = weighted_loss
            total_loss += weighted_loss

        # Masks-sum losses
        masks_sum_loss = mse(Y0.sum(dim=(-1, -2)), X0.sum(dim=(-1, -2)))
        stats['MSE_masks_sum'] = masks_sum_loss
        weighted_loss = masks_sum_loss * self.optimiser_args.w_masks_sum
        stats['MSE_masks_sum/weighted'] = weighted_loss
        total_loss += weighted_loss

        if self.net_args.use_d0:
            d0_input = torch.cat([images, Y0], dim=1)
            d_out_Y0 = self.d0_net(d0_input)
            g_loss = -d_out_Y0.mean()
            g_loss = torch.sigmoid(g_loss)
            weighted_loss = g_loss * self.optimiser_args.w_d0
            stats['d0_g'] = g_loss
            stats['d0_g/weighted'] = weighted_loss
            total_loss += weighted_loss

        if self.net_args.use_d2d:
            bs = X0.shape[0]
            X1_fake = X1.reshape(bs * 3, 2, X1.shape[2])
            Y1_fake = Y1.reshape(bs * 3, 2, Y1.shape[2])
            d_out_X1 = self.d2d_net(X1_fake).squeeze()
            d_out_Y1 = self.d2d_net(Y1_fake).squeeze()
            g_loss = -(d_out_X1.mean() + d_out_Y1.mean()) / 2
            g_loss = torch.sigmoid(g_loss)
            weighted_loss = g_loss * self.optimiser_args.w_d2d
            stats['d2d_g'] = g_loss
            stats['d2d_g/weighted'] = weighted_loss
            total_loss += weighted_loss

        if self.net_args.use_d3d:
            adversarial_loss = torch.nn.BCEWithLogitsLoss()
            bs = X0.shape[0]
            valid = torch.ones(bs, device=self.device)
            X2_fake = X2.reshape(bs)
            g_loss = adversarial_loss(self.d3d_net(X2_fake).squeeze(), valid)
            weighted_loss = g_loss * self.optimiser_args.w_d3d
            stats['d3d_g'] = g_loss
            stats['d3d_g/weighted'] = weighted_loss
            total_loss += weighted_loss

        return total_loss, stats

    def _make_plots(
            self,
            data: Tuple[Tensor, Tensor, List[ObjectId], Tensor, Tensor, Tensor],
            outputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
            train_or_test: str,
            end_of_epoch: bool = False
    ):
        """
        Generate some example plots.
        """
        if self.runtime_args.plot_n_examples > 0 and (
                end_of_epoch or
                (self.runtime_args.plot_every_n_batches > -1
                 and (self.checkpoint.step + 1) % self.runtime_args.plot_every_n_batches == 0)
        ):
            # Select the idxs to plot
            n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
            idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)

            # Unpack variables
            _, _, mask_ids, cam_coeffs_base, points_3d_base, points_2d_base = data
            images, X0, W0, X1, X2, Y0, Y1, Y2 = outputs

            # Fetch masks from database
            masks_docs = {}
            for idx in idxs:
                masks_docs[idx] = SegmentationMasks.objects.get(id=mask_ids[idx])

            # Make plots
            self._plot_masks(X0, W0, Y0, X1, Y1, masks_docs, train_or_test, idxs)
            self._plot_3d_midlines(X2, Y2, masks_docs, train_or_test, idxs)

            # Remove database references to free up memory
            for idx in idxs:
                del (masks_docs[idx])
            del masks_docs

    def _plot_3d_midlines(
            self,
            X2: Tensor,
            Y2: Tensor,
            mask_docs: List[SegmentationMasks],
            train_or_test: str,
            idxs: List[int]
    ):
        """
        Plot generated 3D midline curves.
        """
        X2, Y2 = to_numpy(X2), to_numpy(Y2)
        n_examples = len(idxs)

        fig = plt.figure(figsize=(n_examples * 4, 8))
        gs = gridspec.GridSpec(
            2, n_examples,
            wspace=0.05,
            # hspace=-.99,
            # width_ratios=[1] * n_examples,
            top=0.8,
            bottom=0.01,
            left=0.02,
            right=0.98
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}'
        )

        # Colourmap / facecolors
        cmap = plt.cm.get_cmap('plasma')
        fc = cmap((np.arange(self.dataset_args.n_worm_points) + 0.5) / self.dataset_args.n_worm_points)

        # Scatter plot options
        midline_opts = {
            's': 20,
            'alpha': 0.9,
            'depthshade': True,
            'c': fc
        }

        for i, idx in enumerate(idxs):
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial

            # 3D midline
            ax = fig.add_subplot(gs[0, i], projection='3d')
            vids = "\n\t".join(trial.videos)
            ax.text2D(
                0, 1,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n',
                transform=ax.transAxes
            )

            # Scatter plot of X2
            X = X2[idx].copy()
            ax.scatter(X[0], X[1], X[2], **midline_opts)
            ax.set_title('$X_2$')

            # Scatter plot of Y2
            ax = fig.add_subplot(gs[1, i], projection='3d')
            X = Y2[idx].copy()
            ax.scatter(X[0], X[1], X[2], **midline_opts)
            ax.set_title('$Y_2$')

        self.tb_logger.add_figure(f'3d_midlines_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)

    def _plot_masks(
            self,
            X0: Tensor,
            W0: Tensor,
            Y0: Tensor,
            X1: Tensor,
            Y1: Tensor,
            mask_docs: List[SegmentationMasks],
            train_or_test: str,
            idxs: List[int],
    ):
        """
        Plot the segmentation masks and squared errors.
        """
        X0, W0, Y0, X1, Y1 = to_numpy(X0), to_numpy(W0), to_numpy(Y0), to_numpy(X1), to_numpy(Y1)

        # Colourmap / facecolors
        cmap = plt.cm.get_cmap('plasma')
        fc = cmap((np.arange(self.dataset_args.n_worm_points) + 0.5) / self.dataset_args.n_worm_points)

        # Scatter plot options
        p2d_opts = {
            's': 2,
            'alpha': 0.8,
            'c': fc
        }
        n_examples = len(idxs)

        fig, axes = plt.subplots(
            nrows=4,
            ncols=n_examples,
            figsize=(n_examples * 5, 10),
            gridspec_kw=dict(
                wspace=0.01,
                hspace=-.99,
                width_ratios=[1] * n_examples,
                top=0.8,
                bottom=0.01,
                left=0.02,
                right=0.98
            ),
            constrained_layout=True,
            squeeze=False
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}, '
            f'blur_sigma={self.optimiser_args.renderer_blur_sigma:.1f}'
        )

        # Calculate squared pixel errors
        errors = np.square(X0[idxs] - Y0[idxs])

        for i, idx in enumerate(idxs):
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial
            images = mask_doc.get_images()

            # Stitch images and masks together
            image_triplet = np.concatenate(images, axis=1)
            X0_triplet = np.concatenate(X0[idx], axis=1)
            W0_triplet = np.concatenate(W0[idx], axis=1)
            Y0_triplet = np.concatenate(Y0[idx], axis=1)
            error_triplet = np.concatenate(errors[i], axis=1)

            # Prepped image triplet with overlaid input segmentation masks (X0)
            ax = axes[0, i]
            vids = "\n\t".join(trial.videos)
            ax.text(
                -1, 0,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n'
            )

            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = X0_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X0_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if i == 0:
                ax.text(-0.05, 0.1, '$X_0$', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Prepped image triplet with overlaid 2D coordinates (X1) and rendering (W0)
            ax = axes[1, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = W0_triplet.copy()
            alphas[alphas < 0.2] = 0
            alphas[alphas > 0.6] = 0.8
            ax.imshow(W0_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            ax.axis('off')
            for c in CAMERA_IDXS:
                p2d = X1[idx][2 * c:2 * c + 2].T
                p2d = p2d + np.array([100., 100.])
                p2d[:, 0] += c * 200
                ax.scatter(p2d[:, 0], p2d[:, 1], **p2d_opts)
            if i == 0:
                ax.text(-0.05, 0.1, '$X_1$', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Error between target and output
            ax = axes[2, i]
            m = ax.imshow(error_triplet, cmap=plt.cm.Reds, vmin=errors.min(), vmax=errors.max())
            if i == 0:
                ax.text(-0.05, 0.4, '$|X_0-Y_0|^2$', transform=ax.transAxes, rotation='vertical')
            # Causes memory leak!   -----------
            # if i == n_examples - 1:
            #     cb = fig.colorbar(m, ax=ax, format='%.3f', shrink=0.7)
            ax.axis('off')

            # Prepped image triplet with overlaid 2D coordinates (Y1) and rendering (Y0)
            ax = axes[3, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = Y0_triplet.copy()
            alphas[alphas < 0.2] = 0
            alphas[alphas > 0.6] = 0.8
            ax.imshow(Y0_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            for c in CAMERA_IDXS:
                p2d = Y1[idx][2 * c:2 * c + 2].T
                p2d = p2d + np.array([100., 100.])
                p2d[:, 0] += c * 200
                ax.scatter(p2d[:, 0], p2d[:, 1], **p2d_opts)
            if i == 0:
                ax.text(-0.05, 0.1, '$Y_1$', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)
