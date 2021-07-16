from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from bson import ObjectId
from matplotlib import gridspec
from torch import Tensor
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE, CAMERA_IDXS, logger
from wormlab3d.data.model import SegmentationMasks, NetworkParameters
from wormlab3d.data.model.network_parameters import NetworkParametersRotAE
from wormlab3d.midlines3d.args import *
from wormlab3d.midlines3d.data_loader import get_data_loader
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.midlines3d.rotae_net import RotAENet
from wormlab3d.nn.args import RuntimeArgs
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.util import to_numpy


class ManagerRotAE(BaseManager):
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetRotAECoordsArgs,
            net_args: RotAENetworkArgs,
            optimiser_args: RotAEOptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

    @property
    def input_shape(self) -> Tuple[int]:
        """A triplet of segmentation masks, one from each camera."""
        return (3,) + PREPARED_IMAGE_SIZE

    @property
    def output_shape(self) -> Tuple[int]:
        """A triplet of projected coordinate renderings, one from each camera."""
        return (3,) + PREPARED_IMAGE_SIZE

    def _generate_dataset(self):
        return generate_masks_dataset(self.dataset_args)

    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        return get_data_loader(
            ds=self.ds,
            ds_args=self.dataset_args,
            train_or_test=train_or_test,
            batch_size=self.runtime_args.batch_size
        )

    def _init_network(self) -> Tuple[BaseNet, NetworkParameters]:
        """
        Initialise the network which includes 2-3 subnetworks.
        """
        masks_shape = (3,) + PREPARED_IMAGE_SIZE
        n_supplements = N_CAM_COEFFICIENTS * 3 + 3 + 2 * 3
        c2d_input_shape = (3 + n_supplements,) + PREPARED_IMAGE_SIZE
        c2d_output_shape = (3 + n_supplements, self.dataset_args.n_worm_points, 2)
        c3d_input_shape = (3, self.dataset_args.n_worm_points, 2)
        c3d_output_shape = (self.dataset_args.n_worm_points, 3)

        # Initialise c2d and c3d networks
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

        params = {**{
            'network_type': 'rotae',
            'c2d_net': c2d_net_params,
            'c3d_net': c3d_net_params,
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
            distorted_cameras=True,
        )

        return full_net, net_params

    def _init_metrics(self) -> Tuple[Dict[str, callable], List[str]]:
        return {}, []

    def _process_batch(
            self,
            data: Tuple[
                torch.Tensor,
                List[ObjectId],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Split up data from loader and put on correct device
        X0, mask_ids, cam_coeffs_base, points_3d_base, points_2d_base = data
        X0 = X0.to(self.device)
        cam_coeffs_base = cam_coeffs_base.to(self.device)
        points_3d_base = points_3d_base.to(self.device)
        points_2d_base = points_2d_base.to(self.device)

        # Run through the network
        X1, X2, Y0, Y1, Y2 = self.net(X0, cam_coeffs_base, points_3d_base, points_2d_base)
        outputs = [X0, X1, X2, Y0, Y1, Y2]

        # Calculate losses
        loss, metrics = self.calculate_losses(*outputs)

        return outputs, loss, metrics

    def calculate_losses(self, X0: torch.tensor, X1: torch.tensor, X2: torch.tensor, Y0: torch.tensor, Y1: torch.tensor,
                         Y2: torch.tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate losses
        """
        stats = {}

        def mse(pred, target):
            loss = F.mse_loss(pred, target, reduction='sum')
            loss = loss / len(pred)  # return loss per-datum so different batch sizes can be compared
            return loss

        total_loss = 0
        for k, (pred, target) in {'masks': [X0, Y0], 'c2d': [X1, Y1], 'c3d': [X2, Y2]}.items():
            loss = mse(pred, target)
            stats[f'MSE_{k}'] = loss
            w = getattr(self.optimiser_args, f'w_{k}')
            weighted_loss = loss * w
            stats[f'MSE_{k}/weighted'] = weighted_loss
            total_loss += weighted_loss

        return total_loss, stats

    def _make_plots(
            self,
            data: Tuple[Tensor, List[ObjectId], Tensor, Tensor, Tensor],
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
            _, mask_ids, cam_coeffs_base, points_3d_base, points_2d_base = data
            X0, X1, X2, Y0, Y1, Y2 = outputs

            # Fetch masks from database
            masks_docs = {}
            for idx in idxs:
                masks_docs[idx] = SegmentationMasks.objects.get(id=mask_ids[idx])

            # Make plots
            self._plot_masks(X0, Y0, X1, Y1, masks_docs, train_or_test, idxs)
            self._plot_3d_midlines(X2, Y2, masks_docs, train_or_test, idxs)

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
            X = X2[idx].transpose().copy()
            ax.scatter(X[0], X[1], X[2], **midline_opts)
            ax.set_title('$X_2$')

            # Scatter plot of Y2
            ax = fig.add_subplot(gs[1, i], projection='3d')
            X = Y2[idx].transpose().copy()
            ax.scatter(X[0], X[1], X[2], **midline_opts)
            ax.set_title('$Y_2$')

        self.tb_logger.add_figure(f'3d_midlines_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)

    def _plot_masks(
            self,
            X0: Tensor,
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
        X0, Y0, X1, Y1 = to_numpy(X0), to_numpy(Y0), to_numpy(X1), to_numpy(Y1)

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
            nrows=5,
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

            # Prepped image triplet with overlaid 2D coordinates (X1)
            ax = axes[1, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            for c in CAMERA_IDXS:
                p2d = X1[idx][c]
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
            if i == n_examples - 1:
                fig.colorbar(m, ax=ax, format='%.3f', shrink=0.7)
            ax.axis('off')

            # Prepped image triplet with overlaid 2D coordinates (Y1)
            ax = axes[3, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            for c in CAMERA_IDXS:
                p2d = Y1[idx][c]
                p2d = p2d + np.array([100., 100.])
                p2d[:, 0] += c * 200
                ax.scatter(p2d[:, 0], p2d[:, 1], **p2d_opts)
            if i == 0:
                ax.text(-0.05, 0.1, '$Y_1$', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Prepped image triplet with overlaid output segmentation masks
            ax = axes[4, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = Y0_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(Y0_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if i == 0:
                ax.text(-0.05, 0.1, '$Y_0$', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
        plt.close(fig)
