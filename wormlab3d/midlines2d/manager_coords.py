from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT
from wormlab3d.data.model import Midline2D, NetworkParameters
from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.midlines2d.coords_net import CoordsNet
from wormlab3d.midlines2d.data_loader_coords import get_data_loader
from wormlab3d.midlines2d.generate_midline2d_dataset import generate_midline2d_dataset
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.util import to_numpy


class ManagerCoords(BaseManager):
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetMidline2DArgs,
            net_args: NetworkArgs,
            optimiser_args: OptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

    @property
    def input_shape(self) -> Tuple[int]:
        return 1, PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT

    @property
    def output_shape(self) -> Tuple[int]:
        return self.dataset_args.n_worm_points * 2,

    @property
    def stat_keys(self) -> List[str]:
        """Define the loss keys to track."""
        return []

    def _generate_dataset(self):
        return generate_midline2d_dataset(self.dataset_args)

    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        return get_data_loader(
            ds=self.ds,
            ds_args=self.dataset_args,
            train_or_test=train_or_test,
            batch_size=self.runtime_args.batch_size
        )

    def _init_network(self) -> Tuple[BaseNet, NetworkParameters]:
        """
        """
        net, net_params = super()._init_network()
        full_net = CoordsNet(
            net=net,
            n_worm_points=self.dataset_args.n_worm_points
        )

        return full_net, net_params

    def _process_batch(self, data: Tuple[torch.Tensor, torch.Tensor, List[Midline2D]]) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """

        # Split data into input image and target coords and put on the right device
        # (ignore the midlines returned from the dataloader)
        X, Y_target, _ = data
        X, Y_target = X.to(self.device), Y_target.to(self.device)

        # Put input data through net
        Y_pred = self.predict(X)
        loss, metrics = self.calculate_losses(Y_pred, Y_target)

        return Y_pred, loss, metrics

    def calculate_losses(self, Y_pred: torch.Tensor, Y_target: torch.tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate losses
        """
        loss, stats = super().calculate_losses(Y_pred, Y_target)

        # Calculate losses in both orientations and take the minimum
        loss_a = F.mse_loss(Y_pred, Y_target, reduction='none').sum(dim=(1, 2))
        loss_b = F.mse_loss(Y_pred, Y_target.flip(dims=(1,)), reduction='none').sum(dim=(1, 2))
        loss_min = torch.min(loss_a, loss_b)
        loss = loss_min.mean()
        stats['mse_min'] = loss

        return loss, stats

    def _make_plots(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, List[Midline2D]],
            outputs: torch.Tensor,
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
            self._plot_masks(data, outputs, train_or_test)

    def _plot_masks(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, List[Midline2D]],
            outputs: torch.Tensor,
            train_or_test: str
    ):
        images, coords, midlines = data
        images, coords, outputs = to_numpy(images), to_numpy(coords), to_numpy(outputs)
        images, coords, outputs = images.squeeze(), coords.squeeze(), outputs.squeeze()
        n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
        idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)
        fig, axes = plt.subplots(
            nrows=3,
            ncols=n_examples,
            figsize=(16, n_examples * 4),
            gridspec_kw=dict(
                wspace=0.01, hspace=0.02,
                width_ratios=[1] * n_examples,
                top=0.9,
                bottom=0.05,
                left=0.05,
                right=0.95
            ),
        )

        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step}, '
            f'blur_sigma={self.dataset_args.blur_sigma:.1f}'
        )

        for i, idx in enumerate(idxs):
            midline = midlines[idx]
            trial = midline.frame.trial

            # Target mask
            mask_target = make_segmentation_mask(
                X=coords[idx],
                blur_sigma=self.dataset_args.blur_sigma,
                draw_mode='line_aa',
                image_size=(PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT)
            )

            # Output mask
            mask_output = make_segmentation_mask(
                X=outputs[idx],
                blur_sigma=self.dataset_args.blur_sigma,
                draw_mode='line_aa',
                image_size=(PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
                raise_on_empty=False
            )

            # Shows original (prepped) images with midline annotations
            ax = axes[0, i]
            ax.text(
                -1, -10,
                f'Midline: {midline.id}\n'
                f'Trial: {trial.id}\n'
                f'Video: {trial.videos[midline.camera]}\n'
                f'Frame: {midline.frame.frame_num} (id={midline.frame.id})\n'
                f'Camera: {midline.camera}'
            )
            ax.imshow(midline.get_prepared_image(), cmap='gray', vmin=0, vmax=1)
            X = coords[idx]
            ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=2, alpha=0.8, marker='x')
            if i == 0:
                ax.text(-0.1, 0.25, 'Original+Annotation', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Augmented images
            ax = axes[1, i]
            ax.imshow(images[idx], cmap='gray', vmin=0, vmax=1)
            X = outputs[idx]
            ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=2, alpha=0.8, marker='x')
            if i == 0:
                ax.text(-0.1, 0.25, 'Augmented+Output', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Error between target and output
            ax = axes[2, i]
            ax.imshow(mask_target - mask_output, cmap=plt.cm.PRGn, vmin=-1, vmax=1)
            if i == 0:
                ax.text(-0.1, 0.4, 'Comparison', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        # plt.show()
        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
