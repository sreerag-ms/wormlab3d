from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.data.model import Midline2D
from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.midlines2d.data_loader import get_data_loader
from wormlab3d.midlines2d.generate_midline2d_dataset import generate_midline2d_dataset
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.toolkit.util import to_numpy


class Manager(BaseManager):
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
        return (1,) + PREPARED_IMAGE_SIZE

    @property
    def output_shape(self) -> Tuple[int]:
        return (1,) + PREPARED_IMAGE_SIZE

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

    def _process_batch(self, data: Tuple[torch.Tensor, torch.Tensor, List[Midline2D]]) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """

        # Split data into input image and target mask and put on the right device
        # (ignore the midlines returned from the dataloader)
        X, Y_target, _ = data
        X, Y_target = X.to(self.device), Y_target.to(self.device)

        # Put input data through net
        Y_pred = self.predict(X)
        loss, metrics = self.calculate_losses(Y_pred, Y_target)

        return Y_pred, loss, metrics

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
        images, masks, midlines = data
        images, masks, outputs = to_numpy(images), to_numpy(masks), to_numpy(outputs)
        images, masks, outputs = images.squeeze(), masks.squeeze(), outputs.squeeze()
        n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
        idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)
        fig, axes = plt.subplots(
            nrows=5,
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

        # Calculate squared pixel errors
        loss = np.square(masks[idxs] - outputs[idxs])

        for i, idx in enumerate(idxs):
            midline = midlines[idx]
            trial = midline.frame.trial

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
            X = midline.get_prepared_coordinates()
            ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=2, alpha=0.8, marker='x')
            if i == 0:
                ax.text(-0.1, 0.25, 'Original+Annotation', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Augmented images
            ax = axes[1, i]
            ax.imshow(images[idx], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                ax.text(-0.1, 0.25, 'Augmented Image', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Midline segmentation masks
            ax = axes[2, i]
            ax.imshow(masks[idx], cmap=plt.cm.Blues, vmin=0, vmax=1)
            if i == 0:
                ax.text(-0.1, 0.4, 'Target', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Error between target and output
            ax = axes[3, i]
            m = ax.imshow(loss[i], cmap=plt.cm.Reds, vmin=loss.min(), vmax=loss.max())
            if i == 0:
                ax.text(-0.1, 0.4, 'Error', transform=ax.transAxes, rotation='vertical')
            if i == n_examples - 1:
                fig.colorbar(m, ax=ax, format='%.3f')
            ax.axis('off')

            # Generated segmentation mask
            ax = axes[4, i]
            ax.imshow(outputs[idx], cmap=plt.cm.Blues, vmin=0, vmax=1)
            if i == 0:
                ax.text(-0.1, 0.4, 'Output', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        # plt.show()
        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
