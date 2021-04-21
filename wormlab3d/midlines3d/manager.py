from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE, N_WORM_POINTS
from wormlab3d.data.model import SegmentationMasks, NetworkParameters
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.midlines3d.args.runtime_args import Midline3DRuntimeArgs
from wormlab3d.midlines3d.data_loader import get_data_loader
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS
from wormlab3d.midlines3d.enc_dec import EncDec
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.plot_utils import clear_axes
from wormlab3d.toolkit.util import to_numpy


class Manager(BaseManager):
    def __init__(
            self,
            runtime_args: Midline3DRuntimeArgs,
            dataset_args: DatasetSegmentationMasksArgs,
            net_args: NetworkArgs,
            optimiser_args: OptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

    @property
    def input_shape(self) -> Tuple[int]:
        return (3,) + PREPARED_IMAGE_SIZE

    @property
    def output_shape(self) -> Tuple[int]:
        return 3 * N_WORM_POINTS + 3 * N_CAM_COEFFICIENTS, 1, 1

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
        The network parameters refer to the encoder part of the network.
        Here we wrap this in an encoder-decoder model with the decoder being the camera model.
        """
        net, net_params = super()._init_network()

        full_net = EncDec(
            encoder=net,
            blur_sigma=self.runtime_args.reprojection_blur_sigma
        )

        return full_net, net_params

    def _process_batch(self, data: Tuple[torch.Tensor, List[SegmentationMasks]]) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """

        # Put input data through net
        X, _ = data
        X = X.to(self.device)
        outputs = self.predict(X)
        points_3d, coeffs, points_2d, masks = outputs

        # Calculate losses
        loss, metrics = self.calculate_losses(masks, X)

        return outputs, loss, metrics

    def _make_plots(
            self,
            data: Tuple[torch.Tensor, List[SegmentationMasks]],
            outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
            n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
            idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)
            self._plot_3d_midlines(data, outputs, train_or_test, idxs)
            self._plot_masks(data, outputs, train_or_test, idxs)

    def _plot_3d_midlines(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, List[SegmentationMasks]],
            outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            train_or_test: str,
            idxs: List[int]
    ):
        masks_in, mask_docs = data
        points_3d, coeffs, points_2d, masks_out = outputs
        points_3d = to_numpy(points_3d)
        n_examples = len(idxs)

        fig = plt.figure(figsize=(n_examples * 4, 6))
        gs = gridspec.GridSpec(
            1, n_examples,
            wspace=0.01,
            hspace=-.99,
            width_ratios=[1] * n_examples,
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
        fc = cmap((np.arange(N_WORM_POINTS) + 0.5) / N_WORM_POINTS)

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

            # Prepped image triplet with overlaid input segmentation masks
            ax = fig.add_subplot(gs[0, i], projection='3d')
            vids = "\n\t".join(trial.videos)
            ax.text2D(
                0, 1,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n',
                transform=ax.transAxes
            )
            clear_axes(ax)

            # Scatter plot of midline
            X = points_3d[idx].transpose()
            ax.scatter(X[0], X[1], X[2], **midline_opts)

        self.tb_logger.add_figure(f'3d_midlines_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()

    def _plot_masks(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, List[SegmentationMasks]],
            outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            train_or_test: str,
            idxs: List[int]
    ):
        masks_in, mask_docs = data
        points_3d, coeffs, points_2d, masks_out = outputs
        masks_in, masks_out = to_numpy(masks_in), to_numpy(masks_out)
        n_examples = len(idxs)

        fig, axes = plt.subplots(
            nrows=3,
            ncols=n_examples,
            figsize=(n_examples * 5, 7),
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
            f'blur_sigma={self.runtime_args.reprojection_blur_sigma:.1f}'
        )

        # Calculate squared pixel errors
        errors = np.square(masks_in[idxs] - masks_out[idxs])

        for i, idx in enumerate(idxs):
            mask_doc = mask_docs[idx]
            trial = mask_doc.trial
            images = mask_doc.get_images()

            # Stitch images and masks together
            image_triplet = np.concatenate(images, axis=1)
            masks_in_triplet = np.concatenate(masks_in[idx], axis=1)
            masks_out_triplet = np.concatenate(masks_out[idx], axis=1)
            error_triplet = np.concatenate(errors[i], axis=1)

            # Prepped image triplet with overlaid input segmentation masks
            ax = axes[0, i]
            vids = "\n\t".join(trial.videos)
            ax.text(
                -1, 0,
                f'Trial: {trial.id}\n'
                f'Videos: \n\t{vids}\n'
                f'Frame: {mask_doc.frame.frame_num} (id={mask_doc.frame.id})\n'
            )

            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = masks_in_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(masks_in_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if i == 0:
                ax.text(-0.05, 0.1, 'Original+Masks_In', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

            # Error between target and output
            ax = axes[1, i]
            m = ax.imshow(error_triplet, cmap=plt.cm.Reds, vmin=errors.min(), vmax=errors.max())
            if i == 0:
                ax.text(-0.05, 0.4, 'Error', transform=ax.transAxes, rotation='vertical')
            if i == n_examples - 1:
                fig.colorbar(m, ax=ax, format='%.3f', shrink=0.7)
            ax.axis('off')

            # Prepped image triplet with overlaid output segmentation masks
            ax = axes[2, i]
            ax.imshow(image_triplet, vmin=0, vmax=1, cmap='gray', aspect='auto')
            alphas = masks_out_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(masks_out_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            if i == 0:
                ax.text(-0.05, 0.1, 'Original+Masks_Out', transform=ax.transAxes, rotation='vertical')
            ax.axis('off')

        self.tb_logger.add_figure(f'masks_{train_or_test}', fig, self.checkpoint.step)
        self.tb_logger.flush()
