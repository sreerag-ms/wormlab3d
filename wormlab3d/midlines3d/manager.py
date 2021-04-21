from typing import Dict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from wormlab3d import PREPARED_IMAGE_SIZE, N_WORM_POINTS
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.data.model.network_parameters import NetworkParameters
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.midlines3d.args.runtime_args import Midline3DRuntimeArgs
from wormlab3d.midlines3d.data_loader import get_data_loader
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS
from wormlab3d.midlines3d.enc_dec import EncDec
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs
from wormlab3d.nn.manager import Manager as BaseManager
from wormlab3d.nn.models.basenet import BaseNet


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
            self._plot_masks(data, outputs, train_or_test)

    def _plot_masks(
            self,
            data: Tuple[torch.Tensor, torch.Tensor, List[SegmentationMasks]],
            outputs: torch.Tensor,
            train_or_test: str
    ):
        pass
