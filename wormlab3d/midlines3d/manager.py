from typing import Dict
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from wormlab3d.data.model import Midline2D
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.midlines3d.data_loader import get_data_loader
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.nn.manager import Manager as BaseManager


class Manager(BaseManager):
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetSegmentationMasksArgs,
            net_args: NetworkArgs,
            optimiser_args: OptimiserArgs,
    ):
        super().__init__(runtime_args, dataset_args, net_args, optimiser_args)

    def _generate_dataset(self):
        return generate_masks_dataset(self.dataset_args)

    def _get_data_loader(self, train_or_test: str) -> DataLoader:
        return get_data_loader(
            ds=self.ds,
            ds_args=self.dataset_args,
            train_or_test=train_or_test,
            batch_size=self.runtime_args.batch_size
        )

    def _process_batch(self, data: Tuple[torch.Tensor, List[SegmentationMasks]]) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """

        # Put input data through net
        X, _ = data
        X = X.to(self.device)
        Y_pred = self.predict(X)

        # Calculate losses todo
        # loss = F.mse_loss(Y_pred, Y_target)
        # assert not is_bad(loss)
        # loss = loss / len(X)  # return loss per-datum so different batch sizes can be compared
        loss = 0

        return Y_pred, loss, {}

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
        pass
