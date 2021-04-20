from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor

from wormlab3d import logger
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.data.model.dataset import DatasetSegmentationMasks
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.nn.data_loader import DatasetLoader, make_data_loader


class DatasetSegmentationMasksLoader(DatasetLoader):
    def __init__(
            self,
            ds: DatasetSegmentationMasks,
            ds_args: DatasetSegmentationMasksArgs,
            train_or_test: str,
    ):
        self.masks = list(getattr(ds, 'X_' + train_or_test))
        self.masks_ids = []
        super().__init__(ds, ds_args, train_or_test)

    def _preload_data(self):
        """
        Load all the prepared segmentation masks in the dataset into memory.
        """
        logger.info('Preloading data from database.')
        Xs = []
        masks_ids = []
        for m in self.masks:
            Xs.append(m.X)
            masks_ids.append(m.id)
        self.Xs = Xs
        self.masks_ids = masks_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, SegmentationMasks]:
        """
        Fetch the the segmentation masks.
        """
        if self.preload:
            mask = self.masks[index]
            X = self.Xs[index]
        else:
            mask: SegmentationMasks = self.masks[index]
            X = mask.X

        # Convert to torch tensor
        X = torch.from_numpy(X).contiguous().to(torch.float32)

        return X, mask


def get_data_loader(
        ds: DatasetSegmentationMasks,
        ds_args: DatasetSegmentationMasksArgs,
        train_or_test: str,
        batch_size: int
) -> DataLoader:
    """
    Get a data loader.
    """
    assert train_or_test in ['train', 'test']

    def collate_fn(batch):
        transposed = list(zip(*batch))
        return [
            default_collate(transposed[0]),  # masks
            transposed[1]  # segmentation masks (documents)
        ]

    dataset_loader = DatasetSegmentationMasksLoader(
        ds=ds,
        ds_args=ds_args,
        train_or_test=train_or_test,
    )

    loader = make_data_loader(
        dataset_loader=dataset_loader,
        ds_args=ds_args,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return loader
