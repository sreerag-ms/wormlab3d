import gc
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

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
        self.train_or_test = train_or_test
        self.masks = list(getattr(ds, 'X_' + train_or_test))
        self.cams = ds.get_cameras(tt=train_or_test)
        self.cam_coeffs = ds.get_camera_coefficients(tt=train_or_test)
        self.points_3d = ds.get_points_3d(tt=train_or_test)
        self.points_2d = ds.get_points_2d(tt=train_or_test)
        self.masks_ids = []
        super().__init__(ds, ds_args, train_or_test)

    def _preload_data(self):
        """
        Load all the prepared segmentation masks in the dataset into memory.
        """
        logger.info('Preloading data from database.')
        Xs = []
        masks = []
        for mask in self.masks:
            mask = mask.fetch()
            masks.append(mask)
            Xs.append(mask.X)

        cams = []
        for cam in self.cams:
            cam = cam.fetch()
            cams.append(cam)

        # Replace attributes with the loaded lists of documents
        self.masks = masks
        self.Xs = Xs
        self.cams = cams

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, SegmentationMasks, torch.Tensor]:
        """
        Fetch the the segmentation masks.
        """
        index = 5
        if self.preload:
            mask: SegmentationMasks = self.masks[index]
            X: np.ndarray = self.Xs[index]
        else:
            mask: SegmentationMasks = self.masks[index].fetch()
            X: np.ndarray = mask.X

        coeffs = torch.tensor(self.cam_coeffs[index])
        points_3d_base = torch.tensor(self.points_3d[index], dtype=torch.float32)
        points_2d_base = torch.tensor(self.points_2d[index], dtype=torch.float32)

        # gauss = gauss_test(size=200, sigma=0.1)
        # X = np.stack([gauss, gauss, gauss])

        # Convert mask to torch tensor and normalise
        X = torch.from_numpy(X).contiguous().to(torch.float32)
        X_maxs = torch.amax(X, dim=(1, 2), keepdim=True)
        X_mins = torch.amin(X, dim=(1, 2), keepdim=True)
        X_ranges = X_maxs - X_mins
        X = torch.where(
            X_ranges > 0,
            (X - X_mins) / X_ranges,
            torch.zeros_like(X)
        )

        gc.collect()

        return X, mask, coeffs, points_3d_base, points_2d_base


def gauss_test(size, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    dst = np.sqrt(x * x + y * y)
    arr = np.exp(-((dst)**2 / (2.0 * sigma**2)))
    return arr


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
            transposed[1],  # segmentation masks (documents)
            default_collate(transposed[2]),  # camera coefficients
            default_collate(transposed[3]),  # points 3d
            default_collate(transposed[4]),  # points 2d
        ]

    logger.debug(f'Making {train_or_test} dataset loader.')
    dataset_loader = DatasetSegmentationMasksLoader(
        ds=ds,
        ds_args=ds_args,
        train_or_test=train_or_test,
    )

    logger.debug(f'Making {train_or_test} loader.')
    loader = make_data_loader(
        dataset_loader=dataset_loader,
        ds_args=ds_args,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return loader
