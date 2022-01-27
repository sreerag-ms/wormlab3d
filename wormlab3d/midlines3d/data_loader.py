import os
from typing import Tuple

import numpy as np
import torch
from bson import ObjectId
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from wormlab3d import logger, DATASETS_SEG_MASKS_PATH, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import SegmentationMasks, Frame
from wormlab3d.data.model.dataset import DatasetSegmentationMasks
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.nn.data_loader import DatasetLoader, make_data_loader, get_image_transforms
from wormlab3d.toolkit.util import hash_data


class DatasetSegmentationMasksLoader(DatasetLoader):
    def __init__(
            self,
            ds: DatasetSegmentationMasks,
            ds_args: DatasetSegmentationMasksArgs,
            train_or_test: str,
    ):
        self.train_or_test = train_or_test
        self.images = []
        self.masks = list(getattr(ds, 'X_' + train_or_test))
        self.cams = ds.get_cameras(tt=train_or_test)
        self.cam_coeffs = ds.get_camera_coefficients(tt=train_or_test)
        self.points_3d = ds.get_points_3d(tt=train_or_test)
        self.points_2d = ds.get_points_2d(tt=train_or_test)
        self.affine_transforms = None
        self.image_transforms = None
        super().__init__(ds, ds_args, train_or_test)

    def _preload_data(self):
        """
        Load all the prepared segmentation masks in the dataset into memory.
        Creates a file cache of the masks on disk and loads this if available.
        """
        logger.info('Preloading data.')

        # Collect ids to fetch in bulk
        mask_ids = []
        for mask in self.masks:
            mask_ids.append(mask.id)
        self.mask_ids = mask_ids
        shape = ((len(mask_ids), 3,) + PREPARED_IMAGE_SIZE)

        # Try to load from file cache if available
        hash = hash_data([str(m) for m in mask_ids])
        filename_Xs = f'{hash}x.npz'
        filename_images = f'{hash}i.npz'
        path_Xs = DATASETS_SEG_MASKS_PATH / filename_Xs
        path_images = DATASETS_SEG_MASKS_PATH / filename_images
        if os.path.exists(path_Xs) and os.path.exists(path_images):
            try:
                self.Xs = np.memmap(path_Xs, dtype=np.float32, mode='r', shape=shape)
                self.images = np.memmap(path_images, dtype=np.float32, mode='r', shape=shape)
                logger.info(f'Loaded data from {path_Xs} and {path_images}.')
                return
            except Exception as e:
                logger.warning(f'Could not load from {path_Xs}. {e}')
        else:
            logger.info('File cache unavailable, loading from database.')

        # Fetch masks
        Xs = np.memmap(path_Xs, dtype='float32', mode='w+', shape=shape)
        images = np.memmap(path_images, dtype='float32', mode='w+', shape=shape)
        batch_size = 1000
        i = 0
        while len(mask_ids) > 0:
            mids = mask_ids[:batch_size]
            mask_ids = mask_ids[batch_size:]
            pipeline = [
                {'$match': {'_id': {'$in': mids}}},
                {'$lookup': {'from': 'frame', 'localField': 'frame', 'foreignField': '_id', 'as': 'frame'}},
                {'$unwind': {'path': '$frame'}},
                {'$project': {'_id': 1, 'X': 1, 'images': '$frame.images'}}
            ]
            data = list(SegmentationMasks.objects().aggregate(pipeline, batchSize=batch_size))

            # Convert results to dict as order is not preserved
            dic = {}
            for datum in data:
                dic[datum['_id']] = {
                    'images': Frame.images.to_python(datum['images']),
                    'X': SegmentationMasks.X.to_python(datum['X'])
                }

            # Build arrays
            for mid in mids:
                images[i] = dic[mid]['images']
                Xs[i] = dic[mid]['X']
                i += 1
        assert i == len(self)

        # Save to file
        logger.debug(f'Saving images file cache to {path_images}.')
        images.flush()
        self.images = images
        logger.debug(f'Saving Xs file cache to {path_Xs}.')
        Xs.flush()
        self.Xs = Xs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ObjectId, torch.Tensor]:
        """
        Fetch the the segmentation masks.
        """
        # index = 5
        if self.preload:
            mask_id: ObjectId = self.mask_ids[index]
            images: np.ndarray = self.images[index]
            X: np.ndarray = self.Xs[index]
        else:
            mask: SegmentationMasks = self.masks[index].fetch()
            mask_id: ObjectId = mask.id
            images = mask.get_images()
            X: np.ndarray = mask.X

        coeffs = torch.tensor(self.cam_coeffs[index])
        points_3d_base = torch.tensor(self.points_3d[index], dtype=torch.float32)
        points_2d_base = torch.tensor(self.points_2d[index], dtype=torch.float32)

        # gauss = gauss_test(size=200, sigma=0.1)
        # X = np.stack([gauss, gauss, gauss])

        images = torch.from_numpy(np.array(images).copy()).contiguous().to(torch.float32)

        # Image transforms are only applied to the image
        if self.image_transforms is not None:
            images[0] = self.image_transforms(images[0].unsqueeze(0)).squeeze()
            images[1] = self.image_transforms(images[1].unsqueeze(0)).squeeze()
            images[2] = self.image_transforms(images[2].unsqueeze(0)).squeeze()

        # Convert mask to torch tensor and normalise
        X = torch.from_numpy(X.copy()).contiguous().to(torch.float32)
        X_maxs = torch.amax(X, dim=(1, 2), keepdim=True)
        X_mins = torch.amin(X, dim=(1, 2), keepdim=True)
        X_ranges = X_maxs - X_mins
        X = torch.where(
            X_ranges > 0,
            (X - X_mins) / X_ranges,
            torch.zeros_like(X)
        )

        return images, X, mask_id, coeffs, points_3d_base, points_2d_base

    def _get_transforms(self):
        if not self.augment:
            return

        # These are applied to the image only for use in d0
        self.image_transforms = get_image_transforms()


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
            default_collate(transposed[0]),  # images
            default_collate(transposed[1]),  # masks
            transposed[2],  # segmentation masks object ids
            default_collate(transposed[3]),  # camera coefficients
            default_collate(transposed[4]),  # points 3d
            default_collate(transposed[5]),  # points 2d
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
