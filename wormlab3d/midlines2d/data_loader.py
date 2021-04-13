from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor

from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import DatasetMidline2D
from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.nn.data_loader import DatasetLoader, get_affine_transforms, get_image_transforms, make_data_loader


class DatasetMidline2DLoader(DatasetLoader):
    def __init__(
            self,
            ds: DatasetMidline2D,
            ds_args: DatasetMidline2DArgs,
            train_or_test: str,
    ):
        self.blur_sigma = ds_args.blur_sigma
        self.midlines = list(getattr(ds, 'X_' + train_or_test))
        self.affine_transforms = None
        self.image_transforms = None
        self.images = []
        self.masks = []
        self.midline_ids = []
        super().__init__(ds, ds_args, train_or_test)

    def _preload_data(self):
        """
        Load all the prepared images and segmentation masks for all midlines in the dataset into memory.
        """
        logger.info('Preloading data from database.')
        images = []
        masks = []
        midline_ids = []
        for m in self.midlines:
            images.append(m.get_prepared_image())
            masks.append(m.get_segmentation_mask(blur_sigma=self.blur_sigma))
            midline_ids.append(m.id)

        self.images = images
        self.masks = masks
        self.midline_ids = midline_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Midline2D]:
        """
        Fetch the image, the segmentation mask and the midline database record.
        """
        if self.preload:
            midline = self.midlines[index]
            image = self.images[index]
            mask = self.masks[index]
        else:
            midline: Midline2D = self.midlines[index]
            image = midline.get_prepared_image()
            mask = midline.get_segmentation_mask(blur_sigma=self.blur_sigma)

        # Convert to torch tensors
        image = to_tensor(image)
        mask = to_tensor(mask)

        # Stack the image and mask together so they get the same geometric transformations
        if self.affine_transforms is not None:
            stacked = torch.cat([image, mask], dim=0)
            stacked = self.affine_transforms(stacked)
            image, mask = torch.chunk(stacked, chunks=2, dim=0)

        # Image transforms are only applied to the image
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image, mask, midline

    def _get_transforms(self):
        if not self.augment:
            return

        # These transforms are applied to the image and mask together
        self.affine_transforms = get_affine_transforms()

        # These are applied to the image only
        self.image_transforms = get_image_transforms()


def get_data_loader(
        ds: DatasetMidline2D,
        ds_args: DatasetMidline2DArgs,
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
            transposed[2]  # midlines
        ]

    dataset_loader = DatasetMidline2DLoader(
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
