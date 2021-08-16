from typing import Tuple
from torchvision import transforms

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import DatasetMidline2D
from wormlab3d.midlines2d.args import DatasetMidline2DCoordsArgs
from wormlab3d.nn.data_loader import DatasetLoader, get_image_transforms, make_data_loader


def get_coord_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    ])


class DatasetMidline2DCoordsLoader(DatasetLoader):
    def __init__(
            self,
            ds: DatasetMidline2D,
            ds_args: DatasetMidline2DCoordsArgs,
            train_or_test: str,
            include_images: bool = True
    ):
        self.midlines = list(getattr(ds, 'X_' + train_or_test))
        self.image_transforms = None
        self.coord_transforms = None
        self.images = []
        self.coords = []
        self.midline_ids = []
        self.n_worm_points = ds_args.n_worm_points
        self.include_images = include_images
        super().__init__(ds, ds_args, train_or_test)

    def _preload_data(self):
        """
        Load all the prepared images and segmentation masks for all midlines in the dataset into memory.
        """
        logger.info('Preloading data from database.')
        images = []
        coords = []
        midline_ids = []
        for m in self.midlines:
            if self.include_images:
                images.append(m.get_prepared_image())
            coords.append(m.get_prepared_coordinates())
            midline_ids.append(m.id)

        self.images = images
        self.coords = coords
        self.midline_ids = midline_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Midline2D]:
        """
        Fetch the image, the coordinates and the midline database record.
        """
        if self.preload:
            midline = self.midlines[index]
            if self.include_images:
                image = self.images[index]
            coords = self.coords[index]
        else:
            midline: Midline2D = self.midlines[index]
            if self.include_images:
                image = midline.get_prepared_image()
            coords = midline.get_prepared_coordinates()

        # Resample the coordinates to the target number
        coords = torch.from_numpy(coords)
        coords = coords.transpose(1, 0).unsqueeze(0)
        coords = F.interpolate(coords, size=(self.n_worm_points,), mode='linear', align_corners=True)
        coords = coords.squeeze(0).transpose(1, 0)

        if self.coord_transforms is not None:
            coords = self.coord_transforms(coords)

        # Convert to torch tensors
        if self.include_images:
            image = to_tensor(image)

            # Image transforms are only applied to the image
            if self.image_transforms is not None:
                image = self.image_transforms(image)
        else:
            image = None

        return image, coords, midline

    def _get_transforms(self):
        if not self.augment:
            return

        # These are applied to the image only
        self.image_transforms = get_image_transforms()
        self.coord_transforms = get_coord_transforms()


def get_data_loader(
        ds: DatasetMidline2D,
        ds_args: DatasetMidline2DCoordsArgs,
        train_or_test: str,
        batch_size: int,
        include_images: bool=True
) -> DataLoader:
    """
    Get a data loader.
    """
    assert train_or_test in ['train', 'test']

    def collate_fn(batch):
        transposed = list(zip(*batch))
        ret = []
        if include_images:
            ret.append(default_collate(transposed[0]))
        ret.append(default_collate(transposed[1]))  # coords
        ret.append(transposed[2])  # midlines
        return ret

    dataset_loader = DatasetMidline2DCoordsLoader(
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
