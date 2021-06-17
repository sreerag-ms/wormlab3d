from typing import Tuple

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


class DatasetMidline2DCoordsLoader(DatasetLoader):
    def __init__(
            self,
            ds: DatasetMidline2D,
            ds_args: DatasetMidline2DCoordsArgs,
            train_or_test: str,
    ):
        self.midlines = list(getattr(ds, 'X_' + train_or_test))
        self.image_transforms = None
        self.images = []
        self.coords = []
        self.midline_ids = []
        self.n_worm_points = ds_args.n_worm_points
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
            image = self.images[index]
            coords = self.coords[index]
        else:
            midline: Midline2D = self.midlines[index]
            image = midline.get_prepared_image()
            coords = midline.get_prepared_coordinates()

        # Convert to torch tensors
        image = to_tensor(image)

        # Resample the coordinates to the target number
        coords = torch.from_numpy(coords)
        coords = coords.transpose(1, 0).unsqueeze(0)
        coords = F.interpolate(coords, size=(self.n_worm_points,), mode='linear', align_corners=True)
        coords = coords.squeeze(0).transpose(1, 0)

        # Image transforms are only applied to the image
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image, coords, midline

    def _get_transforms(self):
        if not self.augment:
            return

        # These are applied to the image only
        self.image_transforms = get_image_transforms()


def get_data_loader(
        ds: DatasetMidline2D,
        ds_args: DatasetMidline2DCoordsArgs,
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
            default_collate(transposed[1]),  # coords
            transposed[2]  # midlines
        ]

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
