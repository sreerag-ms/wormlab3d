from typing import Tuple

import torch
from mongoengine import DoesNotExist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import DatasetMidline2D
from wormlab3d.midlines2d.args import DatasetArgs


class DatasetLoader(DatasetTorch):
    def __init__(
            self,
            ds: DatasetMidline2D,
            train_or_test: str,
            augment: bool = False,
            blur_sigma: float = 0,
            preload: bool = False
    ):
        assert train_or_test in ['train', 'test']
        self.ds = ds
        self.train_or_test = train_or_test
        self.augment = augment
        self.blur_sigma = blur_sigma
        self.midlines = list(getattr(self.ds, 'X_' + train_or_test))
        self.affine_transforms, self.image_transforms = self._get_transforms()
        self.preload = preload
        if preload:
            self._init_data()

    def _init_data(self):
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

    def __len__(self) -> int:
        return self.ds.get_size(self.train_or_test)

    def _get_transforms(self) -> Tuple:
        if not self.augment:
            return None, None

        # These transforms are applied to the image and mask together
        affine_transforms = transforms.Compose([
            transforms.RandomOrder([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=180,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=50
                )
            ])
        ])

        # These are applied to the image only
        image_transforms = transforms.Compose([
            transforms.RandomOrder([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.01, 10)),
                transforms.RandomErasing(p=0.3, scale=(0.01, 0.2)),
            ])
        ])

        return affine_transforms, image_transforms


def get_data_loader(
        ds: DatasetMidline2D,
        ds_args: DatasetArgs,
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

    dataset_loader = DatasetLoader(
        ds=ds,
        blur_sigma=ds_args.blur_sigma,
        augment=ds_args.augment,
        train_or_test=train_or_test,
        preload=ds_args.preload_from_database
    )

    loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=ds_args.n_dataloader_workers,
        drop_last=True,
        collate_fn=collate_fn
    )

    return loader



def load_dataset(dataset_args: DatasetArgs) -> DatasetMidline2D:
    """
    Load an existing dataset from the database.
    """
    ds = None

    # If we have a dataset id then load this from the database
    if dataset_args.ds_id is not None:
        ds = DatasetMidline2D.objects.get(id=dataset_args.ds_id)
    else:
        # Otherwise, try to find one matching the same parameters
        datasets = DatasetMidline2D.find_from_args(dataset_args)
        if datasets.count() > 0:
            ds = datasets[0]
            logger.info(f'Found {len(datasets)} suitable datasets in database, using most recent.')

    if ds is None:
        raise DoesNotExist()

    logger.info(f'Loaded dataset (id={ds.id}, created={ds.created}).')

    return ds
