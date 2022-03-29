from abc import ABC, abstractmethod

import torch
from mongoengine import DoesNotExist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch
from torchvision import transforms

from wormlab3d import logger
from wormlab3d.data.model import Dataset
from wormlab3d.data.model.dataset import DatasetEigentraces, DATASET_TYPE_EIGENTRACES
from wormlab3d.nn.args import DatasetArgs


def get_affine_transforms() -> transforms.Compose:
    return transforms.Compose([
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


def get_image_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomOrder([
            transforms.GaussianBlur(kernel_size=9, sigma=(0.01, 10)),
            transforms.RandomErasing(p=0.6, scale=(0.05, 0.2)),
        ])
    ])


class DatasetLoader(DatasetTorch, ABC):
    def __init__(
            self,
            ds: Dataset,
            ds_args: DatasetArgs,
            train_or_test: str,
    ):
        assert train_or_test in ['train', 'test']
        self.ds = ds
        self.train_or_test = train_or_test
        self.augment = ds_args.augment
        self._get_transforms()
        self.preload = ds_args.preload_from_database
        if self.preload:
            self._preload_data()

    @abstractmethod
    def _preload_data(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self) -> int:
        return self.ds.get_size(self.train_or_test)

    def _get_transforms(self):
        pass


def make_data_loader(
        dataset_loader: DatasetLoader,
        ds_args: DatasetArgs,
        batch_size: int,
        collate_fn: callable = None,
) -> DataLoader:
    """
    Instantiate a torch data loader.
    """
    loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=ds_args.n_dataloader_workers,
        drop_last=True,
        collate_fn=collate_fn
    )

    return loader


def load_dataset(dataset_args: DatasetArgs) -> Dataset:
    """
    Load an existing dataset from the database.
    """
    ds = None

    # If we have a dataset id then load this from the database
    if dataset_args.dataset_id is not None:
        ds = Dataset.objects.get(id=dataset_args.dataset_id)
    else:
        # Otherwise, try to find one matching the same parameters
        if dataset_args.dataset_type == DATASET_TYPE_EIGENTRACES:
            datasets = DatasetEigentraces.find_from_args(dataset_args)
        else:
            datasets = Dataset.find_from_args(dataset_args)
        if datasets.count() > 0:
            ds = datasets[0]
            logger.info(f'Found {datasets.count()} suitable datasets in database, using most recent.')

    if ds is None:
        raise DoesNotExist()

    logger.info(f'Loaded dataset (id={ds.id}, created={ds.created}).')

    return ds
