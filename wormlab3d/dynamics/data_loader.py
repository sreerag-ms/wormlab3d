from typing import Tuple

import numpy as np
import torch
from bson import ObjectId
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DatasetTorch

from wormlab3d.data.model.dataset import DatasetEigentraces
from wormlab3d.dynamics.args import DynamicsDatasetArgs
from wormlab3d.nn.data_loader import make_data_loader


class DatasetEigentracesLoader(DatasetTorch):
    def __init__(
            self,
            ds: DatasetEigentraces,
            ds_args: DynamicsDatasetArgs,
            train_or_test: str,
    ):
        assert train_or_test in ['train', 'test']
        self.ds = ds
        self.ds_args = ds_args
        self.train_or_test = train_or_test
        self.X = getattr(ds, f'X_{train_or_test}')

    def __len__(self) -> int:
        """
        Rough approximation, allowing for 50% overlap.
        """
        return int(self.ds.get_size(self.train_or_test) / self.ds_args.sample_duration * 2)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ObjectId, torch.Tensor]:
        """
        Fetch an eigentrace sample at random.
        """
        sequence = self.X[np.random.randint(self.ds.n_sequences)]
        ws = self.ds_args.sample_duration
        tries = 0
        max_tries = 10
        while len(sequence) < ws:
            sequence = self.X[np.random.randint(self.ds.n_sequences)]
            tries += 1
            if tries >= max_tries:
                raise RuntimeError(
                    f'Cannot find a sufficiently long sequence to sample from after {max_tries} attempts.'
                )
        start = np.random.randint(len(sequence) - ws)
        X = torch.from_numpy(sequence[start:start + ws].T).to(torch.float32)

        return X


def get_data_loader(
        ds: DatasetEigentraces,
        ds_args: DynamicsDatasetArgs,
        train_or_test: str,
        batch_size: int
) -> DataLoader:
    """
    Get a data loader.
    """
    assert train_or_test in ['train', 'test']

    dataset_loader = DatasetEigentracesLoader(
        ds=ds,
        ds_args=ds_args,
        train_or_test=train_or_test,
    )

    loader = make_data_loader(
        dataset_loader=dataset_loader,
        ds_args=ds_args,
        batch_size=batch_size,
    )

    return loader
