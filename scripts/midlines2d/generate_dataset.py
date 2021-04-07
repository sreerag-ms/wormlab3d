from typing import List

from wormlab3d.midlines2d.generate_dataset import generate_dataset
from wormlab3d.midlines2d.args import DatasetArgs


def generate_2d_midlines_dataset(
        train_test_split: float = 0.8,
        restrict_tags: List[int] = None,
        restrict_concs: List[float] = None,
        centre_3d_max_error: float = 50,
):
    dataset_params = DatasetArgs(
        train_test_split=train_test_split,
        restrict_tags=restrict_tags,
        restrict_concs=restrict_concs,
        centre_3d_max_error=centre_3d_max_error,
        exclude_trials=[258,]
    )
    generate_dataset(dataset_params, fix_frames=True)


if __name__ == '__main__':
    generate_2d_midlines_dataset(
        train_test_split=0.8,
        centre_3d_max_error=None
    )
