from typing import List

from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.midlines2d.generate_midline2d_dataset import generate_midline2d_dataset


def generate_dataset(
        train_test_split: float = 0.8,
        restrict_tags: List[int] = None,
        restrict_concs: List[float] = None,
        centre_3d_max_error: float = 50,
):
    dataset_params = DatasetMidline2DArgs(
        train_test_split=train_test_split,
        restrict_tags=restrict_tags,
        restrict_concs=restrict_concs,
        centre_3d_max_error=centre_3d_max_error,
        exclude_trials=[258, 183, 88, 115, 86, 29, 84, 85, 218]
    )
    generate_midline2d_dataset(dataset_params, fix_frames=True)


if __name__ == '__main__':
    generate_dataset(
        train_test_split=0.8,
        centre_3d_max_error=None
    )
