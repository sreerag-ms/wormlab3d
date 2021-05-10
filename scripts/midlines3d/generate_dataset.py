from typing import List

from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.midlines3d.generate_masks_dataset import generate_masks_dataset


def generate_dataset(
        checkpoint_id: str = None,
        train_test_split: float = 0.8,
        restrict_tags: List[int] = None,
        restrict_concs: List[float] = None,
        centre_3d_max_error: float = 50,
):
    dataset_params = DatasetSegmentationMasksArgs(
        masks_model_checkpoint_id=checkpoint_id,
        train_test_split=train_test_split,
        restrict_tags=restrict_tags,
        restrict_concs=restrict_concs,
        centre_3d_max_error=centre_3d_max_error,
    )
    generate_masks_dataset(dataset_params)


if __name__ == '__main__':
    generate_dataset(
        checkpoint_id='607edc8cd549270c25c73346',
        train_test_split=0.8,
        centre_3d_max_error=3
    )
