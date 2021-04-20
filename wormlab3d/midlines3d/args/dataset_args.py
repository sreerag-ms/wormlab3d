from argparse import ArgumentParser

from wormlab3d.data.model.dataset import DATASET_TYPE_SEGMENTATION_MASKS
from wormlab3d.nn.args import DatasetArgs


class DatasetSegmentationMasksArgs(DatasetArgs):
    def __init__(
            self,
            masks_model_checkpoint_id: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_type = DATASET_TYPE_SEGMENTATION_MASKS
        self.masks_model_checkpoint_id = masks_model_checkpoint_id

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = DatasetArgs.add_args(parser)
        group.add_argument('--masks-model-checkpoint-id', type=str, default=None,
                           help='Load results from this checkpoint')
