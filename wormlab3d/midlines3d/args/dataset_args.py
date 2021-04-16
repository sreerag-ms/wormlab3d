from argparse import ArgumentParser, Namespace

from wormlab3d.data.model.dataset import DATASET_TYPE_SEGMENTATION_MASKS
from wormlab3d.nn.args import DatasetArgs


class DatasetSegmentationMasksArgs(DatasetArgs):
    def __init__(
            self,
            blur_sigma: float = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        # todo: self.blur_sigma = blur_sigma
        self.dataset_type = DATASET_TYPE_SEGMENTATION_MASKS

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        DatasetArgs.add_args(parser)
        # todo
        # parser.add_argument('--blur-sigma', type=float, default=0,
        #                     help='Fatten the midline mask with a gaussian blur using this sigma value (in pixels).')

    @staticmethod
    def from_args(args: Namespace) -> 'DatasetSegmentationMasksArgs':
        """
        Create a DatasetArgs instance from command-line arguments.
        """
        return DatasetSegmentationMasksArgs(**args)
