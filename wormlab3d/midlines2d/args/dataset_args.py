from argparse import ArgumentParser

from wormlab3d.data.model.dataset import DATASET_TYPE_2D_MIDLINE
from wormlab3d.nn.args import DatasetArgs


class DatasetMidline2DArgs(DatasetArgs):
    def __init__(
            self,
            blur_sigma: float = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.blur_sigma = blur_sigma
        self.dataset_type = DATASET_TYPE_2D_MIDLINE

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = DatasetArgs.add_args(parser)
        group.add_argument('--blur-sigma', type=float, default=0,
                           help='Fatten the midline mask with a gaussian blur using this sigma value (in pixels).')
