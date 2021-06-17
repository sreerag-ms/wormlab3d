from argparse import ArgumentParser

from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.nn.args import DatasetArgs


class DatasetMidline2DCoordsArgs(DatasetMidline2DArgs):
    def __init__(
            self,
            n_worm_points: int = 20,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_worm_points = n_worm_points

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = DatasetArgs.add_args(parser)
        group.add_argument('--n-worm-points', type=int, default=20,
                           help='Number of coordinates used to define the midline.')
