from argparse import ArgumentParser, _ArgumentGroup
from typing import List

from wormlab3d.data.model.dataset import DATASET_TYPE_3D_MIDLINE
from wormlab3d.nn.args import DatasetArgs


class DatasetMidline3DArgs(DatasetArgs):
    def __init__(
            self,
            n_worm_points: int = -1,
            restrict_sources: List[str] = None,
            min_reconstruction_frames: int = None,
            mf_depth: int = -1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_type = DATASET_TYPE_3D_MIDLINE
        self.n_worm_points = n_worm_points
        if restrict_sources is None:
            restrict_sources = []
        self.min_reconstruction_frames = min_reconstruction_frames
        self.restrict_sources = restrict_sources
        self.mf_depth = mf_depth

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = DatasetArgs.add_args(parser)
        group.add_argument('--n-worm-points', type=int, default=128,
                           help='Number of coordinates to use to define the midline. May require resampling.')
        group.add_argument('--restrict-sources', type=lambda s: [str(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of sources.')
        group.add_argument('--min-reconstruction-frames', type=int,
                           help='Minimum reconstruction frames.')
        group.add_argument('--mf-depth', type=int, default=-1,
                           help='MidlineFinder depth. -1 = deepest available.')
        return group
