from argparse import ArgumentParser

from wormlab3d.nn.args import RuntimeArgs


class Midline3DRuntimeArgs(RuntimeArgs):
    def __init__(
            self,
            reprojection_blur_sigma: float = 0.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.reprojection_blur_sigma = reprojection_blur_sigma

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = RuntimeArgs.add_args(parser)
        group.add_argument('--reprojection-blur-sigma', type=float, default=0.,
                           help='Fatten the reprojected midline mask with a gaussian blur using this sigma value (in pixels).')
