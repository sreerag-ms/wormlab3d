from argparse import ArgumentParser

from wormlab3d.nn.args import RuntimeArgs
from wormlab3d.toolkit.util import str2bool


class DynamicsRuntimeArgs(RuntimeArgs):
    def __init__(
            self,
            save_plots: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_plots = save_plots

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = RuntimeArgs.add_args(parser)
        group.add_argument('--save-plots', type=str2bool, default=False,
                           help='Save plot images to disk. Default = False.')
