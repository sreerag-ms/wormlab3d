from argparse import ArgumentParser

from wormlab3d.nn.args import OptimiserArgs


class DynamicsOptimiserArgs(OptimiserArgs):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = OptimiserArgs.add_args(parser)
