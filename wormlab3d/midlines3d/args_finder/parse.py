from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.midlines3d.args_finder import ModelArgs, OptimiserArgs, RuntimeArgs, SourceArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) \
        -> Tuple[RuntimeArgs, SourceArgs, ModelArgs, OptimiserArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Inverse optimisation')

    # Add the different argument types
    RuntimeArgs.add_args(parser)
    SourceArgs.add_args(parser)
    ModelArgs.add_args(parser)
    OptimiserArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)
    source_args = SourceArgs.from_args(args)
    model_args = ModelArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)

    return runtime_args, source_args, model_args, optimiser_args
