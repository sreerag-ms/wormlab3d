from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.midlines3d.args_finder import ParameterArgs, RuntimeArgs, SourceArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) \
        -> Tuple[RuntimeArgs, SourceArgs, ParameterArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='3D Midline Finder')

    # Add the different argument types
    RuntimeArgs.add_args(parser)
    SourceArgs.add_args(parser)
    ParameterArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)
    source_args = SourceArgs.from_args(args)
    parameter_args = ParameterArgs.from_args(args)

    return runtime_args, source_args, parameter_args
