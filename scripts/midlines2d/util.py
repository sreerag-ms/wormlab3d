from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.midlines2d.args import DatasetArgs, NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) -> Tuple[DatasetArgs, NetworkArgs, RuntimeArgs, OptimiserArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test 2D midline detector')

    # Add the different argument types
    DatasetArgs.add_args(parser)
    NetworkArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    dataset_args = DatasetArgs.from_args(args)
    net_args = NetworkArgs.from_args(args)
    runtime_args = RuntimeArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)

    return dataset_args, net_args, runtime_args, optimiser_args
