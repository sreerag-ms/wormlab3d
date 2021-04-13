from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) -> Tuple[DatasetMidline2DArgs, NetworkArgs, RuntimeArgs, OptimiserArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test 2D midline detector')

    # Add the different argument types
    DatasetMidline2DArgs.add_args(parser)
    NetworkArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    dataset_args = DatasetMidline2DArgs.from_args(args)
    net_args = NetworkArgs.from_args(args)
    runtime_args = RuntimeArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)

    return dataset_args, net_args, runtime_args, optimiser_args
