from argparse import ArgumentParser
from typing import Tuple, Union

from wormlab3d.midlines2d.args import DatasetMidline2DArgs, DatasetMidline2DCoordsArgs
from wormlab3d.nn.args import NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(coordinate_mode: bool = False, printout: bool = True) \
        -> Tuple[Union[DatasetMidline2DArgs, DatasetMidline2DCoordsArgs], NetworkArgs, RuntimeArgs, OptimiserArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test 2D midline detector')

    # Add the different argument types
    if coordinate_mode:
        DatasetMidline2DCoordsArgs.add_args(parser)
    else:
        DatasetMidline2DArgs.add_args(parser)
    NetworkArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    if coordinate_mode:
        dataset_args = DatasetMidline2DCoordsArgs.from_args(args)
    else:
        dataset_args = DatasetMidline2DArgs.from_args(args)
    net_args = NetworkArgs.from_args(args)
    runtime_args = RuntimeArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)

    return dataset_args, net_args, runtime_args, optimiser_args
