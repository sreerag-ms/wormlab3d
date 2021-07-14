from argparse import ArgumentParser
from typing import Tuple, Union

from wormlab3d.midlines3d.args import *
from wormlab3d.nn.args import OptimiserArgs, RuntimeArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(rotae: bool = False, printout: bool = True) \
        -> Tuple[
            Union[DatasetSegmentationMasksArgs, DatasetRotAECoordsArgs],
            Union[Midline3DNetworkArgs, RotAENetworkArgs],
            Union[Midline3DRuntimeArgs, RuntimeArgs],
            Union[OptimiserArgs, RotAEOptimiserArgs]
        ]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test 3D midline detector')

    # Add the different argument types
    if rotae:
        DatasetRotAECoordsArgs.add_args(parser)
        RotAENetworkArgs.add_args(parser)
        RuntimeArgs.add_args(parser)
        RotAEOptimiserArgs.add_args(parser)
    else:
        DatasetSegmentationMasksArgs.add_args(parser)
        Midline3DNetworkArgs.add_args(parser)
        Midline3DRuntimeArgs.add_args(parser)
        OptimiserArgs.add_args(parser)

    # Do the parsing
    def parse_args():
        args, argv = parser.parse_known_args()

        # If arguments remain, try parsing again. This allows parsing of multiple subparsers.
        if argv:
            args, argv = parser.parse_known_args(args=argv, namespace=args)
        if argv:
            raise RuntimeError(f'Unrecognized arguments: {" ".join(argv)}')
        return args

    args = parse_args()

    if printout:
        print_args(args)

    # Instantiate the parameter holders
    if rotae:
        dataset_args = DatasetRotAECoordsArgs.from_args(args)
        net_args = RotAENetworkArgs.from_args(args)
        runtime_args = RuntimeArgs.from_args(args)
        optimiser_args = RotAEOptimiserArgs.from_args(args)
    else:
        dataset_args = DatasetSegmentationMasksArgs.from_args(args)
        net_args = Midline3DNetworkArgs.from_args(args)
        runtime_args = Midline3DRuntimeArgs.from_args(args)
        optimiser_args = OptimiserArgs.from_args(args)

    return dataset_args, net_args, runtime_args, optimiser_args
