from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.dynamics.args import DynamicsDatasetArgs, DynamicsNetworkArgs, DynamicsOptimiserArgs, DynamicsRuntimeArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(
        printout: bool = True
) -> Tuple[DynamicsDatasetArgs, DynamicsNetworkArgs, DynamicsOptimiserArgs, DynamicsRuntimeArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test dynamics clusterer.')

    # Add the different argument types
    DynamicsRuntimeArgs.add_args(parser)
    DynamicsDatasetArgs.add_args(parser)
    DynamicsNetworkArgs.add_args(parser)
    DynamicsOptimiserArgs.add_args(parser)

    # Do the parsing
    def parse_args():
        args, argv = parser.parse_known_args()

        # If arguments remain, try parsing again. This allows parsing of multiple subparsers.
        max_attempts = 10
        attempt = 1
        while argv and attempt < max_attempts:
            args, argv = parser.parse_known_args(args=argv, namespace=args)
            attempt += 1
        if argv:
            raise RuntimeError(f'Unrecognized arguments: {" ".join(argv)}')
        return args

    args = parse_args()
    if printout:
        print_args(args)

    runtime_args = DynamicsRuntimeArgs.from_args(args)
    dataset_args = DynamicsDatasetArgs.from_args(args)
    net_args = DynamicsNetworkArgs.from_args(args)
    optimiser_args = DynamicsOptimiserArgs.from_args(args)

    return runtime_args, dataset_args, net_args, optimiser_args
