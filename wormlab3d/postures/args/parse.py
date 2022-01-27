from argparse import ArgumentParser

from wormlab3d.postures.args import DatasetMidline3DArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) -> DatasetMidline3DArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Postures scripts')

    # Add the different argument types
    DatasetMidline3DArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    dataset_args = DatasetMidline3DArgs.from_args(args)

    return dataset_args
