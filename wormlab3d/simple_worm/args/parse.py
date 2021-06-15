from argparse import ArgumentParser
from typing import Tuple

from wormlab3d.simple_worm.args import FrameSequenceArgs, OptimiserArgs, RegularisationArgs, RuntimeArgs, SimulationArgs
from wormlab3d.toolkit.util import print_args


def parse_arguments(printout: bool = True) \
        -> Tuple[RuntimeArgs, FrameSequenceArgs, SimulationArgs, OptimiserArgs, RegularisationArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Inverse optimisation')

    # Add the different argument types
    RuntimeArgs.add_args(parser)
    FrameSequenceArgs.add_args(parser)
    SimulationArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RegularisationArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)
    frame_sequence_args = FrameSequenceArgs.from_args(args)
    simulation_args = SimulationArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)
    regularisation_args = RegularisationArgs.from_args(args)

    return runtime_args, frame_sequence_args, simulation_args, optimiser_args, regularisation_args
