from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from simple_worm.control_gates import ControlGateNumpy
from simple_worm.controls import CONTROL_KEYS, ControlsNumpy
from simple_worm.plot3d import plot_gates
from wormlab3d.simple_worm.args import SimulationArgs
from wormlab3d.simple_worm.args.parse import parse_arguments
from wormlab3d.toolkit.util import print_args

N = 100


def parse_arguments(printout: bool = True) -> SimulationArgs:
    parser = ArgumentParser()
    SimulationArgs.add_args(parser)
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Add dummy args
    args.worm_length = N
    args.duration = 1
    args.dt = 0.1

    simulation_args = SimulationArgs.from_args(args)
    return simulation_args


def plot_gates_from_args():
    """
    Test script for checking the control gates using command line arguments.
    """
    simulation_args = parse_arguments()

    # Build control gates
    gates = {}
    gate_arg_keys = ['block', 'grad_up', 'offset_up', 'grad_down', 'offset_down']
    for k in CONTROL_KEYS:
        gate_args = {
            gak: getattr(simulation_args, f'{k}_gate_{gak}')
            for gak in gate_arg_keys
        }
        if any([v is not None for v in gate_args.values()]):
            gates[f'{k}_gate'] = ControlGateNumpy(N=N, **gate_args)

    # Generate optimisable control sequence
    CS = ControlsNumpy(
        alpha=np.zeros(N),
        beta=np.zeros(N),
        gamma=np.zeros(N - 1),
        **gates
    )

    plot_gates(CS)
    plt.show()


if __name__ == '__main__':
    plot_gates_from_args()
