from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Dict

from wormlab3d.nn.args.base_args import BaseArgs


class SimulationArgs(BaseArgs):
    def __init__(
            self,
            sim_id: str = None,
            worm_length: int = 10,
            duration: float = None,
            dt: float = None,
            K: float = 2.,
            K_rot: float = 1.,
            A: float = 1.,
            B: float = 0.,
            C: float = 1.,
            D: float = 0.,
            **kwargs
    ):
        if sim_id is None:
            assert all(v is not None for v in [worm_length, duration, dt]), \
                'Simulation parameters not well defined.'
        else:
            assert all(v is None for v in [worm_length, duration, dt]), \
                'A simulation id will override any command-line simulation arguments.'
        self.sim_id = sim_id

        # Simulation parameters
        self.worm_length = worm_length
        self.duration = duration
        self.dt = dt

        # Material parameters
        self.K = K
        self.K_rot = K_rot
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Simulation Args')
        group.add_argument('--sim-id', type=str,
                           help='Load a simulation configuration by its database id.')
        group.add_argument('--worm-length', type=int, default=10,
                           help='Number of worm points along the body (default=10).')
        group.add_argument('--duration', type=float,
                           help='Time (in seconds) to run the simulation for.')
        group.add_argument('--dt', type=float,
                           help='Simulation timestep.')
        group.add_argument('--K', type=float, default=2.,
                           help='The external force exerted on the worm by the fluid (default=2).')
        group.add_argument('--K_rot', type=float, default=1.,
                           help='The external moment (default=1).')
        group.add_argument('--A', type=float, default=1.,
                           help='The bending modulus (default=1).')
        group.add_argument('--B', type=float, default=0.,
                           help='The bending viscosity (default=0).')
        group.add_argument('--C', type=float, default=1.,
                           help='The twisting modulus (default=1).')
        group.add_argument('--D', type=float, default=0.,
                           help='The twisting viscosity (default=0).')
        return group

    def get_config_dict(self) -> Dict[str, Union[int, float]]:
        return {
            'worm_length': self.worm_length,
            'duration': self.duration,
            'dt': self.dt,
            'K': self.K,
            'K_rot': self.K_rot,
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'D': self.D,
        }
