from argparse import ArgumentParser, _ArgumentGroup
from typing import Union, Dict

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


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
            alpha_gate_block: bool = None,
            alpha_gate_grad_up: float = None,
            alpha_gate_offset_up: float = None,
            alpha_gate_grad_down: float = None,
            alpha_gate_offset_down: float = None,
            beta_gate_block: bool = None,
            beta_gate_grad_up: float = None,
            beta_gate_offset_up: float = None,
            beta_gate_grad_down: float = None,
            beta_gate_offset_down: float = None,
            gamma_gate_block: bool = None,
            gamma_gate_grad_up: float = None,
            gamma_gate_offset_up: float = None,
            gamma_gate_grad_down: float = None,
            gamma_gate_offset_down: float = None,
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

        # Control gates
        self.alpha_gate_block = alpha_gate_block
        self.alpha_gate_grad_up = alpha_gate_grad_up
        self.alpha_gate_offset_up = alpha_gate_offset_up
        self.alpha_gate_grad_down = alpha_gate_grad_down
        self.alpha_gate_offset_down = alpha_gate_offset_down
        self.beta_gate_block = beta_gate_block
        self.beta_gate_grad_up = beta_gate_grad_up
        self.beta_gate_offset_up = beta_gate_offset_up
        self.beta_gate_grad_down = beta_gate_grad_down
        self.beta_gate_offset_down = beta_gate_offset_down
        self.gamma_gate_block = gamma_gate_block
        self.gamma_gate_grad_up = gamma_gate_grad_up
        self.gamma_gate_offset_up = gamma_gate_offset_up
        self.gamma_gate_grad_down = gamma_gate_grad_down
        self.gamma_gate_offset_down = gamma_gate_offset_down

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
        group.add_argument('--alpha-gate-block', type=str2bool, default=None,
                           help='Block alpha controls completely.')
        group.add_argument('--alpha-gate-grad-up', type=float, default=None,
                           help='Alpha control gate steepness of turning on (head-to-tail).')
        group.add_argument('--alpha-gate-offset-up', type=float, default=None,
                           help='Alpha control gate offset for turning on (0=head, 1=tail).')
        group.add_argument('--alpha-gate-grad-down', type=float, default=None,
                           help='Alpha control gate steepness of turning off (head-to-tail).')
        group.add_argument('--alpha-gate-offset-down', type=float, default=None,
                           help='Alpha control gate offset for turning off (0=head, 1=tail).')
        group.add_argument('--beta-gate-block', type=str2bool, default=None,
                           help='Block beta controls completely.')
        group.add_argument('--beta-gate-grad-up', type=float, default=None,
                           help='Beta control gate steepness of turning on (head-to-tail).')
        group.add_argument('--beta-gate-offset-up', type=float, default=None,
                           help='Beta control gate offset for turning on (0=head, 1=tail).')
        group.add_argument('--beta-gate-grad-down', type=float, default=None,
                           help='Beta control gate steepness of turning off (head-to-tail).')
        group.add_argument('--beta-gate-offset-down', type=float, default=None,
                           help='Beta control gate offset for turning off (0=head, 1=tail).')
        group.add_argument('--gamma-gate-block', type=str2bool, default=None,
                           help='Block gamma controls completely.')
        group.add_argument('--gamma-gate-grad-up', type=float, default=None,
                           help='Gamma control gate steepness of turning on (head-to-tail).')
        group.add_argument('--gamma-gate-offset-up', type=float, default=None,
                           help='Gamma control gate offset for turning on (0=head, 1=tail).')
        group.add_argument('--gamma-gate-grad-down', type=float, default=None,
                           help='Gamma control gate steepness of turning off (head-to-tail).')
        group.add_argument('--gamma-gate-offset-down', type=float, default=None,
                           help='Gamma control gate offset for turning off (0=head, 1=tail).')
        return group

    def get_config_dict(self) -> Dict[str, Union[int, float]]:
        return {
            'worm_length': self.worm_length,
            'duration': self.duration,
            'dt': self.dt,
            'gates': {
                'alpha': {
                    'block': self.alpha_gate_block,
                    'grad_up': self.alpha_gate_grad_up,
                    'offset_up': self.alpha_gate_offset_up,
                    'grad_down': self.alpha_gate_grad_down,
                    'offset_down': self.alpha_gate_offset_down,
                },
                'beta': {
                    'block': self.beta_gate_block,
                    'grad_up': self.beta_gate_grad_up,
                    'offset_up': self.beta_gate_offset_up,
                    'grad_down': self.beta_gate_grad_down,
                    'offset_down': self.beta_gate_offset_down,
                },
                'gamma': {
                    'block': self.gamma_gate_block,
                    'grad_up': self.gamma_gate_grad_up,
                    'offset_up': self.gamma_gate_offset_up,
                    'grad_down': self.gamma_gate_grad_down,
                    'offset_down': self.gamma_gate_offset_down,
                }
            },

            # Explicitly set MPs to None here
            'K': None,
            'K_rot': None,
            'A': None,
            'B': None,
            'C': None,
            'D': None,
        }

    def get_mp_dict(self) -> Dict[str, Union[int, float]]:
        return {
            'K': self.K,
            'K_rot': self.K_rot,
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'D': self.D,
        }
