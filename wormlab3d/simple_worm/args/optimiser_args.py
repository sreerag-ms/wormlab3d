from argparse import ArgumentParser, _ArgumentGroup
from typing import Dict

from simple_worm.material_parameters import MP_KEYS
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            batch_size: int,
            estimate_psi: bool,
            optimise_MP_K: bool,
            optimise_MP_K_rot: bool,
            optimise_MP_A: bool,
            optimise_MP_B: bool,
            optimise_MP_C: bool,
            optimise_MP_D: bool,
            optimise_F0: bool,
            optimise_CS: bool,
            init_noise_std_K: float,
            init_noise_std_K_rot: float,
            init_noise_std_A: float,
            init_noise_std_B: float,
            init_noise_std_C: float,
            init_noise_std_D: float,
            init_noise_std_psi0: float,
            init_noise_std_alpha: float,
            init_noise_std_beta: float,
            init_noise_std_gamma: float,
            inverse_opt_max_iter: int = 2,
            inverse_opt_tol: float = 1e-8,
            **kwargs
    ):
        self.batch_size = batch_size
        self.estimate_psi = estimate_psi
        self.optimise_MP_K = optimise_MP_K
        self.optimise_MP_K_rot = optimise_MP_K_rot
        self.optimise_MP_A = optimise_MP_A
        self.optimise_MP_B = optimise_MP_B
        self.optimise_MP_C = optimise_MP_C
        self.optimise_MP_D = optimise_MP_D
        self.optimise_F0 = optimise_F0
        self.optimise_CS = optimise_CS
        self.init_noise_std_K = init_noise_std_K
        self.init_noise_std_K_rot = init_noise_std_K_rot
        self.init_noise_std_A = init_noise_std_A
        self.init_noise_std_B = init_noise_std_B
        self.init_noise_std_C = init_noise_std_C
        self.init_noise_std_D = init_noise_std_D
        self.init_noise_std_psi0 = init_noise_std_psi0
        self.init_noise_std_alpha = init_noise_std_alpha
        self.init_noise_std_beta = init_noise_std_beta
        self.init_noise_std_gamma = init_noise_std_gamma
        self.inverse_opt_max_iter = inverse_opt_max_iter
        self.inverse_opt_tol = inverse_opt_tol

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')
        group.add_argument('--batch-size', type=int, default=1,
                           help='Number of optimisations to run concurrently from different initial starting points.')
        group.add_argument('--estimate-psi', type=str2bool, default=True,
                           help='Estimate initial value of psi from the initial midline position.')

        # Optimisation flags
        group.add_argument('--optimise-MP-K', type=str2bool, default=False,
                           help='Optimise the material parameter K.')
        group.add_argument('--optimise-MP-K-rot', type=str2bool, default=False,
                           help='Optimise the material parameter K_rot.')
        group.add_argument('--optimise-MP-A', type=str2bool, default=False,
                           help='Optimise the material parameter A.')
        group.add_argument('--optimise-MP-B', type=str2bool, default=False,
                           help='Optimise the material parameter B.')
        group.add_argument('--optimise-MP-C', type=str2bool, default=False,
                           help='Optimise the material parameter C.')
        group.add_argument('--optimise-MP-D', type=str2bool, default=False,
                           help='Optimise the material parameter D.')
        group.add_argument('--optimise-F0', type=str2bool, default=True,
                           help='Optimise the initial frame.')
        group.add_argument('--optimise-CS', type=str2bool, default=True,
                           help='Optimise the control sequence.')

        # Initial values
        group.add_argument('--init-noise-std-K', type=float, default=1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_K estimate with this standard deviation')
        group.add_argument('--init-noise-std-K-rot', type=float, default=0.1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_K_rot estimate with this standard deviation')
        group.add_argument('--init-noise-std-A', type=float, default=0.1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_A estimate with this standard deviation')
        group.add_argument('--init-noise-std-B', type=float, default=0.1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_B estimate with this standard deviation')
        group.add_argument('--init-noise-std-C', type=float, default=0.1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_C estimate with this standard deviation')
        group.add_argument('--init-noise-std-D', type=float, default=0.1,
                           help='Add normally-distributed, zero-mean noise to the initial MP_D estimate with this standard deviation')
        group.add_argument('--init-noise-std-psi0', type=float, default=10,
                           help='Add normally-distributed, zero-mean noise to the initial psi0 estimate with this standard deviation.')
        group.add_argument('--init-noise-std-alpha', type=float, default=0.01,
                           help='Add normally-distributed, zero-mean noise to the initial alpha estimate with this standard deviation.')
        group.add_argument('--init-noise-std-beta', type=float, default=0.01,
                           help='Add normally-distributed, zero-mean noise to the initial beta estimate with this standard deviation.')
        group.add_argument('--init-noise-std-gamma', type=float, default=0.001,
                           help='Add normally-distributed, zero-mean noise to the initial gamma estimate with this standard deviation.')
        group.add_argument('--inverse-opt-max-iter', type=int, default=1,
                           help='Maximum number of inverse optimisation iterations per step.')
        group.add_argument('--inverse-opt-tol', type=float, default=1e-8,
                           help='Inverse optimisation stopping tolerance.')
        return group

    def get_mp_opt_flags(self, prefix: str = '') -> Dict[str, bool]:
        return {
            prefix + k: getattr(self, f'optimise_MP_{k}')
            for k in MP_KEYS
        }
