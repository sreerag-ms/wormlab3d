from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            batch_size: int,
            estimate_psi: bool,
            optimise_F0: bool,
            optimise_CS: bool,
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
        self.optimise_F0 = optimise_F0
        self.optimise_CS = optimise_CS
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
        group.add_argument('--optimise-F0', type=str2bool, default=True,
                           help='Optimise the initial frame.')
        group.add_argument('--optimise-CS', type=str2bool, default=True,
                           help='Optimise the control sequence.')
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
