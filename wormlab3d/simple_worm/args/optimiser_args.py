from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            optimise_F0: bool,
            optimise_CS: bool,
            inverse_opt_max_iter: int = 2,
            inverse_opt_tol: float = 1e-8,
            **kwargs
    ):
        self.optimise_F0 = optimise_F0
        self.optimise_CS = optimise_CS
        self.inverse_opt_max_iter = inverse_opt_max_iter
        self.inverse_opt_tol = inverse_opt_tol

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')
        group.add_argument('--optimise-F0', type=str2bool, default=True,
                           help='Optimise the initial frame.')
        group.add_argument('--optimise-CS', type=str2bool, default=True,
                           help='Optimise the control sequence.')
        group.add_argument('--inverse-opt-max-iter', type=int, default=1,
                           help='Maximum number of inverse optimisation iterations per step.')
        group.add_argument('--inverse-opt-tol', type=float, default=1e-8,
                           help='Inverse optimisation stopping tolerance.')
        return group
