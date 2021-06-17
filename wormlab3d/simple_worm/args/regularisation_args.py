from argparse import ArgumentParser, _ArgumentGroup
from typing import Dict

from wormlab3d.nn.args.base_args import BaseArgs


class RegularisationArgs(BaseArgs):
    def __init__(
            self,
            reg_id: str = None,
            l2_alpha: float = 0.,
            l2_beta: float = 0.,
            l2_gamma: float = 0.,
            grad_t_alpha: float = 0.,
            grad_t_beta: float = 0.,
            grad_t_gamma: float = 0.,
            grad_x_alpha: float = 0.,
            grad_x_beta: float = 0.,
            grad_x_psi0: float = 0.,
            **kwargs
    ):
        self.reg_id = reg_id
        self.l2_alpha = l2_alpha
        self.l2_beta = l2_beta
        self.l2_gamma = l2_gamma
        self.grad_t_alpha = grad_t_alpha
        self.grad_t_beta = grad_t_beta
        self.grad_t_gamma = grad_t_gamma
        self.grad_x_alpha = grad_x_alpha
        self.grad_x_beta = grad_x_beta
        self.grad_x_psi0 = grad_x_psi0

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Regularisation Args')
        group.add_argument('--reg-id', type=str,
                           help='Load a regularisation configuration by its database id.')
        group.add_argument('--l2-alpha', type=float, default=0,
                           help='L2 regularisation penalty on the sum of alpha.')
        group.add_argument('--l2-beta', type=float, default=0,
                           help='L2 regularisation penalty on the sum of beta.')
        group.add_argument('--l2-gamma', type=float, default=0,
                           help='L2 regularisation penalty on the sum of gamma.')
        group.add_argument('--grad-t-alpha', type=float, default=0,
                           help='Smoothness in time regularisation penalty on alpha.')
        group.add_argument('--grad-t-beta', type=float, default=0,
                           help='Smoothness in time regularisation penalty on beta.')
        group.add_argument('--grad-t-gamma', type=float, default=0,
                           help='Smoothness in time regularisation penalty on gamma.')
        group.add_argument('--grad-x-alpha', type=float, default=0,
                           help='Smoothness along the body regularisation penalty on alpha.')
        group.add_argument('--grad-x-beta', type=float, default=0,
                           help='Smoothness along the body regularisation penalty on beta.')
        group.add_argument('--grad-x-psi0', type=float, default=0,
                           help='Smoothness along the body regularisation penalty on psi0 (initial twist).')

        return group

    def get_reg_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Convert the regularisation parameters to a nested dictionary of weights.
        """
        return {
            'L2': {
                'alpha': self.l2_alpha,
                'beta': self.l2_beta,
                'gamma': self.l2_gamma,
            },
            'grad_t': {
                'alpha': self.grad_t_alpha,
                'beta': self.grad_t_beta,
                'gamma': self.grad_t_gamma,
            },
            'grad_x': {
                'alpha': self.grad_x_alpha,
                'beta': self.grad_x_beta,
                'psi0': self.grad_x_psi0,
            }
        }
