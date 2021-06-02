from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs

LOSS_MSE = 'mse'
LOSS_KL = 'kl'
LOSS_BCE = 'bce'

LOSSES = [
    LOSS_MSE,
    LOSS_KL,
    LOSS_BCE
]

OPTIMISER_ADADELTA = 'Adadelta'
OPTIMISER_ADAGRAD = 'Adagrad'
OPTIMISER_ADAM = 'Adam'
OPTIMISER_ADAMW = 'AdamW'
OPTIMISER_LBFGS = 'LBFGS'
OPTIMISER_RMSPROP = 'RMSprop'
OPTIMISER_SGD = 'SGD'

OPTIMISER_ALGORITHMS = [
    OPTIMISER_ADADELTA,
    OPTIMISER_ADAGRAD,
    OPTIMISER_ADAM,
    OPTIMISER_ADAMW,
    OPTIMISER_LBFGS,
    OPTIMISER_RMSPROP,
    OPTIMISER_SGD,
]


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            algorithm: str,
            loss: str = LOSS_MSE,
            lr_init: float = 0.1,
            lr_gamma: float = 0.1,
            weight_decay: float = 1e-5,
            **kwargs
    ):
        assert loss in LOSSES
        assert algorithm in OPTIMISER_ALGORITHMS
        self.loss = loss
        self.algorithm = algorithm
        self.lr_init = lr_init
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')
        group.add_argument('--loss', type=str, choices=LOSSES, default=LOSS_MSE,
                           help='The principal loss measure to minimise.')
        group.add_argument('--algorithm', type=str, choices=OPTIMISER_ALGORITHMS, default=OPTIMISER_ADAM,
                           help='Optimisation algorithm.')
        group.add_argument('--lr-init', type=float, default=0.1,
                           help='Initial learning rate.')
        group.add_argument('--lr-gamma', type=float, default=0.1,
                           help='Multiplicative factor of learning rate decay.')
        group.add_argument('--weight-decay', type=float, default=1e-5,
                           help='Weight decay.')
        return group
