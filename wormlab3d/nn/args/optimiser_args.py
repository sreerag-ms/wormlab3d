from argparse import ArgumentParser, Namespace

OPTIMISER_ADADELTA = 'Adadelta'
OPTIMISER_ADAGRAD = 'Adagrad'
OPTIMISER_ADAM = 'Adam'
OPTIMISER_LBFGS = 'LBFGS'
OPTIMISER_RMSPROP = 'RMSprop'
OPTIMISER_SGD = 'SGD'

OPTIMISER_ALGORITHMS = [
    OPTIMISER_ADADELTA,
    OPTIMISER_ADAGRAD,
    OPTIMISER_ADAM,
    OPTIMISER_LBFGS,
    OPTIMISER_RMSPROP,
    OPTIMISER_SGD,
]


class OptimiserArgs:
    def __init__(
            self,
            algorithm: str,
            lr_init: float = 0.1,
            lr_gamma: float = 0.1,
            weight_decay: float = 1e-5,
    ):
        assert algorithm in OPTIMISER_ALGORITHMS
        self.algorithm = algorithm
        self.lr_init = lr_init
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        parser.add_argument('--algorithm', type=str, choices=OPTIMISER_ALGORITHMS, default=OPTIMISER_ADAM,
                            help='Optimisation algorithm.')
        parser.add_argument('--lr-init', type=float, default=0.1,
                            help='Initial learning rate.')
        parser.add_argument('--lr-gamma', type=float, default=0.1,
                            help='Multiplicative factor of learning rate decay.')
        parser.add_argument('--weight-decay', type=float, default=1e-5,
                            help='Weight decay.')

    @staticmethod
    def from_args(args: Namespace) -> 'OptimiserArgs':
        """
        Create a OptimiserParameters instance from command-line arguments.
        """
        return OptimiserArgs(
            algorithm=args.algorithm,
            lr_init=args.lr_init,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )
