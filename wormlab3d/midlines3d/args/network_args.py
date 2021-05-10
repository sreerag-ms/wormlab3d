from argparse import ArgumentParser, Namespace

from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import str2bool

ENCODING_MODE_DELTA_VECTORS = 'delta_vectors'
ENCODING_MODE_DELTA_ANGLES = 'delta_angles'
ENCODING_MODE_DELTA_ANGLES_BASIS = 'delta_angles_basis'
ENCODING_MODES = [
    ENCODING_MODE_DELTA_VECTORS,
    ENCODING_MODE_DELTA_ANGLES,
    ENCODING_MODE_DELTA_ANGLES_BASIS
]

MAX_DECAY_FACTOR = 10


class Midline3DNetworkArgs(NetworkArgs):
    def __init__(
            self,
            encoding_mode: str = ENCODING_MODE_DELTA_ANGLES,
            n_basis_fns: int = 0,
            use_discriminator: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        if encoding_mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
            assert n_basis_fns > 0, \
                f'Must specify the number of basis functions to use when encoding when using `{ENCODING_MODE_DELTA_ANGLES_BASIS}` encoding.'

        self.encoding_mode = encoding_mode
        self.n_basis_fns = n_basis_fns
        self.use_discriminator = use_discriminator

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = NetworkArgs.add_args(parser)
        group.add_argument('--encoding-mode', type=str, choices=ENCODING_MODES, default=ENCODING_MODE_DELTA_ANGLES,
                           help='Pick the encoding mode option.')
        group.add_argument('--n-basis-fns', type=int, default=0,
                           help='Select the number of basis functions to use when encoding to basis coefficients.')
        group.add_argument('--use-discriminator', type=str2bool, default=False,
                           help='Build, train and use a discriminator network.')

    @classmethod
    def from_args(cls, args: Namespace) -> 'Midline3DNetworkArgs':
        """
        Create a Midline3DNetworkArgs instance from command-line arguments.
        """
        hyperparameters = NetworkArgs.extract_hyperparameter_args(args)

        return Midline3DNetworkArgs(
            net_id=args.net_id,
            load=args.load_dataset,
            base_net=args.base_net,
            hyperparameters=hyperparameters,
            encoding_mode=args.encoding_mode,
            n_basis_fns=args.n_basis_fns,
            use_discriminator=args.use_discriminator,
        )
