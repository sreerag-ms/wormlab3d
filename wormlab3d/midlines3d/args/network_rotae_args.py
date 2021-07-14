from argparse import ArgumentParser, Namespace

from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import str2bool


class RotAENetworkArgs(NetworkArgs):
    def __init__(
            self,
            args_c2d: NetworkArgs,
            args_c3d: NetworkArgs,
            args_discriminator: NetworkArgs,
            use_discriminator: bool = False,
            **kwargs
    ):
        super().__init__(base_net='rotae', **kwargs)
        self.args_c2d = args_c2d
        self.args_c3d = args_c3d
        self.args_discriminator = args_discriminator
        self.use_discriminator = use_discriminator

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Rotational Auto Encoder')

        group.add_argument('--net-id', type=str,
                           help='Load a network by its database id.')
        group.add_argument('--load-net', type=str2bool, default=True,
                           help='Try to load an existing network if available matching the given parameters.')
        group.add_argument('--use-discriminator', type=str2bool, default=False,
                           help='Build, train and use a discriminator network.')

        subparsers = parser.add_subparsers(title='Networks', dest='networks',
                                           help='Define the different networks: c2d, c3d and (optionally) disc.')
        NetworkArgs.add_args(subparsers.add_parser('c2d'), prefix='c2d')
        NetworkArgs.add_args(subparsers.add_parser('c3d'), prefix='c3d')
        NetworkArgs.add_args(subparsers.add_parser('disc'), prefix='disc')

    @classmethod
    def from_args(cls, args: Namespace) -> 'RotAENetworkArgs':
        """
        Create a RotAENetworkArgs instance from command-line arguments.
        """
        net_args = {}
        hyperparameters = {}

        # Create a dummy namespace for each of the sub-networks and extract the parameters.
        for prefix in ['c2d', 'c3d', 'discriminator']:
            args_i = Namespace()
            if prefix == 'discriminator':
                if not args.use_discriminator:
                    net_args[prefix] = args_i
                    continue
            assert hasattr(args, f'{prefix}_base_net'), f'{prefix} network parameters not defined.'
            for k, v in vars(args).items():
                if k[:len(prefix) + 1] == prefix + '_':
                    setattr(args_i, k[len(prefix) + 1:], v)
            hps = NetworkArgs.extract_hyperparameter_args(args_i)
            hyperparameters[prefix] = hps
            args_i.hyperparameters = hps
            net_args[prefix] = args_i

        return RotAENetworkArgs(
            net_id=args.net_id,
            load=args.load_net,
            hyperparameters=hyperparameters,
            args_c2d=NetworkArgs.from_args(net_args['c2d']),
            args_c3d=NetworkArgs.from_args(net_args['c3d']),
            args_discriminator=net_args['discriminator'],
            use_discriminator=args.use_discriminator,
        )
