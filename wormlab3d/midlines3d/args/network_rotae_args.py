from argparse import ArgumentParser, Namespace

from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import str2bool


class RotAENetworkArgs(NetworkArgs):
    def __init__(
            self,
            args_c2d: NetworkArgs,
            args_c3d: NetworkArgs,
            args_d2d: NetworkArgs,
            args_d3d: NetworkArgs,
            use_d2d: bool = False,
            use_d3d: bool = False,
            **kwargs
    ):
        super().__init__(base_net='rotae', **kwargs)
        self.args_c2d = args_c2d
        self.args_c3d = args_c3d
        self.args_d2d = args_d2d
        self.args_d3d = args_d3d
        self.use_d2d = use_d2d
        self.use_d3d = use_d3d

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
        group.add_argument('--use-d2d', type=str2bool, default=False,
                           help='Build, train and use a discriminator network for 2D midlines.')
        group.add_argument('--use-d3d', type=str2bool, default=False,
                           help='Build, train and use a discriminator network for 3D midlines.')

        subparsers = parser.add_subparsers(title='Networks', dest='networks',
                                           help='Define the different networks: c2d, c3d and (optionally) disc.')
        NetworkArgs.add_args(subparsers.add_parser('c2d'), prefix='c2d')
        NetworkArgs.add_args(subparsers.add_parser('c3d'), prefix='c3d')
        NetworkArgs.add_args(subparsers.add_parser('d2d'), prefix='d2d')
        NetworkArgs.add_args(subparsers.add_parser('d3d'), prefix='d3d')

    @classmethod
    def from_args(cls, args: Namespace) -> 'RotAENetworkArgs':
        """
        Create a RotAENetworkArgs instance from command-line arguments.
        """
        net_args = {}
        hyperparameters = {}

        # Create a dummy namespace for each of the sub-networks and extract the parameters.
        for prefix in ['c2d', 'c3d', 'd2d', 'd3d']:
            args_i = Namespace()
            if prefix == 'd2d' and not args.use_d2d:
                net_args[prefix] = args_i
                continue
            if prefix == 'd3d' and not args.use_d3d:
                net_args[prefix] = args_i
                continue
            assert hasattr(args, f'{prefix}_base_net'), f'{prefix} network parameters not defined.'
            for k, v in vars(args).items():
                if k[:len(prefix) + 1] == prefix + '_':
                    setattr(args_i, k[len(prefix) + 1:], v)
            hps = NetworkArgs.extract_hyperparameter_args(args_i)
            hyperparameters[prefix] = hps
            args_i.hyperparameters = hps
            if (prefix == 'd2d' and not args.use_d2d) \
                    or (prefix == 'd3d' and not args.use_d3d):
                net_args[prefix] = None
            else:
                net_args[prefix] = NetworkArgs.from_args(args_i)

        return RotAENetworkArgs(
            net_id=args.net_id,
            load=args.load_net,
            hyperparameters=hyperparameters,
            args_c2d=net_args['c2d'],
            args_d2d=net_args['d2d'],
            args_c3d=net_args['c3d'],
            args_d3d=net_args['d3d'],
            use_d2d=args.use_d2d,
            use_d3d=args.use_d3d,
        )
