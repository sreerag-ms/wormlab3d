from argparse import ArgumentParser, Namespace

from wormlab3d.nn.args import NetworkArgs
from wormlab3d.toolkit.util import str2bool


class DynamicsNetworkArgs(NetworkArgs):
    def __init__(
            self,
            latent_size: int,
            args_classifier: NetworkArgs,
            args_dynamics: NetworkArgs,
            **kwargs
    ):
        super().__init__(base_net='dynamics_clusterer', **kwargs)
        self.latent_size = latent_size
        self.args_classifier = args_classifier
        self.args_dynamics = args_dynamics

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Dynamics clusterer')

        group.add_argument('--net-id', type=str,
                           help='Load a network by its database id.')
        group.add_argument('--load-net', type=str2bool, default=True,
                           help='Try to load an existing network if available matching the given parameters.')
        group.add_argument('--latent-size', type=int, default=4,
                           help='Latent encoding dimension.')

        subparsers = parser.add_subparsers(title='Networks', dest='networks',
                                           help='Define the different networks: classifier and dynamics.')
        NetworkArgs.add_args(subparsers.add_parser('classifier'), prefix='classifier')
        NetworkArgs.add_args(subparsers.add_parser('dynamics'), prefix='dynamics')

    @classmethod
    def from_args(cls, args: Namespace) -> 'DynamicsNetworkArgs':
        """
        Create a DynamicsNetworkArgs instance from command-line arguments.
        """
        net_args = {}
        hyperparameters = {}

        # Create a dummy namespace for each of the sub-networks and extract the parameters.
        for prefix in ['classifier', 'dynamics']:
            args_i = Namespace()
            assert hasattr(args, f'{prefix}_base_net'), f'{prefix} network parameters not defined.'
            for k, v in vars(args).items():
                if k[:len(prefix) + 1] == prefix + '_':
                    setattr(args_i, k[len(prefix) + 1:], v)
            hps = NetworkArgs.extract_hyperparameter_args(args_i)
            hyperparameters[prefix] = hps
            args_i.hyperparameters = hps
            net_args[prefix] = NetworkArgs.from_args(args_i)

        return DynamicsNetworkArgs(
            net_id=args.net_id,
            load=args.load_net,
            hyperparameters=hyperparameters,
            args_classifier=net_args['classifier'],
            args_dynamics=net_args['dynamics'],
            latent_size=args.latent_size
        )
