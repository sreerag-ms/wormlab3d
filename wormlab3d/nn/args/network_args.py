import sys
from argparse import ArgumentParser, Namespace

from wormlab3d.data.model.network_parameters import *
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool

NETWORK_TYPES = ['densenet', 'fcnet', 'resnet', 'pyramidnet', 'aenet', 'nunet', 'rdn', 'red', 'rotae']


class NetworkArgs(BaseArgs):
    def __init__(
            self,
            net_id: str = None,
            load: bool = True,
            base_net: str = None,
            hyperparameters: dict = None,
            **kwargs
    ):
        assert net_id is not None or base_net in NETWORK_TYPES
        self.load = load
        self.net_id = net_id
        self.base_net = base_net
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str = None):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group(
            'Network Args' + (f'_Prefix={prefix}' if prefix is not None else '')
        )
        if prefix is not None:
            prefix = prefix + '-'
        else:
            prefix = ''
        group.add_argument(f'--{prefix}net-id', type=str,
                           help='Load a network by its database id.')
        group.add_argument(f'--{prefix}load-net', type=str2bool, default=True,
                           help='Try to load an existing network if available matching the given parameters.')

        # Add network-specific options
        if f'--{prefix}net-id' not in sys.argv:
            # if prefix == '':
            #     subparsers = parser.add_subparsers(title='Base network', dest=f'base_net',
            #                                        help='What type of network to use as the base.')
            #     subparsers.required = True
            # else:
            subparsers = parser.add_subparsers(title='Base network', dest=f'{prefix.replace("-", "_")}base_net',
                                               help='What type of network to use as the base.')
            subparsers.required = True
            NetworkArgs._add_fcnet_args(subparsers.add_parser('fcnet'), prefix)
            NetworkArgs._add_aenet_args(subparsers.add_parser('aenet'), prefix)
            NetworkArgs._add_resnet_args(subparsers.add_parser('resnet'), prefix)
            NetworkArgs._add_densenet_args(subparsers.add_parser('densenet'), prefix)
            NetworkArgs._add_pyramidnet_args(subparsers.add_parser('pyramidnet'), prefix)
            NetworkArgs._add_nunet_args(subparsers.add_parser('nunet'), prefix)
            NetworkArgs._add_rdn_args(subparsers.add_parser('rdn'), prefix)
            NetworkArgs._add_red_args(subparsers.add_parser('red'), prefix)

        return group

    @staticmethod
    def _add_fcnet_args(parser: Namespace, prefix: str = None):
        """
        Fully-connected network parameters.
        """
        parser.add_argument(f'--{prefix}layers-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='100,100',
                            help='Comma delimited list of layer sizes.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _parse_fcnet_params(parser, prefix: str = None):
        """
        Fully-connected network parameters.
        """
        parser.add_argument(f'--{prefix}layers-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='100,100',
                            help='Comma delimited list of layer sizes.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_aenet_args(parser, prefix: str = None):
        """
        Auto-encoder network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}growth-rate', type=int, default=8,
                            help='Growth rate for each layer in the blocks (k).')
        parser.add_argument(f'--{prefix}compression-factor', type=float, default=0.5,
                            help='Factor to reduce resolution by in transition layers (theta).')
        parser.add_argument(f'--{prefix}block-config-enc', type=lambda s: [int(item) for item in s.split(',')],
                            default='6,6,6',
                            help='Comma delimited list of layers for each encoder block. Number of entries determines number of encoder blocks.')
        parser.add_argument(f'--{prefix}block-config-dec', type=lambda s: [int(item) for item in s.split(',')],
                            default='6,6,6',
                            help='Comma delimited list of layers for each decoder block. Number of entries determines number of decoder blocks.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')
        parser.add_argument(f'--{prefix}latent-size', type=int, default=10,
                            help='Number of additional latent variables to store.')

    @staticmethod
    def _add_resnet_args(parser, prefix: str = None):
        """
        ResNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=64,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='3,4,6,3',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}shortcut-type', type=str, choices=['id', 'conv'], default='id',
                            help='Shortcut operation to use when dimensions change.')
        parser.add_argument(f'--{prefix}use-bottlenecks', type=str2bool, default=False,
                            help='Use bottleneck type residual layers.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_densenet_args(parser, prefix: str = None):
        """
        DenseNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}growth-rate', type=int, default=8,
                            help='Growth rate for each layer in the blocks (k).')
        parser.add_argument(f'--{prefix}compression-factor', type=float, default=0.5,
                            help='Factor to reduce resolution by in transition layers (theta).')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='6,6,6',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_pyramidnet_args(parser, prefix: str = None):
        """
        PyramidNet network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                            default='3,4,6,3',
                            help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
        parser.add_argument(f'--{prefix}alpha', type=int, default=420,
                            help='The widening factor which defines how quickly the pyramid expands at each layer.')
        parser.add_argument(f'--{prefix}shortcut-type', type=str, choices=['id', 'conv'], default='id',
                            help='Shortcut operation to use when dimensions change.')
        parser.add_argument(f'--{prefix}use-bottlenecks', type=str2bool, default=False,
                            help='Use bottleneck type residual layers.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_nunet_args(parser, prefix: str = None):
        """
        U-Net variant (nunet) network parameters.
        """
        parser.add_argument(f'--{prefix}n-init-channels', type=int, default=16,
                            help='Number of channels in the first input layer.')
        parser.add_argument(f'--{prefix}depth', type=int, default=5,
                            help='Network depth.')
        parser.add_argument(f'--{prefix}up-mode', type=str, choices=['upconv', 'upsample'], default='upconv',
                            help='How to increase spatial/temporal dims. upconv: transposed convolutions, upsample: bilinear upsampling')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0.2,
                            help='Dropout probability.')

    @staticmethod
    def _add_rdn_args(parser, prefix: str = None):
        """
        Residual Dense Network (RDN) network parameters.
        """
        parser.add_argument(f'--{prefix}D', type=int, default=1,
                            help='Multiscale depth. Defaults to 1 (no multiscale).')
        parser.add_argument(f'--{prefix}K', type=int, default=16,
                            help='Primary number of channels.')
        parser.add_argument(f'--{prefix}M', type=int, default=5,
                            help='Number of RDBs (Residual Dense Blocks).')
        parser.add_argument(f'--{prefix}N', type=int, default=3,
                            help='Number of convolution layers in each RDB.')
        parser.add_argument(f'--{prefix}G', type=int, default=3,
                            help='Growth rate in each RDB - how many channels each convolution layer adds.')
        parser.add_argument(f'--{prefix}kernel-size', type=int, default=3,
                            help='Spatial convolution size.')
        parser.add_argument(f'--{prefix}activation', type=str, default='relu',
                            help='Activation function, relu, elu, gelu.')
        parser.add_argument(f'--{prefix}act-out', type=str, default=False,
                            help='Apply activation at output, eg tanh.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0,
                            help='Dropout probability.')
        parser.add_argument(f'--{prefix}batch-norm', type=str2bool, default=False,
                            help='Apply batch normalisation after convolution and activation.')

    @staticmethod
    def _add_red_args(parser, prefix: str = None):
        """
        Recursive Encoding Discriminator (RED) network parameters.
        """
        parser.add_argument(f'--{prefix}latent-size', type=int, required=True,
                            help='Size of latent representation vector.')
        parser.add_argument(f'--{prefix}K', type=int, default=16,
                            help='Primary number of channels.')
        parser.add_argument(f'--{prefix}M', type=int, default=5,
                            help='Number of RDBs (Residual Dense Blocks).')
        parser.add_argument(f'--{prefix}N', type=int, default=3,
                            help='Number of convolution layers in each RDB.')
        parser.add_argument(f'--{prefix}G', type=int, default=3,
                            help='Growth rate in each RDB - how many channels each convolution layer adds.')
        parser.add_argument(f'--{prefix}discriminator-layers', type=lambda s: [int(item) for item in s.split(',')],
                            default='', help='Comma delimited list of discriminator layer sizes.')
        parser.add_argument(f'--{prefix}kernel-size', type=int, default=3,
                            help='Spatial convolution size.')
        parser.add_argument(f'--{prefix}activation', type=str, default='relu',
                            help='Activation function, relu, elu, gelu.')
        parser.add_argument(f'--{prefix}act-out', type=str, default=False,
                            help='Apply activation at output, eg tanh.')
        parser.add_argument(f'--{prefix}dropout-prob', type=float, default=0,
                            help='Dropout probability.')
        parser.add_argument(f'--{prefix}batch-norm', type=str2bool, default=False,
                            help='Apply batch normalisation after convolution and activation.')

    @staticmethod
    def extract_hyperparameter_args(args: Namespace) -> dict:
        """
        Create a NetworkParameters instance from command-line arguments.
        """
        if args.base_net == 'fcnet':
            hyperparameters = {
                'layers_config': args.layers_config,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'aenet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'growth_rate': args.growth_rate,
                'compression_factor': args.compression_factor,
                'block_config_enc': args.block_config_enc,
                'block_config_dec': args.block_config_dec,
                'latent_size': args.latent_size,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'resnet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'blocks_config': args.blocks_config,
                'shortcut_type': args.shortcut_type,
                'use_bottlenecks': args.use_bottlenecks,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'densenet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'growth_rate': args.growth_rate,
                'compression_factor': args.compression_factor,
                'blocks_config': args.blocks_config,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'pyramidnet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'blocks_config': args.blocks_config,
                'alpha': args.alpha,
                'shortcut_type': args.shortcut_type,
                'use_bottlenecks': args.use_bottlenecks,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'nunet':
            hyperparameters = {
                'n_init_channels': args.n_init_channels,
                'depth': args.depth,
                'up_mode': args.up_mode,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'rdn':
            hyperparameters = {
                'D': args.D,
                'K': args.K,
                'M': args.M,
                'N': args.N,
                'G': args.G,
                'kernel_size': args.kernel_size,
                'activation': args.activation,
                'act_out': args.act_out if args.act_out is not False else None,
                'batch_norm': args.batch_norm,
                'dropout_prob': args.dropout_prob,
            }

        elif args.base_net == 'red':
            hyperparameters = {
                'latent_size': args.latent_size,
                'K': args.K,
                'M': args.M,
                'N': args.N,
                'G': args.G,
                'discriminator_layers': args.discriminator_layers,
                'kernel_size': args.kernel_size,
                'activation': args.activation,
                'act_out': args.act_out if args.act_out is not False else None,
                'batch_norm': args.batch_norm,
                'dropout_prob': args.dropout_prob,
            }
        return hyperparameters

    @classmethod
    def from_args(cls, args: Namespace) -> 'NetworkArgs':
        """
        Create a NetworkParameters instance from command-line arguments.
        """
        hyperparameters = NetworkArgs.extract_hyperparameter_args(args)

        return NetworkArgs(
            net_id=args.net_id,
            load=args.load_net,
            base_net=args.base_net,
            hyperparameters=hyperparameters,
        )
