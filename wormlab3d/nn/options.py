def add_training_params(parser):
    # -------------------- Training params -------------------- #
    resume_parser = parser.add_mutually_exclusive_group(required=False)
    resume_parser.add_argument('--resume', action='store_true',
                               help='Resume from a previous checkpoint.')
    resume_parser.add_argument('--no-resume', action='store_false', dest='resume',
                               help='Do not resume from a previous checkpoint. Any existing data will be wiped')
    resume_parser.set_defaults(resume=False)
    parser.add_argument('--resume-from', type=str,
                        help='Resume from a specific timestamp.')
    # parser.set_defaults(resume=True)
    parser.add_argument('--gpu-only', action='store_true',
                        help='Abort if no gpus are detected.')
    parser.add_argument('--data-type', type=str,
                        choices=['bishop', 'xyz', 'cpca', 'xyz_inv'], default='xyz',
                        help='Which data type to use.')
    parser.add_argument('--rebuild-dataset', action='store_true',
                        help='Rebuild the dataset even if it already exists.')
    parser.add_argument('--restrict-classes', type=lambda s: [int(item) for item in s.split(',')],
                        help='Comma delimited list of class numbers to include.')
    parser.add_argument('--include-mirrors', action='store_true',
                        help='Include mirrored versions of the data (with unique class labels)')
    parser.add_argument('--n-cpca-components', type=int, default=2,
                        help='Number of CPCA components to use.')
    parser.add_argument('--n-frames', type=int, default=50,
                        help='Number of frames to use for sliding window.')
    parser.add_argument('--frame-shift', type=int, default=5,
                        help='Number of frames to shift between windows.')
    parser.add_argument('--augment', action='store_true', dest='augment',
                        help='Apply data augmentation.')
    parser.add_argument('--train-test-split', type=float, default=0.8,
                        help='Train/test split.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size to use for training and testing')
    parser.add_argument('--n-epochs', type=int, default=300,
                        help='Number of epochs to run for.')
    parser.add_argument('--lr-init', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--lr-update', type=float, default=None,
                        help='Use this learning rate (used to change learning rate of loaded checkpoints).')
    parser.add_argument('--sgd-momentum', type=float, default=0.9,
                        help='SGD momentum.')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay.')
    parser.add_argument('--n-dataloader-workers', type=int, default=4,
                        help='Number of dataloader worker processes.')


def _add_densenet_params(parser):
    # -------------------- DenseNet params -------------------- #
    parser.add_argument('--growth-rate', type=int, default=8,
                        help='Growth rate for each layer in the blocks (k).')
    parser.add_argument('--n-init-channels', type=int, default=16,
                        help='Number of channels in the first input layer.')
    parser.add_argument('--compression-factor', type=float, default=0.5,
                        help='Factor to reduce resolution by in transition layers (theta).')
    parser.add_argument('--blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                        default='6,6,6',
                        help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
    parser.add_argument('--dropout-prob', type=float, default=0.2,
                        help='Dropout probability.')


def _add_resnet_params(parser):
    # -------------------- ResNet params -------------------- #
    parser.add_argument('--n-init-channels', type=int, default=64,
                        help='Number of channels in the first input layer.')
    parser.add_argument('--blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                        default='3,4,6,3',
                        help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
    parser.add_argument('--shortcut-type', type=str, choices=['id', 'conv'], default='id',
                        help='Shortcut operation to use when dimensions change.')
    bottleneck_parser = parser.add_mutually_exclusive_group(required=False)
    bottleneck_parser.add_argument('--use-bottlenecks', action='store_true',
                                   help='Use bottleneck type residual layers.')
    bottleneck_parser.add_argument('--no-bottlenecks', dest='use_bottlenecks', action='store_false',
                                   help='Don\'t use bottleneck type residual layers.')
    bottleneck_parser.set_defaults(use_bottlenecks=False)
    parser.add_argument('--dropout-prob', type=float, default=0.2,
                        help='Dropout probability.')


def _add_pyramidnet_params(parser):
    # -------------------- PyramidNet params -------------------- #
    parser.add_argument('--n-init-channels', type=int, default=16,
                        help='Number of channels in the first input layer.')
    parser.add_argument('--blocks-config', type=lambda s: [int(item) for item in s.split(',')],
                        default='3,4,6,3',
                        help='Comma delimited list of layers for each block. Number of entries determines number of blocks.')
    parser.add_argument('--alpha', type=int, default=420,
                        help='The widening factor which defines how quickly the pyramid expands at each layer.')
    parser.add_argument('--shortcut-type', type=str, choices=['id', 'conv'], default='id',
                        help='Shortcut operation to use when dimensions change.')
    bottleneck_parser = parser.add_mutually_exclusive_group(required=False)
    bottleneck_parser.add_argument('--use-bottlenecks', action='store_true',
                                   help='Use bottleneck type residual layers.')
    bottleneck_parser.add_argument('--no-bottlenecks', dest='use_bottlenecks', action='store_false',
                                   help='Don\'t use bottleneck type residual layers.')
    bottleneck_parser.set_defaults(use_bottlenecks=False)
    parser.add_argument('--dropout-prob', type=float, default=0.2,
                        help='Dropout probability.')


def _add_fcnet_params(parser):
    # -------------------- FCNet params -------------------- #
    parser.add_argument('--layers-config', type=lambda s: [int(item) for item in s.split(',')],
                        default='100,100',
                        help='Comma delimited list of layer sizes.')
    parser.add_argument('--dropout-prob', type=float, default=0.2,
                        help='Dropout probability.')


def add_model_params(model_type, parser):
    if model_type == 'densenet':
        _add_densenet_params(parser)
    elif model_type == 'resnet':
        _add_resnet_params(parser)
    elif model_type == 'pyramidnet':
        _add_pyramidnet_params(parser)
    elif model_type == 'fcnet':
        _add_fcnet_params(parser)


def print_args(args):
    print('--- Arguments ---')
    for arg_name, arg_value in vars(args).items():
        print('{}: {}'.format(arg_name, arg_value))
    print('-----------------\n')
