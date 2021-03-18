def add_dataset_params(parser):
    # -------------------- Dataset params -------------------- #
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
