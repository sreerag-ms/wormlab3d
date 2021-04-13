from argparse import ArgumentParser, Namespace

from wormlab3d.data.model.dataset import DATASET_TYPE_2D_MIDLINE
from wormlab3d.nn.args import DatasetArgs


class DatasetMidline2DArgs(DatasetArgs):
    def __init__(
            self,
            blur_sigma: float = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.blur_sigma = blur_sigma
        self.dataset_type = DATASET_TYPE_2D_MIDLINE

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        DatasetArgs.add_args(parser)
        parser.add_argument('--blur-sigma', type=float, default=0,
                            help='Fatten the midline mask with a gaussian blur using this sigma value (in pixels).')

    @staticmethod
    def from_args(args: Namespace) -> 'DatasetArgs':
        """
        Create a DatasetArgs instance from command-line arguments.
        """
        return DatasetMidline2DArgs(
            ds_id=args.dataset_id,
            load=args.load_dataset,
            train_test_split=args.train_test_split,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
            augment=args.augment,
            n_dataloader_workers=args.n_dataloader_workers,
            preload_from_database=args.preload_from_database,
            blur_sigma=args.blur_sigma,
        )
