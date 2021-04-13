from argparse import ArgumentParser, Namespace
from typing import List

from wormlab3d.toolkit.util import str2bool


class DatasetArgs:
    def __init__(
            self,
            ds_id: str = None,
            load: bool = True,
            train_test_split: float = 0.8,
            restrict_tags: List[int] = None,
            restrict_concs: List[float] = None,
            centre_3d_max_error: float = None,
            exclude_trials: List[int] = None,
            include_trials: List[int] = None,
            exclude_experiments: List[int] = None,
            include_experiments: List[int] = None,
            augment: bool = False,
            n_dataloader_workers: int = 4,
            preload_from_database: bool = False
    ):
        self.dataset_type = None
        self.ds_id = ds_id
        if ds_id is not None:
            assert load, 'Dataset id defined, this is incompatible with load=False'
        self.load = load
        self.train_test_split = train_test_split
        if restrict_tags is None:
            restrict_tags = []
        self.restrict_tags = restrict_tags
        if restrict_concs is None:
            restrict_concs = []
        self.restrict_concs = restrict_concs

        if centre_3d_max_error is None:
            centre_3d_max_error = 0
        self.centre_3d_max_error = centre_3d_max_error

        if exclude_experiments is None:
            exclude_experiments = []
        self.exclude_experiments = exclude_experiments

        if include_experiments is None:
            include_experiments = []
        self.include_experiments = include_experiments

        if exclude_trials is None:
            exclude_trials = []
        self.exclude_trials = exclude_trials

        if include_trials is None:
            include_trials = []
        self.include_trials = include_trials

        self.augment = augment
        self.n_dataloader_workers = n_dataloader_workers
        self.preload_from_database = preload_from_database

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        parser.add_argument('--dataset-id', type=str,
                            help='Load a dataset by its database id.')
        parser.add_argument('--load-dataset', type=str2bool, default=True,
                            help='Try to load an existing dataset if available matching the given parameters.')
        parser.add_argument('--train-test-split', type=float, default=0.8,
                            help='Train/test split.')
        parser.add_argument('--restrict-tags', type=lambda s: [int(item) for item in s.split(',')],
                            help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of tag ids.')
        parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                            help='Restrict the dataset to only include data from experiments at given (comma delimited) concentrations.')
        parser.add_argument('--centre-3d-max-error', type=float, default=50,
                            help='Maximum allowed reprojection error for the frame\'s 3d centre points.')
        parser.add_argument('--exclude-experiments', type=lambda s: [int(item) for item in s.split(',')],
                            help='Exclude data from these experiments.')
        parser.add_argument('--include-experiments', type=lambda s: [int(item) for item in s.split(',')],
                            help='Only include data from these experiments.')
        parser.add_argument('--exclude-trials', type=lambda s: [int(item) for item in s.split(',')],
                            help='Exclude data from these trials.')
        parser.add_argument('--include-trials', type=lambda s: [int(item) for item in s.split(',')],
                            help='Only include data from these trials.')
        parser.add_argument('--augment', type=str2bool, default=True,
                            help='Apply data augmentation.')
        parser.add_argument('--n-dataloader-workers', type=int, default=4,
                            help='Number of dataloader worker processes.')
        parser.add_argument('--preload-from-database', type=str2bool, default=False,
                            help='Preload all data from the database before starting, as opposed to loading on demand.')

    @staticmethod
    def from_args(args: Namespace) -> 'DatasetArgs':
        """
        Create a DatasetArgs instance from command-line arguments.
        """
        return DatasetArgs(
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
        )

    def get_info(self) -> List[str]:
        return [
            f'Train/test split={self.train_test_split}.',
            f'Restrict tags={self.restrict_tags}.',
            f'Restrict concs={self.restrict_concs}.',
            f'Max centre 3d reprojection error={self.centre_3d_max_error}.',
            f'Include experiments={self.include_experiments}.',
            f'Exclude experiments={self.exclude_experiments}.',
            f'Include trials={self.include_trials}.',
            f'Exclude trials={self.exclude_trials}.',
        ]
