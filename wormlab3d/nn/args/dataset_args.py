from argparse import ArgumentParser, _ArgumentGroup
from typing import List

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class DatasetArgs(BaseArgs):
    def __init__(
            self,
            dataset_id: str = None,
            load_dataset: bool = True,
            train_test_split: float = 0.8,
            restrict_users: List[str] = None,
            restrict_strains: List[str] = None,
            restrict_sexes: List[str] = None,
            restrict_ages: List[str] = None,
            restrict_tags: List[int] = None,
            restrict_concs: List[float] = None,
            centre_3d_max_error: float = None,
            exclude_experiments: List[int] = None,
            include_experiments: List[int] = None,
            exclude_trials: List[int] = None,
            include_trials: List[int] = None,
            min_trial_quality: int = 9,
            reconstructions: List[str] = None,
            augment: bool = False,
            n_dataloader_workers: int = 4,
            preload_from_database: bool = False,
            **kwargs
    ):
        self.dataset_type = None
        self.dataset_id = dataset_id
        if dataset_id is not None:
            assert load_dataset, 'Dataset id defined, this is incompatible with load=False.'
        self.load = load_dataset
        self.train_test_split = train_test_split
        if restrict_users is None:
            restrict_users = []
        self.restrict_users = restrict_users
        if restrict_strains is None:
            restrict_strains = []
        self.restrict_strains = restrict_strains
        if restrict_sexes is None:
            restrict_sexes = []
        self.restrict_sexes = restrict_sexes
        if restrict_ages is None:
            restrict_ages = []
        self.restrict_ages = restrict_ages
        if restrict_tags is None:
            restrict_tags = []
        self.restrict_tags = restrict_tags
        if restrict_concs is None:
            restrict_concs = []
        self.restrict_concs = restrict_concs
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
        self.min_trial_quality = min_trial_quality
        if reconstructions is not None:
            assert len(self.include_experiments) == 0, 'reconstructions cannot be defined with include_experiments!'
            assert len(self.exclude_experiments) == 0, 'reconstructions cannot be defined with exclude_experiments!'
            assert len(self.include_trials) == 0, 'reconstructions cannot be defined with include_trials!'
            assert len(self.exclude_trials) == 0, 'reconstructions cannot be defined with exclude_trials!'
        else:
            reconstructions = []
        self.reconstructions = reconstructions
        self.augment = augment
        self.n_dataloader_workers = n_dataloader_workers
        self.preload_from_database = preload_from_database

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Dataset Args')
        group.add_argument('--dataset-id', type=str,
                           help='Load a dataset by its database id.')
        group.add_argument('--load-dataset', type=str2bool, default=True,
                           help='Try to load an existing dataset if available matching the given parameters.')
        group.add_argument('--train-test-split', type=float, default=0.8,
                           help='Train/test split.')
        group.add_argument('--restrict-users', type=lambda s: [str(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of users.')
        group.add_argument('--restrict-strains', type=lambda s: [str(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of strains.')
        group.add_argument('--restrict-sexes', type=lambda s: [str(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of sexes.')
        group.add_argument('--restrict-ages', type=lambda s: [str(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of ages.')
        group.add_argument('--restrict-tags', type=lambda s: [int(item) for item in s.split(',')],
                           help='Restrict the dataset to only include items matching (any of) the given (comma delimited) list of tag ids.')
        group.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                           help='Restrict the dataset to only include data from experiments at given (comma delimited) concentrations.')
        group.add_argument('--centre-3d-max-error', type=float,
                           help='Maximum allowed reprojection error for the frame\'s 3d centre points.')
        group.add_argument('--exclude-experiments', type=lambda s: [int(item) for item in s.split(',')],
                           help='Exclude data from these experiments.')
        group.add_argument('--include-experiments', type=lambda s: [int(item) for item in s.split(',')],
                           help='Only include data from these experiments.')
        group.add_argument('--exclude-trials', type=lambda s: [int(item) for item in s.split(',')],
                           help='Exclude data from these trials.')
        group.add_argument('--include-trials', type=lambda s: [int(item) for item in s.split(',')],
                           help='Only include data from these trials.')
        group.add_argument('--min-trial-quality', type=int, default=9,
                           help='Minimum trial quality.')
        group.add_argument('--reconstructions', type=lambda s: [str(item) for item in s.split(',')],
                           help='Only include data from these reconstructions.')
        group.add_argument('--augment', type=str2bool, default=True,
                           help='Apply data augmentation.')
        group.add_argument('--n-dataloader-workers', type=int, default=4,
                           help='Number of dataloader worker processes.')
        group.add_argument('--preload-from-database', type=str2bool, default=False,
                           help='Preload all data from the database before starting, as opposed to loading on demand.')
        return group

    def get_info(self) -> List[str]:
        return [
            f'Train/test split={self.train_test_split}.',
            f'Restrict users={self.restrict_users}.',
            f'Restrict strains={self.restrict_strains}.',
            f'Restrict sexes={self.restrict_sexes}.',
            f'Restrict ages={self.restrict_ages}.',
            f'Restrict tags={self.restrict_tags}.',
            f'Restrict concs={self.restrict_concs}.',
            f'Max centre 3d reprojection error={int(self.centre_3d_max_error)}.',
            f'Include experiments={self.include_experiments}.',
            f'Exclude experiments={self.exclude_experiments}.',
            f'Include trials={self.include_trials}.',
            f'Exclude trials={self.exclude_trials}.',
            f'Minimum trial quality={self.min_trial_quality}.',
            f'Reconstructions={self.reconstructions}.',
        ]
