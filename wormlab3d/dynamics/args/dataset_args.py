from argparse import ArgumentParser, _ArgumentGroup
from typing import List

from wormlab3d.data.model.dataset import DATASET_TYPE_EIGENTRACES
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class DynamicsDatasetArgs(BaseArgs):
    def __init__(
            self,
            dataset_id: str = None,
            load_dataset: bool = True,
            train_test_split: float = 0.8,
            n_dataloader_workers: int = 4,

            dataset_m3d: str = None,
            reconstruction: str = None,
            eigenworms: str = None,
            n_components: int = 10,
            smoothing_window: int = 25,

            sample_duration: int = 100,
            X0_duration: int = 10,
            **kwargs
    ):
        self.dataset_type = DATASET_TYPE_EIGENTRACES
        self.dataset_id = dataset_id
        if dataset_id is not None:
            assert load_dataset, 'Dataset id defined, this is incompatible with load=False.'
        self.load = load_dataset
        self.train_test_split = train_test_split
        self.n_dataloader_workers = n_dataloader_workers

        assert not (dataset_m3d is None and reconstruction is None), \
            'Either dataset-m3d or reconstruction must be specified.'
        assert not (dataset_m3d is not None and reconstruction is not None), \
            'Both dataset-m3d and reconstruction cannot be specified.'
        self.dataset_m3d = dataset_m3d
        self.reconstruction = reconstruction
        self.eigenworms = eigenworms
        self.n_components = n_components
        self.smoothing_window = smoothing_window

        self.sample_duration = sample_duration
        self.X0_duration = X0_duration

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
        group.add_argument('--n-dataloader-workers', type=int, default=4,
                           help='Number of dataloader worker processes.')

        group.add_argument('--dataset-m3d', type=str, default=None,
                           help='Midline3D dataset id.')
        group.add_argument('--reconstruction', type=str, default=None,
                           help='Reconstruction id.')
        group.add_argument('--eigenworms', type=str, default=None,
                           help='Eigenworms id.')
        group.add_argument('--n-components', type=int, default=10,
                           help='Number of eigenworm components to use. Default=10.')
        group.add_argument('--smoothing-window', type=int, default=25,
                           help='Number of frames to smooth the data by. Default=25.')

        group.add_argument('--sample-duration', type=int, default=100,
                           help='Number of frames to use per sample. Default=100.')
        group.add_argument('--X0-duration', type=int, default=10,
                           help='Number of initial frames from each sample to condition the dynamics networks. Default=10.')
        return group

    def get_info(self) -> List[str]:
        return [
            f'Train/test split={self.train_test_split}.',
            f'Dataset M3D={self.dataset_m3d}.',
            f'Reconstruction={self.reconstruction}.',
            f'Eigenworms={self.eigenworms}.',
            f'Num components={self.n_components}.',
            f'Smoothing window={self.smoothing_window}.',
            f'Sample duration={self.sample_duration}.',
            f'X0 duration={self.X0_duration}.',
        ]
