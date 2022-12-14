from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.data.model.midline3d import M3D_SOURCE_RECONST, M3D_SOURCES
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class FrameSequenceArgs(BaseArgs):
    def __init__(
            self,
            fs_id: str = None,
            sw_run_id: str = None,
            load_fs: bool = True,
            trial: int = None,
            reconstruction: str = None,
            start_frame: int = None,
            midline_source: str = M3D_SOURCE_RECONST,
            midline_source_file: str = None,
            **kwargs
    ):
        self.fs_id = fs_id
        if fs_id is not None:
            assert load_fs, 'Frame sequence id defined, this is incompatible with load=False.'
        self.sw_run_id = sw_run_id
        if sw_run_id is not None:
            assert load_fs, 'Simple worm simulation run id defined, this is incompatible with load=False.'
            assert fs_id is None, 'Can only load a FS or SW run as the target, not both!'
        self.load = load_fs
        self.trial_id = trial
        self.reconstruction_id = reconstruction
        if start_frame is None:
            start_frame = 0
        self.start_frame = start_frame
        self.midline_source = midline_source
        self.midline_source_file = midline_source_file

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Frame Sequence Args')
        group.add_argument('--fs-id', type=str,
                           help='Load a FS by its database id.')
        group.add_argument('--sw-run-id', type=str,
                           help='Load a simple worm simulation run by its database id.')
        group.add_argument('--load-fs', type=str2bool, default=True,
                           help='Try to load an existing FS if available matching the given parameters.')
        group.add_argument('--trial', type=int,
                           help='Trial database id.')
        group.add_argument('--reconstruction', type=str,
                           help='Reconstruction database id.')
        group.add_argument('--start-frame', type=int, default=0,
                           help='Starting frame number of the sequence.')
        group.add_argument('--midline-source', type=str, default=M3D_SOURCE_RECONST, choices=M3D_SOURCES,
                           help='Source to use for the reconstructed midline.')
        group.add_argument('--midline-source-file', type=str,
                           help='Source file to use for the reconstructed midline.')
        return group
