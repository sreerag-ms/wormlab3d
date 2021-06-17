from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.data.model.midline3d import M3D_SOURCE_RECONST, M3D_SOURCES
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class FrameSequenceArgs(BaseArgs):
    def __init__(
            self,
            fs_id: str = None,
            load_fs: bool = True,
            trial: int = None,
            start_frame: str = None,
            midline_source: str = M3D_SOURCE_RECONST,
            **kwargs
    ):
        self.fs_id = fs_id
        if fs_id is not None:
            assert load_fs, 'Frame sequence id defined, this is incompatible with load=False.'
        else:
            assert all(v is not None for v in [trial, start_frame]), 'Frame sequence not well defined.'
        self.load = load_fs
        self.trial_id = trial
        self.start_frame = start_frame
        self.midline_source = midline_source

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Frame Sequence Args')
        group.add_argument('--fs-id', type=str,
                           help='Load a FS by its database id.')
        group.add_argument('--load-fs', type=str2bool, default=True,
                           help='Try to load an existing FS if available matching the given parameters.')
        group.add_argument('--trial', type=int,
                           help='Trial database id.')
        group.add_argument('--start-frame', type=int, default=0,
                           help='Starting frame number of the sequence.')
        group.add_argument('--midline-source', type=str, default=M3D_SOURCE_RECONST, choices=M3D_SOURCES,
                           help='Source to use for the reconstructed midline.')
        return group
