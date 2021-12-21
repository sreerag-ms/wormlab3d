from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs


class SourceArgs(BaseArgs):
    def __init__(
            self,
            trial_id: str,
            start_frame: int = 0,
            end_frame: int = -1,
            **kwargs
    ):
        self.trial_id = trial_id
        self.start_frame = start_frame
        self.end_frame = end_frame

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Source Args')
        group.add_argument('--trial-id', type=str,
                           help='Database id for a Trial instance to use as the source.')
        parser.add_argument('--start-frame', type=int, default=0, help='Frame number to start from.')
        parser.add_argument('--end-frame', type=int, default=-1, help='Frame number to end at.')
