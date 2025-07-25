from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs


class SourceArgs(BaseArgs):
    def __init__(
            self,
            trial_id: int,
            start_frame: int = 0,
            end_frame: int = -1,
            head_and_tail_coordinates: str = 'data/head_and_tail_coords_dataset.csv',
            **kwargs
    ):
        self.trial_id = trial_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.head_and_tail_coordinates = head_and_tail_coordinates
        if end_frame == -1 or start_frame < end_frame:
            self.direction = 1
        else:
            self.direction = -1


    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group('Source Args')
        group.add_argument('--trial-id',   type=int,   help='Database id for a Trial instance to use as the source.')
        group.add_argument('--start-frame', type=int, default=0, help='Frame number to start from.')
        group.add_argument('--end-frame',   type=int, default=-1, help='Frame number to end at.')
        group.add_argument('--head-and-tail-coordinates', type=str, default='data/head_and_tail_coords_dataset.csv', help='Path to head and tail coordinates dataset CSV file.')
        

