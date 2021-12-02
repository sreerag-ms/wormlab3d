from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs


class SourceArgs(BaseArgs):
    def __init__(
            self,
            trial_id: str = None,
            masks_id: str = None,
            start_frame: int = 0,
            end_frame: int = -1,
            masks_target_ceil_threshold: float = 0.4,
            **kwargs
    ):
        self.trial_id = trial_id
        self.masks_id = masks_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        assert (trial_id is None and masks_id is not None) \
               or (trial_id is not None and masks_id is None), \
            'Must specify one of trial or masks, not both.'

        self.masks_target_ceil_threshold = masks_target_ceil_threshold

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Source Args')
        group.add_argument('--trial-id', type=str,
                           help='Database id for a Trial instance to use as the source.')
        group.add_argument('--masks-id', type=str,
                           help='Database id for a SegmentationMasks instance to use as the source.')
        parser.add_argument('--start-frame', type=int, default=0, help='Frame number to start from.')
        parser.add_argument('--end-frame', type=int, default=-1, help='Frame number to end at.')
        group.add_argument('--masks-target-ceil-threshold', type=float, default=0.4,
                           help='Ceiling threshold above which target mask is set to 1. Default=0.4.')
