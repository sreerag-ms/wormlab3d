from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs


class SourceArgs(BaseArgs):
    def __init__(
            self,
            masks_id: str = None,
            masks_target_ceil_threshold: float = 0.4,
            **kwargs
    ):
        self.masks_id = masks_id
        self.masks_target_ceil_threshold = masks_target_ceil_threshold

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Source Args')
        group.add_argument('--masks-id', type=str,
                           help='Database id for a SegmentationMasks instance to use as the source.')
        group.add_argument('--masks-target-ceil-threshold', type=float, default=0.4,
                           help='Ceiling threshold above which target mask is set to 1. Default=0.4.')
