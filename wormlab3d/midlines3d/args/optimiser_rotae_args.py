from argparse import ArgumentParser

from wormlab3d.nn.args import OptimiserArgs


class RotAEOptimiserArgs(OptimiserArgs):
    def __init__(
            self,
            renderer_blur_sigma: float = 0.1,
            max_rotation: float = 0.,
            w_masks: float = 1.,
            w_masks_coords: float = 0.,
            w_masks_sum: float = 0.,
            w_c2d: float = 1.,
            w_c3d: float = 1.,
            w_d0: float = 1.,
            w_d2d: float = 1.,
            w_d3d: float = 1.,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert renderer_blur_sigma > 0, '"--renderer-blur-sigma" must be > 0'
        self.renderer_blur_sigma = renderer_blur_sigma
        self.max_rotation = max_rotation
        self.w_masks = w_masks
        self.w_masks_coords = w_masks_coords
        self.w_masks_sum = w_masks_sum
        self.w_c2d = w_c2d
        self.w_c3d = w_c3d
        self.w_d2d = w_d2d
        self.w_d0 = w_d0
        self.w_d3d = w_d3d

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = OptimiserArgs.add_args(parser)
        group.add_argument('--max-rotation', type=float, default=0.,
                           help='Upper bound for the uniform distributions the rotation angles are drawn from (in degrees). '
                                'Set to 0 to disable rotations.')
        group.add_argument('--renderer-blur-sigma', type=float, default=0.1,
                           help='Fatten the rendered midline masks with a gaussian blur using this sigma value (in pixels). Must be > 0.')
        group.add_argument('--w-masks', type=float, default=1., help='Weighting to apply to the masks loss.')
        group.add_argument('--w-masks-coords', type=float, default=1.,
                           help='Weighting to apply to the masks_coords loss.')
        group.add_argument('--w-masks-sum', type=float, default=1.,
                           help='Weighting to apply to the masks_sum loss.')
        group.add_argument('--w-c2d', type=float, default=1., help='Weighting to apply to the c2d loss.')
        group.add_argument('--w-c3d', type=float, default=1., help='Weighting to apply to the c3d loss.')
        group.add_argument('--w-d0', type=float, default=1., help='Weighting to apply to the d0 loss.')
        group.add_argument('--w-d2d', type=float, default=1., help='Weighting to apply to the d2d loss.')
        group.add_argument('--w-d3d', type=float, default=1., help='Weighting to apply to the d3d loss.')
