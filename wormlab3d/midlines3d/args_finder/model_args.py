from argparse import ArgumentParser

from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_POINTS, ENCODING_MODES, ENCODING_MODE_MSC
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class ModelArgs(BaseArgs):
    def __init__(
            self,
            skeletoniser_id: str = None,
            load: bool = True,
            model_id: str = None,
            n_cloud_points: int = 1000,
            n_curve_points: int = 50,
            curve_mode: str = ENCODING_MODE_POINTS,
            ms_curve_depth: int = 5,
            n_curve_basis_fns: int = 4,
            blur_sigmas_cloud_init: float = 0.01,
            blur_sigmas_curve_init: float = 0.1,
            blur_sigma_vols: float = 1,
            max_revolutions: int = 2,
            **kwargs
    ):
        self.skeletoniser_id = skeletoniser_id
        self.load = load
        self.model_id = model_id
        self.n_cloud_points = n_cloud_points
        self.n_curve_points = n_curve_points
        self.curve_mode = curve_mode
        self.ms_curve_depth= ms_curve_depth
        self.n_curve_basis_fns = n_curve_basis_fns

        if curve_mode == ENCODING_MODE_MSC:
            N = 0
            for mi in range(ms_curve_depth):
                N += 2**mi
            self.n_curve_points = N
            self.n_cloud_points = 0
            self.n_curve_basis_fns = 0

        # Blur sigmas (controls the size of rendered points)
        self.blur_sigma_masks_cloud_init = blur_sigmas_cloud_init
        self.blur_sigma_masks_curve_init = blur_sigmas_curve_init
        self.blur_sigma_vols = blur_sigma_vols

        self.max_revolutions = max_revolutions

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Model Args')
        group.add_argument('--skeletoniser-id', type=str,
                           help='Load a skeletoniser model by its database id.')
        group.add_argument('--load-model', type=str2bool, default=True,
                           help='Try to load an existing network if available matching the given parameters.')
        group.add_argument('--model-id', type=str,
                           help='Load a model by its database id.')
        group.add_argument('--n-cloud-points', type=int, default=500,
                           help='Number of cloud points. Default=500.')
        group.add_argument('--n-curve-points', type=int, default=50,
                           help='Number of curve points. Default=50.')
        group.add_argument('--curve-mode', type=str, default=ENCODING_MODE_POINTS, choices=ENCODING_MODES,
                           help='Curve parametrisation mode.')
        group.add_argument('--ms-curve-depth', type=int, default=5,
                           help='Depth of multi-scale curves to use. Default=5 (=1,2,4,8,16).')
        group.add_argument('--n-curve-basis-fns', type=int, default=4,
                           help='Number of basis functions to use for basis curve encoding. Default=4.')
        group.add_argument('--blur-sigmas-cloud-init', type=float, default=0.01,
                           help='Blur sigmas for rendering cloud points. Default=0.01.')
        group.add_argument('--blur-sigmas-curve-init', type=float, default=0.1,
                           help='Blur sigmas for rendering curve points. Default=0.1.')
        group.add_argument('--blur-sigmas-vols', type=float, default=1,
                           help='Blur sigmas for rendering points in the volume. Default=1.')
        group.add_argument('--max-revolutions', type=float, default=2.,
                           help='Maximum number of full revolutions the worm can do (limits curvature). Default=2.')

    def get_db_params(self) -> dict:
        return {
            'n_cloud_points': self.n_cloud_points,
            'n_curve_points': self.n_curve_points,
            'curve_mode': self.curve_mode,
            'ms_curve_depth': self.ms_curve_depth,
            'n_curve_basis_fns': self.n_curve_basis_fns,
            'blur_sigmas_cloud_init': self.blur_sigma_masks_cloud_init,
            'blur_sigmas_curve_init': self.blur_sigma_masks_curve_init,
            'blur_sigma_vols': self.blur_sigma_vols,
            'max_revolutions': self.max_revolutions,
        }
