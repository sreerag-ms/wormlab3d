from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.nn.args.optimiser_args import OPTIMISER_ALGORITHMS, LOSS_MSE, OPTIMISER_ADAM, LOSSES
from wormlab3d.toolkit.util import str2bool

LOSS_CURVE_TARGET_MASKS = 'masks'
LOSS_CURVE_TARGET_CLOUD = 'cloud'
LOSS_CURVE_TARGETS = [
    LOSS_CURVE_TARGET_MASKS,
    LOSS_CURVE_TARGET_CLOUD,
]


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            window_size: int = 9,
            n_steps_cc_init: int = 100,
            n_steps_curve_init: int = 100,
            n_steps_cc: int = 500,
            n_steps_curve: int = 1000,

            optimise_cam_coeffs: bool = True,
            optimise_cloud: bool = True,
            optimise_curve: bool = True,
            optimise_cloud_sigmas: bool = True,
            optimise_curve_sigmas: bool = True,
            optimise_curve_length: bool = True,

            loss_cc: str = LOSS_MSE,
            loss_cc_multiscale: bool = True,
            loss_curve: str = LOSS_MSE,
            loss_curve_multiscale: bool = True,
            loss_curve_3d: bool = False,
            loss_3d_cloud_threshold: float = 1e-4,
            loss_curve_target: str = LOSS_CURVE_TARGET_CLOUD,
            loss_cloud_temporal_smoothing: float = 1e-4,
            loss_curve_temporal_smoothing: float = 1e-4,

            algorithm_cc: str = OPTIMISER_ADAM,
            algorithm_curve: str = OPTIMISER_ADAM,

            relocate_every_n_steps: int = -1,
            relocate_score_threshold: float = 1e-4,
            relocate_max_points: int = None,

            lr_cam_coeffs: float = 1e-5,
            lr_cloud_points: float = 1e-2,
            lr_curve_points: float = 1e-3,
            lr_cloud_sigmas: float = 1e-3,
            lr_curve_sigmas: float = 1e-3,
            **kwargs
    ):
        assert window_size % 2 == 1, 'Window size must be an odd number.'
        self.window_size = window_size
        self.n_steps_cc_init = n_steps_cc_init
        self.n_steps_curve_init = n_steps_curve_init
        self.n_steps_cc = n_steps_cc
        self.n_steps_curve = n_steps_curve

        self.optimise_cam_coeffs = optimise_cam_coeffs
        self.optimise_cloud = optimise_cloud
        self.optimise_curve = optimise_curve
        self.optimise_cloud_sigmas = optimise_cloud_sigmas
        self.optimise_curve_sigmas = optimise_curve_sigmas
        self.optimise_curve_length = optimise_curve_length

        self.loss_cc = loss_cc
        self.loss_cc_multiscale = loss_cc_multiscale
        self.loss_curve = loss_curve
        self.loss_curve_multiscale = loss_curve_multiscale
        self.loss_curve_3d = loss_curve_3d
        self.loss_3d_cloud_threshold = loss_3d_cloud_threshold
        self.loss_curve_target = loss_curve_target
        assert not (self.loss_curve_3d and self.loss_curve_target == LOSS_CURVE_TARGET_MASKS), \
            'Can\'t have 2D masks comparisons in 3D!'
        self.loss_cloud_temporal_smoothing = loss_cloud_temporal_smoothing
        self.loss_curve_temporal_smoothing = loss_curve_temporal_smoothing

        assert algorithm_cc in OPTIMISER_ALGORITHMS
        self.algorithm_cc = algorithm_cc
        self.algorithm_curve = algorithm_curve

        self.relocate_every_n_steps = relocate_every_n_steps
        self.relocate_score_threshold = relocate_score_threshold
        self.relocate_max_points = relocate_max_points

        self.lr_cam_coeffs = lr_cam_coeffs
        self.lr_cloud_points = lr_cloud_points
        self.lr_curve_points = lr_curve_points
        self.lr_cloud_sigmas = lr_cloud_sigmas
        self.lr_curve_sigmas = lr_curve_sigmas

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')

        group.add_argument('--window-size', type=int, default=9,
                           help='Sliding window size.')
        group.add_argument('--n-steps-cc-init', type=int, default=100,
                           help='Number of steps to train the camera coefficients and cloud points on the first batch.')
        group.add_argument('--n-steps-curve-init', type=int, default=100,
                           help='Number of steps to train the curve parameters for on the first batch.')
        group.add_argument('--n-steps-cc', type=int, default=500,
                           help='Number of steps to train the camera coefficients and cloud points.')
        group.add_argument('--n-steps-curve', type=int, default=10000,
                           help='Number of steps to train the curve parameters for.')

        group.add_argument('--optimise-cam-coeffs', type=str2bool, default=True,
                           help='Optimise the camera coefficients. Default = True.')
        group.add_argument('--optimise-cloud', type=str2bool, default=True,
                           help='Optimise the cloud points. Default = True.')
        group.add_argument('--optimise-curve', type=str2bool, default=True,
                           help='Optimise the curve parameters. Default = True.')
        group.add_argument('--optimise-curve-length', type=str2bool, default=True,
                           help='Optimise the curve length. Default = True.')
        group.add_argument('--optimise-cloud-sigmas', type=str2bool, default=True,
                           help='Optimise the cloud sigmas. Default = True.')
        group.add_argument('--optimise-curve-sigmas', type=str2bool, default=True,
                           help='Optimise the curve sigmas. Default = True.')

        group.add_argument('--loss-cc', type=str, choices=LOSSES, default=LOSS_MSE,
                           help='The principal loss to minimise for camera coefficient and cloud points optimisation.')
        group.add_argument('--loss-cc-multiscale', type=str2bool, default=True,
                           help='Whether to sum losses from a cascade of resolutions. Default=True.')
        group.add_argument('--loss-curve', type=str, choices=LOSSES, default=LOSS_MSE,
                           help='The principal loss to minimise for curve parameters optimisation.')
        group.add_argument('--loss-curve-multiscale', type=str2bool, default=True,
                           help='Whether to sum losses from a cascade of resolutions. Default=True.')
        group.add_argument('--loss-curve-3d', type=str2bool, default=False,
                           help='Use a 3D volumetric difference (otherwise stick to 2D masks). (Default=False)')
        group.add_argument('--loss-3d-cloud-threshold', type=float, default=1e-4,
                           help='Exclude any cloud points scoring below this threshold for curve fitting. Default=1e-4.')
        group.add_argument('--loss-curve-target', type=str, default=LOSS_CURVE_TARGET_CLOUD, choices=LOSS_CURVE_TARGETS,
                           help='What to compare the rendered curve against, either "cloud" or "masks". Only "cloud" is valid in 3D.')
        group.add_argument('--loss-cloud-temporal-smoothing', type=float, default=1e-4,
                           help='Temporal smoothing for the cloud points.')
        group.add_argument('--loss-curve-temporal-smoothing', type=float, default=1e-4,
                           help='Temporal smoothing for the curve points.')

        group.add_argument('--algorithm-cc', type=str, choices=OPTIMISER_ALGORITHMS, default=OPTIMISER_ADAM,
                           help='Optimisation algorithm for the camera coefficients, cloud points and cloud sigmas.')
        group.add_argument('--algorithm-curve', type=str, choices=OPTIMISER_ALGORITHMS, default=OPTIMISER_ADAM,
                           help='Optimisation algorithm for the curve points and sigmas.')

        group.add_argument('--relocate-every-n-steps', type=int, default=1,
                           help='Relocate cloud points with low scores near to points with high scores every n steps. Default=1.')
        group.add_argument('--relocate-score-threshold', type=float, default=1e-4,
                           help='Threshold below which points may be relocated. Default=1e-4.')
        group.add_argument('--relocate-max-points', type=int, default=None,
                           help='Maximum number of points to relocate at a time. Default=n_cloud_points*0.01.')

        group.add_argument('--lr-cam-coeffs', type=float, default=1e-5,
                           help='Learning rate for the camera coefficients.')
        group.add_argument('--lr-cloud-points', type=float, default=1e-2,
                           help='Learning rate for the cloud points.')
        group.add_argument('--lr-curve-points', type=float, default=1e-3,
                           help='Learning rate for the curve points.')
        group.add_argument('--lr-cloud-sigmas', type=float, default=1e-3,
                           help='Learning rate for the cloud sigmas.')
        group.add_argument('--lr-curve-sigmas', type=float, default=1e-3,
                           help='Learning rate for the curve sigmas.')
        return group
