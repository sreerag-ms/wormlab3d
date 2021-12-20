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
            n_steps_cc_init: int = 500,
            n_steps_curve_init: int = 500,
            n_steps_cc: int = 100,
            n_steps_curve: int = 100,
            cloud_points_perturbation: float = 1e-3,
            cloud_sigmas_perturbation: float = 1e-4,
            curve_points_perturbation: float = 0.,
            curve_sigmas_perturbation: float = 0.,

            loss_target_cc: float = None,
            loss_target_curve: float = None,

            optimise_cam_coeffs: bool = True,
            optimise_cam_intrinsics: bool = True,
            optimise_cam_rotations: bool = True,
            optimise_cam_translations: bool = True,
            optimise_cam_distortions: bool = True,
            optimise_cam_shifts: bool = True,

            optimise_cloud: bool = True,
            optimise_curve: bool = True,
            optimise_cloud_sigmas: bool = True,
            optimise_curve_sigmas: bool = True,
            optimise_curve_intensities: bool = True,
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
            loss_curve_curvature: float = 0,
            loss_curve_length: float = 0,
            loss_cloud_sigmas: float = 0,
            loss_cloud_scores: float = 0,
            loss_cloud_neighbours: float = 0,
            loss_cloud_neighbours_rate: float = 0,

            loss_curve_masks: float = 0,
            loss_curve_dists_neighbours: float = 0,
            loss_curve_dists_parents: float = 0,
            loss_curve_dists_aunts: float = 0,
            loss_curve_scores: float = 0,
            loss_curve_sigmas: float = 0,
            loss_curve_intensities: float = 0,
            loss_curve_smoothness: float = 0,

            algorithm_cc: str = OPTIMISER_ADAM,
            algorithm_curve: str = OPTIMISER_ADAM,

            relocate_every_n_steps: int = -1,
            relocate_score_threshold: float = 1e-4,
            relocate_max_points: int = None,
            relocate_points_randomly: int = 0,

            cluster_every_n_steps: int = -1,
            cluster_score_threshold: float = 1e-4,
            cluster_density_threshold: float = None,
            cluster_max_points: int = None,
            cluster_nhd_size: int = None,

            lr_cam_coeffs: float = 1e-5,
            lr_cloud_points: float = 1e-2,
            lr_curve_points: float = 1e-3,
            lr_curve_length: float = 1e-3,
            lr_cloud_sigmas: float = 1e-3,
            lr_curve_sigmas: float = 1e-3,
            lr_curve_intensities: float = 1e-3,
            **kwargs
    ):
        assert window_size % 2 == 1, 'Window size must be an odd number.'
        self.window_size = window_size
        self.n_steps_cc_init = n_steps_cc_init
        self.n_steps_curve_init = n_steps_curve_init
        self.n_steps_cc = n_steps_cc
        self.n_steps_curve = n_steps_curve
        self.cloud_points_perturbation = cloud_points_perturbation
        self.cloud_sigmas_perturbation = cloud_sigmas_perturbation
        self.curve_points_perturbation = curve_points_perturbation
        self.curve_sigmas_perturbation = curve_sigmas_perturbation

        self.loss_target_cc = loss_target_cc
        self.loss_target_curve = loss_target_curve

        self.optimise_cam_coeffs = optimise_cam_coeffs
        self.optimise_cam_intrinsics = optimise_cam_intrinsics
        self.optimise_cam_rotations = optimise_cam_rotations
        self.optimise_cam_translations = optimise_cam_translations
        self.optimise_cam_distortions = optimise_cam_distortions
        self.optimise_cam_shifts = optimise_cam_shifts

        self.optimise_cloud = optimise_cloud
        self.optimise_curve = optimise_curve
        self.optimise_cloud_sigmas = optimise_cloud_sigmas
        self.optimise_curve_sigmas = optimise_curve_sigmas
        self.optimise_curve_intensities = optimise_curve_intensities
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
        self.loss_curve_curvature = loss_curve_curvature
        self.loss_curve_length = loss_curve_length
        self.loss_cloud_sigmas = loss_cloud_sigmas
        self.loss_cloud_scores = loss_cloud_scores
        self.loss_cloud_neighbours = loss_cloud_neighbours
        self.loss_cloud_neighbours_rate = loss_cloud_neighbours_rate

        # MSC losses
        self.loss_curve_masks = loss_curve_masks
        self.loss_curve_dists_neighbours = loss_curve_dists_neighbours
        self.loss_curve_dists_parents = loss_curve_dists_parents
        self.loss_curve_dists_aunts = loss_curve_dists_aunts
        self.loss_curve_scores = loss_curve_scores
        self.loss_curve_sigmas = loss_curve_sigmas
        self.loss_curve_intensities = loss_curve_intensities
        self.loss_curve_smoothness = loss_curve_smoothness

        assert algorithm_cc in OPTIMISER_ALGORITHMS
        self.algorithm_cc = algorithm_cc
        self.algorithm_curve = algorithm_curve

        self.relocate_every_n_steps = relocate_every_n_steps
        self.relocate_score_threshold = relocate_score_threshold
        self.relocate_max_points = relocate_max_points
        self.relocate_points_randomly = relocate_points_randomly

        self.cluster_every_n_steps = cluster_every_n_steps
        self.cluster_score_threshold = cluster_score_threshold
        self.cluster_density_threshold = cluster_density_threshold
        self.cluster_max_points = cluster_max_points
        self.cluster_nhd_size = cluster_nhd_size

        self.lr_cam_coeffs = lr_cam_coeffs
        self.lr_cloud_points = lr_cloud_points
        self.lr_curve_points = lr_curve_points
        self.lr_curve_length = lr_curve_length
        self.lr_cloud_sigmas = lr_cloud_sigmas
        self.lr_curve_sigmas = lr_curve_sigmas
        self.lr_curve_intensities = lr_curve_intensities

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')

        group.add_argument('--window-size', type=int, default=9,
                           help='Sliding window size.')
        group.add_argument('--n-steps-cc-init', type=int, default=500,
                           help='Number of steps to train the camera coefficients and cloud points on the first batch.')
        group.add_argument('--n-steps-curve-init', type=int, default=500,
                           help='Number of steps to train the curve parameters for on the first batch.')
        group.add_argument('--n-steps-cc', type=int, default=100,
                           help='Number of steps to train the camera coefficients and cloud points.')
        group.add_argument('--n-steps-curve', type=int, default=100,
                           help='Number of steps to train the curve parameters for.')
        group.add_argument('--cloud-points-perturbation', type=float, default=1e-3,
                           help='Noise level (std) to add to the previous frame\'s cloud points.')
        group.add_argument('--cloud-sigmas-perturbation', type=float, default=0,
                           help='Noise level (std) to add to the previous frame\'s cloud sigmas.')
        group.add_argument('--curve-points-perturbation', type=float, default=1e-3,
                           help='Noise level (std) to add to the previous frame\'s curve points.')
        group.add_argument('--curve-sigmas-perturbation', type=float, default=0,
                           help='Noise level (std) to add to the previous frame\'s curve sigmas.')

        group.add_argument('--loss-target-cc', type=float,
                           help='Target overall loss level for the camera coefficients and cloud points metrics.')
        group.add_argument('--loss-target-curve', type=float,
                           help='Target overall loss level for the camera coefficients and cloud points metrics.')

        group.add_argument('--optimise-cam-coeffs', type=str2bool, default=True,
                           help='Optimise the camera coefficients. Master flag, overrides individual settings. Default = True.')
        group.add_argument('--optimise-cam-intrinsics', type=str2bool, default=True,
                           help='Optimise the intrinsic camera coefficients (fx, fy, cx, cy). Default = True.')
        group.add_argument('--optimise-cam-rotations', type=str2bool, default=True,
                           help='Optimise the camera rotation matrix (R). Default = True.')
        group.add_argument('--optimise-cam-translations', type=str2bool, default=True,
                           help='Optimise the camera translation vector (t). Default = True.')
        group.add_argument('--optimise-cam-distortions', type=str2bool, default=True,
                           help='Optimise the camera distortion coefficients (d). Default = True.')
        group.add_argument('--optimise-cam-shifts', type=str2bool, default=True,
                           help='Optimise the camera shifts (s). Default = True.')

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
        group.add_argument('--optimise-curve-intensities', type=str2bool, default=True,
                           help='Optimise the curve intensities. Default = True.')

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
        group.add_argument('--loss-curve-curvature', type=float, default=0,
                           help='Regularisation weight for penalising high curvature.')
        group.add_argument('--loss-curve-length', type=float, default=0,
                           help='Regularisation weight for penalising deviations from the database value of the worm length.')
        group.add_argument('--loss-cloud-sigmas', type=float, default=0,
                           help='Regularisation weight for penalising cloud sigma variance.')
        group.add_argument('--loss-cloud-scores', type=float, default=0,
                           help='Regularisation weight for rewarding large cloud scores.')
        group.add_argument('--loss-cloud-neighbours', type=float, default=0,
                           help='Regularisation weight for penalising distances between neighbouring cloud points.')
        group.add_argument('--loss-cloud-neighbours-rate', type=float, default=0.1,
                           help='Exponential rating coefficient for neighbourhood regularisation. Higher=more global. Lower=more local.')

        group.add_argument('--loss-curve-masks', type=float, default=1,
                           help='Weighting for masks losses.')
        group.add_argument('--loss-curve-dists-neighbours', type=float, default=1,
                           help='Weighting for regularising the distances between neighbours.')
        group.add_argument('--loss-curve-dists-parents', type=float, default=1,
                           help='Weighting for regularising the distances between children and parent.')
        group.add_argument('--loss-curve-dists-aunts', type=float, default=0,
                           help='Weighting for regularising the distances between children and aunts.')
        group.add_argument('--loss-curve-scores', type=float, default=0,
                           help='Regularisation weight for rewarding large curve scores.')
        group.add_argument('--loss-curve-sigmas', type=float, default=0,
                           help='Regularisation weight for penalising cloud sigma variance.')
        group.add_argument('--loss-curve-intensities', type=float, default=0,
                           help='Regularisation weight for penalising rendering intensity variance.')
        group.add_argument('--loss-curve-smoothness', type=float, default=0,
                           help='Regularisation weight for penalising curve non-smoothness.')

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
        group.add_argument('--relocate-points-randomly', type=int, default=0,
                           help='Relocate this number of points at random at a time. Default=0.')

        group.add_argument('--cluster-every-n-steps', type=int, default=1,
                           help='Relocate cloud points with low neighbourhood densities near to points with high densities every n steps. Default=1.')
        group.add_argument('--cluster-score-threshold', type=float, default=1e-4,
                           help='Threshold below which points may be clustered. Default=1e-4.')
        group.add_argument('--cluster-density-threshold', type=float,
                           help='Density threshold below which points may be clustered. Default=None (always cluster least dense points).')
        group.add_argument('--cluster-max-points', type=int, default=None,
                           help='Maximum number of points to cluster at a time. Default=n_cloud_points*0.01.')
        group.add_argument('--cluster-nhd-size', type=int, default=None,
                           help='Neighbourhood size used to determine density. Default=n_cloud_points*0.01.')

        group.add_argument('--lr-cam-coeffs', type=float, default=1e-5,
                           help='Learning rate for the camera coefficients.')
        group.add_argument('--lr-cloud-points', type=float, default=1e-2,
                           help='Learning rate for the cloud points.')
        group.add_argument('--lr-curve-points', type=float, default=1e-3,
                           help='Learning rate for the curve points.')
        group.add_argument('--lr-curve-length', type=float, default=1e-3,
                           help='Learning rate for the curve length.')
        group.add_argument('--lr-cloud-sigmas', type=float, default=1e-3,
                           help='Learning rate for the cloud sigmas.')
        group.add_argument('--lr-curve-sigmas', type=float, default=1e-3,
                           help='Learning rate for the curve sigmas.')
        group.add_argument('--lr-curve-intensities', type=float, default=1e-3,
                           help='Learning rate for the curve intensities.')
        return group
