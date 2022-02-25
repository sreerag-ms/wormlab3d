from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.data.model.mf_parameters import MFParameters, RENDER_MODE_GAUSSIANS, RENDER_MODES
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.nn.args.optimiser_args import OPTIMISER_ALGORITHMS, LOSS_MSE, OPTIMISER_ADAM, LOSSES
from wormlab3d.toolkit.util import str2bool


class ParameterArgs(BaseArgs):
    def __init__(
            self,
            load: bool = True,
            params_id: str = None,

            depth: int = 5,
            window_size: int = 1,
            use_master: bool = True,
            sigmas_init: float = 0.1,
            masks_threshold: float = 0.4,
            render_mode: str = RENDER_MODE_GAUSSIANS,

            n_steps_init: int = 5000,
            n_steps_max: int = 500,
            convergence_tau_fast: int = 10,
            convergence_tau_slow: int = 100,
            convergence_threshold: float = 0.1,
            convergence_patience: int = 25,

            optimise_cam_coeffs: bool = True,
            optimise_cam_intrinsics: bool = True,
            optimise_cam_rotations: bool = True,
            optimise_cam_translations: bool = True,
            optimise_cam_distortions: bool = True,
            optimise_cam_shifts: bool = True,

            optimise_sigmas: bool = True,
            optimise_exponents: bool = True,
            optimise_intensities: bool = True,

            loss_masks_metric: str = LOSS_MSE,
            loss_masks_multiscale: bool = True,
            loss_masks: float = 1.,
            loss_neighbours: float = 1.,
            loss_parents: float = 0.,
            loss_aunts: float = 1.,
            loss_scores: float = 0.,
            loss_sigmas: float = 1.,
            loss_exponents: float = 1.,
            loss_intensities: float = 1.,
            loss_smoothness: float = 1.,
            loss_curvature: float = 1.,
            loss_temporal: float = 0.,

            algorithm: str = OPTIMISER_ADAM,

            lr_cam_coeffs: float = 1e-5,
            lr_points: float = 1e-3,
            lr_sigmas: float = 1e-3,
            lr_exponents: float = 1e-3,
            lr_intensities: float = 1e-3,
            **kwargs
    ):
        self.load = load
        self.params_id = params_id
        self.depth = depth
        assert window_size % 2 == 1, 'Window size must be an odd number.'
        self.window_size = window_size
        if window_size == 1:
            assert use_master, 'Must set use-master=True when using a window size of 1.'
        self.use_master = use_master
        self.sigmas_init = sigmas_init
        self.masks_threshold = masks_threshold
        self.render_mode = render_mode
        self.n_steps_init = n_steps_init
        self.n_steps_max = n_steps_max
        self.convergence_tau_fast = convergence_tau_fast
        self.convergence_tau_slow = convergence_tau_slow
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience

        # Calculate total number of curve points
        N = 0
        for mi in range(depth):
            N += 2**mi
        self.n_points_total = N

        self.optimise_cam_coeffs = optimise_cam_coeffs
        self.optimise_cam_intrinsics = optimise_cam_intrinsics
        self.optimise_cam_rotations = optimise_cam_rotations
        self.optimise_cam_translations = optimise_cam_translations
        self.optimise_cam_distortions = optimise_cam_distortions
        self.optimise_cam_shifts = optimise_cam_shifts

        self.optimise_sigmas = optimise_sigmas
        self.optimise_exponents = optimise_exponents
        self.optimise_intensities = optimise_intensities

        self.loss_masks_metric = loss_masks_metric
        self.loss_masks_multiscale = loss_masks_multiscale
        self.loss_masks = loss_masks
        self.loss_neighbours = loss_neighbours
        self.loss_parents = loss_parents
        self.loss_aunts = loss_aunts
        self.loss_scores = loss_scores
        self.loss_sigmas = loss_sigmas
        self.loss_exponents = loss_exponents
        self.loss_intensities = loss_intensities
        self.loss_smoothness = loss_smoothness
        self.loss_curvature = loss_curvature
        self.loss_temporal = loss_temporal

        assert algorithm in OPTIMISER_ALGORITHMS
        self.algorithm = algorithm

        self.lr_cam_coeffs = lr_cam_coeffs
        self.lr_points = lr_points
        self.lr_sigmas = lr_sigmas
        self.lr_exponents = lr_exponents
        self.lr_intensities = lr_intensities

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Parameters')

        group.add_argument('--load', type=str2bool, default=True,
                           help='Try to load an existing parameters database object if available matching the given parameters.')
        group.add_argument('--params-id', type=str,
                           help='Load existing parameters by its database id.')

        group.add_argument('--depth', type=int, default=5,
                           help='Depth of multi-scale curves to use. Default=5 (=1,2,4,8,16).')
        group.add_argument('--window-size', type=int, default=9,
                           help='Sliding window size.')
        group.add_argument('--use-master', type=str2bool, default=True,
                           help='Optimise a single parameter set for the full window. Default = True.')
        group.add_argument('--sigmas-init', type=float, default=0.01,
                           help='Blur sigmas for rendering points. Default=0.01.')
        group.add_argument('--masks-threshold', type=float, default=0.4,
                           help='Threshold value to use for binarising the frame images. Default=0.4.')
        group.add_argument('--render-mode', type=str, default=RENDER_MODE_GAUSSIANS, choices=RENDER_MODES,
                           help='How to render the points, either as gaussian blobs (gaussians) or as circles (circles). Default=gaussians.')

        group.add_argument('--n-steps-init', type=int, default=5000,
                           help='Fixed number of steps to train on the first batch/frame.')
        group.add_argument('--n-steps-max', type=int, default=500,
                           help='Maximum number of steps to train on each frame.')
        group.add_argument('--convergence-tau-fast', type=int, default=10,
                           help='Fast moving estimate of loss to use for convergence detection.')
        group.add_argument('--convergence-tau-slow', type=int, default=100,
                           help='Slow moving estimate of loss to use for convergence detection.')
        group.add_argument('--convergence-threshold', type=float, default=0.1,
                           help='Relative ratio of convergence estimates to qualify as potentially converged.')
        group.add_argument('--convergence-patience', type=int, default=25,
                           help='How many steps to wait after convergence is detected.')

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

        group.add_argument('--optimise-sigmas', type=str2bool, default=True,
                           help='Optimise the rendering sigmas. Default = True.')
        group.add_argument('--optimise-exponents', type=str2bool, default=True,
                           help='Optimise the rendering exponents. Default = True.')
        group.add_argument('--optimise-intensities', type=str2bool, default=True,
                           help='Optimise the rendering intensities. Default = True.')

        group.add_argument('--loss-masks-metric', type=str, choices=LOSSES, default=LOSS_MSE,
                           help='The rendering loss to minimise. Default=mse.')
        group.add_argument('--loss-masks-multiscale', type=str2bool, default=True,
                           help='Whether to sum losses from a cascade of resolutions. Default=True.')
        group.add_argument('--loss-masks', type=float, default=1.,
                           help='Weighting for masks losses.')
        group.add_argument('--loss-neighbours', type=float, default=1.,
                           help='Weighting for regularising the distances between neighbours.')
        group.add_argument('--loss-parents', type=float, default=0.,
                           help='Weighting for regularising the distances between children and parent.')
        group.add_argument('--loss-aunts', type=float, default=1.,
                           help='Weighting for regularising the distances between children and aunts.')
        group.add_argument('--loss-scores', type=float, default=0.,
                           help='Regularisation weight for rewarding large projection scores.')
        group.add_argument('--loss-sigmas', type=float, default=1.,
                           help='Regularisation weight for penalising rendering sigma variance.')
        group.add_argument('--loss-exponents', type=float, default=1.,
                           help='Regularisation weight for penalising rendering exponents variance.')
        group.add_argument('--loss-intensities', type=float, default=1.,
                           help='Regularisation weight for penalising rendering intensity variance.')
        group.add_argument('--loss-smoothness', type=float, default=1.,
                           help='Regularisation weight for penalising curve non-smoothness.')
        group.add_argument('--loss-curvature', type=float, default=1.,
                           help='Regularisation weight for penalising curvature.')
        group.add_argument('--loss-temporal', type=float, default=0.,
                           help='Temporal smoothing weight between frames.')

        group.add_argument('--algorithm', type=str, choices=OPTIMISER_ALGORITHMS, default=OPTIMISER_ADAM,
                           help='Optimisation algorithm.')

        group.add_argument('--lr-cam-coeffs', type=float, default=1e-5,
                           help='Learning rate for the camera coefficients.')
        group.add_argument('--lr-points', type=float, default=1e-3,
                           help='Learning rate for the curve points.')
        group.add_argument('--lr-sigmas', type=float, default=1e-2,
                           help='Learning rate for the rendering sigmas.')
        group.add_argument('--lr-exponents', type=float, default=1e-3,
                           help='Learning rate for the rendering exponents.')
        group.add_argument('--lr-intensities', type=float, default=1e-3,
                           help='Learning rate for the rendering intensities.')
        return group

    def get_db_params(self) -> dict:
        p = {}
        for k in MFParameters._fields.keys():
            if hasattr(self, k):
                p[k] = getattr(self, k)
        return p
