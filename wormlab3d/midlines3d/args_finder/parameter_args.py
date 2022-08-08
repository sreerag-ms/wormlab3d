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
            depth_min: int = 0,
            window_size: int = 1,
            window_image_diff_threshold: float = 3e3,
            use_master: bool = True,
            masks_threshold: float = 0.4,
            render_mode: str = RENDER_MODE_GAUSSIANS,
            second_render_prob: float = 0.5,
            filter_size: int = None,

            sigmas_init: float = 0.1,
            sigmas_min: float = 0.04,
            sigmas_max: float = 0.08,
            intensities_min: float = 0.4,

            curvature_mode: bool = False,
            curvature_deltas: bool = False,
            curvature_max: float = 2.,
            curvature_relaxation_factor: float = None,
            length_min: float = None,
            length_max: float = None,
            length_shrink_factor: float = None,
            length_init: float = None,
            length_warmup_steps: int = None,
            length_regrow_steps: int = None,
            dX0_limit: float = None,
            dl_limit: float = None,
            dk_limit: float = None,
            dpsi_limit: float = None,
            clamp_X0: bool = True,

            centre_shift_every_n_steps: int = None,
            centre_shift_threshold: float = None,
            centre_shift_adj: int = None,

            frame_skip: int = None,
            n_steps_init: int = 5000,
            n_steps_max: int = 500,
            convergence_tau_fast: int = 10,
            convergence_tau_slow: int = 100,
            convergence_threshold: float = 0.1,
            convergence_patience: int = 25,
            convergence_loss_target: float = 50.,

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
            loss_smoothness: float = 1.,
            loss_curvature: float = 1.,
            loss_temporal: float = 0.,
            loss_intersections: float = 0.,
            loss_alignment: float = 0.,
            loss_consistency: float = 0.,

            algorithm: str = OPTIMISER_ADAM,

            lr_cam_coeffs: float = 1e-5,
            lr_points: float = 1e-3,
            lr_sigmas: float = 1e-3,
            lr_exponents: float = 1e-3,
            lr_intensities: float = 1e-3,
            lr_filters: float = 1e-3,
            **kwargs
    ):
        self.load = load
        self.params_id = params_id

        self.depth = depth
        self.depth_min = depth_min
        self.window_size = window_size
        self.window_image_diff_threshold = window_image_diff_threshold
        self.use_master = use_master
        self.masks_threshold = masks_threshold
        self.render_mode = render_mode
        self.second_render_prob = second_render_prob
        if filter_size == 0:
            filter_size = None
        if filter_size is not None:
            assert filter_size >= 3, 'Filter size must be >= 3.'
            assert filter_size % 2 == 1, 'Filter size must be odd.'
        self.filter_size = filter_size

        self.sigmas_init = sigmas_init
        self.sigmas_min = sigmas_min
        self.sigmas_max = sigmas_max
        self.intensities_min = intensities_min

        self.curvature_mode = curvature_mode
        self.curvature_max = curvature_max
        if not curvature_mode:
            curvature_deltas = False
            length_min = None
            length_max = None
            length_shrink_factor = None
            length_init = None
            length_warmup_steps = None
            length_regrow_steps = None
        else:
            if length_min is None:
                length_min = 0.5
            if length_max is None:
                length_max = 2.
            if length_init is None:
                length_init = 0.2
            if length_warmup_steps is None:
                length_warmup_steps = 100
            if length_shrink_factor is not None:
                if length_regrow_steps is None:
                    length_regrow_steps = int(n_steps_max / 2)
                else:
                    assert length_regrow_steps <= n_steps_max
            elif length_shrink_factor is None:
                length_regrow_steps = None
        self.curvature_deltas = curvature_deltas
        self.curvature_relaxation_factor = curvature_relaxation_factor
        self.length_min = length_min
        self.length_max = length_max
        self.length_shrink_factor = length_shrink_factor
        self.length_init = length_init
        self.length_warmup_steps = length_warmup_steps
        self.length_regrow_steps = length_regrow_steps
        self.dX0_limit = dX0_limit
        self.dl_limit = dl_limit
        self.dk_limit = dk_limit
        self.dpsi_limit = dpsi_limit
        self.clamp_X0 = clamp_X0

        if not curvature_mode:
            centre_shift_every_n_steps = None
        self.centre_shift_every_n_steps = centre_shift_every_n_steps
        if centre_shift_every_n_steps is not None:
            if centre_shift_threshold is None:
                centre_shift_threshold = 0.01
            else:
                assert centre_shift_threshold > 0
            if centre_shift_adj is None:
                centre_shift_adj = 1
            else:
                assert centre_shift_adj >= 1
        else:
            centre_shift_threshold = None
            centre_shift_adj = None
        self.centre_shift_threshold = centre_shift_threshold
        self.centre_shift_adj = centre_shift_adj

        if frame_skip == 1:
            frame_skip = None
        self.frame_skip = frame_skip
        self.n_steps_init = n_steps_init
        self.n_steps_max = n_steps_max
        self.convergence_tau_fast = convergence_tau_fast
        self.convergence_tau_slow = convergence_tau_slow
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.convergence_loss_target = convergence_loss_target

        # Calculate total number of curve points
        N = 0
        for mi in range(depth_min, depth):
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
        self.loss_smoothness = loss_smoothness
        self.loss_curvature = loss_curvature
        self.loss_temporal = loss_temporal
        self.loss_intersections = loss_intersections
        self.loss_alignment = loss_alignment
        self.loss_consistency = loss_consistency

        assert algorithm in OPTIMISER_ALGORITHMS
        self.algorithm = algorithm

        self.lr_cam_coeffs = lr_cam_coeffs
        self.lr_points = lr_points
        self.lr_sigmas = lr_sigmas
        self.lr_exponents = lr_exponents
        self.lr_intensities = lr_intensities
        self.lr_filters = lr_filters

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
        group.add_argument('--depth-min', type=int, default=0,
                           help='Minimum depth of multi-scale curves to use. Default=0.')
        group.add_argument('--window-size', type=int, default=1,
                           help='Sliding window size.')
        group.add_argument('--window-image-diff-threshold', type=float, default=3e3,
                           help='Minimum image difference threshold between subsequent frames in the window.')
        group.add_argument('--use-master', type=str2bool, default=True,
                           help='Optimise a single parameter set for the full window. Default = True.')
        group.add_argument('--masks-threshold', type=float, default=0.4,
                           help='Threshold value to use for binarising the frame images. Default=0.4.')
        group.add_argument('--render-mode', type=str, default=RENDER_MODE_GAUSSIANS, choices=RENDER_MODES,
                           help='How to render the points, either as gaussian blobs (gaussians) or as circles (circles). Default=gaussians.')
        group.add_argument('--second-render-prob', type=float, default=0.5,
                           help='Probability of generating a second render scaled by tapered scores. Default=0.5.')
        group.add_argument('--filter-size', type=int,
                           help='Set >= 3 to train a set of 3 2D convolutional filters, one for each camera render.')

        group.add_argument('--sigmas-init', type=float, default=0.04,
                           help='Initial rendering sigmas for points. Default=0.04.')
        group.add_argument('--sigmas-min', type=float, default=0.04,
                           help='Minimum rendering sigma. Tapers to this value at head and tail. Default=0.04.')
        group.add_argument('--sigmas-max', type=float, default=0.08,
                           help='Maximum rendering sigma. Default=0.08.')
        group.add_argument('--intensities-min', type=float, default=0.4,
                           help='Minimum rendering intensity. Tapers to this value at head and tail. Default=0.4.')

        group.add_argument('--curvature-mode', type=str2bool, default=False,
                           help='Optimise the curvature rather than the points. Default=False.')
        group.add_argument('--curvature-deltas', type=str2bool, default=False,
                           help='Use future frame curvatures as delta values. Only applicable in curvature mode .Default=False.')
        group.add_argument('--curvature-max', type=float, default=2.,
                           help='Maximum allowed curvature in terms of coils/revolutions. '
                                'Used in curvature-loss for points-mode or as a hard limit when in curvature-mode. Default=2.')
        group.add_argument('--curvature-relaxation-factor', type=float,
                           help='The curvature is scaled by this factor at the start of each new frame, if defined.')
        group.add_argument('--length-min', type=float,
                           help='Minimum worm length (only used in curvature mode). Default=0.5.')
        group.add_argument('--length-max', type=float,
                           help='Maximum worm length (only used in curvature mode). Default=2.')
        group.add_argument('--length-shrink-factor', type=float,
                           help='The length is reduced by this factor at the start of each new frame, if defined.')
        group.add_argument('--length-init', type=float,
                           help='Initial worm length (only used in curvature mode). Default=0.2.')
        group.add_argument('--length-warmup-steps', type=int,
                           help='Number of initial steps to linearly grow worm length from length_init to length_min (only used in curvature mode). Default=100.')
        group.add_argument('--length-regrow-steps', type=int,
                           help='Number of steps to linearly grow worm length from shrunken length back to length_min (only used if length-shrink-factor is defined). Default=n_steps_max/2.')
        group.add_argument('--dX0-limit', type=float,
                           help='Maximum allowable change in X0 between batched frames (only used in curvature mode). Default=None.')
        group.add_argument('--dl-limit', type=float,
                           help='Maximum allowable change in length between batched frames (only used in curvature mode). Default=None.')
        group.add_argument('--dk-limit', type=float,
                           help='Maximum allowable change in scalar curvature between batched frames (only used in delta-curvatures mode). Default=None.')
        group.add_argument('--dpsi-limit', type=float,
                           help='Maximum allowable change in curvature angle between batched frames (only used in delta-curvatures mode). Default=None.')
        group.add_argument('--clamp-X0', type=str2bool,
                           help='Clamp the X0 (midpoint) coordinate to [-0.5,+0.5] before integration. Default=True.')

        group.add_argument('--centre-shift-every-n-steps', type=int,
                           help='Shift the curve along the midline to centre the scores every n steps. Default=None (no shifting).')
        group.add_argument('--centre-shift-threshold', type=float,
                           help='Start shifting when the central index is > threshold*N away from the midpoint. Default=0.01.')
        group.add_argument('--centre-shift-adj', type=int,
                           help='When centre shifting move at most this number of vertices in either direction. Default=1.')

        group.add_argument('--frame-skip', type=int,
                           help='Number of frames to skip between optimisations. Interim frames will populate parameters with linear interpolation.')
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
        group.add_argument('--convergence-loss-target', type=float, default=50.,
                           help='Keep going until this loss target is reached, even if converged.')

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
        group.add_argument('--loss-smoothness', type=float, default=1.,
                           help='Regularisation weight for penalising curve non-smoothness.')
        group.add_argument('--loss-curvature', type=float, default=1.,
                           help='Regularisation weight for penalising curvature.')
        group.add_argument('--loss-temporal', type=float, default=0.,
                           help='Temporal smoothing weight between frames.')
        group.add_argument('--loss-intersections', type=float, default=0.,
                           help='Self-intersections loss.')
        group.add_argument('--loss-alignment', type=float, default=0.,
                           help='Shape-space alignment loss.')
        group.add_argument('--loss-consistency', type=float, default=0.,
                           help='Loss to control difference between head and tail integrations.')

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
        group.add_argument('--lr-filters', type=float, default=1e-3,
                           help='Learning rate for the filters.')
        return group

    def get_db_params(self) -> dict:
        p = {}
        for k in MFParameters._fields.keys():
            if hasattr(self, k):
                p[k] = getattr(self, k)
        return p
