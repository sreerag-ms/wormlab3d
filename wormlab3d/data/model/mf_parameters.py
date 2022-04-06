import datetime

from mongoengine import *

from wormlab3d.nn.args.optimiser_args import LOSSES, OPTIMISER_ALGORITHMS

RENDER_MODE_GAUSSIANS = 'gaussians'
RENDER_MODE_CIRCLES = 'circles'
RENDER_MODES = [
    RENDER_MODE_GAUSSIANS,
    RENDER_MODE_CIRCLES
]


class MFParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    depth = IntField()
    depth_min = IntField()
    n_points_total = IntField()
    window_size = IntField()
    window_image_diff_threshold = FloatField()
    use_master = BooleanField()
    masks_threshold = FloatField()
    render_mode = StringField(choices=RENDER_MODES)
    second_render_prob = FloatField()

    sigmas_init = FloatField()
    sigmas_min = FloatField()
    sigmas_max = FloatField()
    intensities_min = FloatField()

    curvature_mode = BooleanField()
    curvature_deltas = BooleanField()
    curvature_max = FloatField()
    curvature_relaxation_factor = FloatField()
    length_min = FloatField()
    length_max = FloatField()
    length_init = FloatField()
    length_warmup_steps = IntField()
    dX0_limit = FloatField()
    dl_limit = FloatField()
    dk_limit = FloatField()
    dpsi_limit = FloatField()

    frame_skip = IntField()
    n_steps_init = IntField()
    n_steps_max = IntField()
    convergence_tau_fast = IntField()
    convergence_tau_slow = IntField()
    convergence_threshold = FloatField()
    convergence_patience = IntField()
    convergence_loss_target = FloatField()

    optimise_cam_coeffs = BooleanField()
    optimise_cam_intrinsics = BooleanField()
    optimise_cam_rotations = BooleanField()
    optimise_cam_translations = BooleanField()
    optimise_cam_distortions = BooleanField()
    optimise_cam_shifts = BooleanField()

    optimise_sigmas = BooleanField()
    optimise_exponents = BooleanField()
    optimise_intensities = BooleanField()

    loss_masks_metric = StringField(choices=LOSSES)
    loss_masks_multiscale = BooleanField()
    loss_masks = FloatField()
    loss_neighbours = FloatField()
    loss_parents = FloatField()
    loss_aunts = FloatField()
    loss_scores = FloatField()
    loss_smoothness = FloatField()
    loss_curvature = FloatField()
    loss_temporal = FloatField()
    loss_intersections = FloatField()

    # --- Deprecated
    loss_sigmas = FloatField()
    loss_exponents = FloatField()
    loss_intensities = FloatField()
    # ---

    algorithm = StringField(choices=OPTIMISER_ALGORITHMS)

    lr_cam_coeffs = FloatField()
    lr_points = FloatField()
    lr_sigmas = FloatField()
    lr_exponents = FloatField()
    lr_intensities = FloatField()

    meta = {
        'collection': 'mf_parameters',
        'ordering': ['-created'],
    }
