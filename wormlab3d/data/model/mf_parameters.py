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
    n_points_total = IntField()
    window_size = IntField()
    use_master = BooleanField()
    sigmas_init = FloatField()
    masks_threshold = FloatField()
    render_mode = StringField(choices=RENDER_MODES)
    frame_skip = IntField()

    n_steps_init = IntField()
    n_steps_max = IntField()
    convergence_tau_fast = IntField()
    convergence_tau_slow = IntField()
    convergence_threshold = FloatField()
    convergence_patience = IntField()

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
    loss_sigmas = FloatField()
    loss_exponents = FloatField()
    loss_intensities = FloatField()
    loss_smoothness = FloatField()
    loss_curvature = FloatField()
    loss_temporal = FloatField()

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
