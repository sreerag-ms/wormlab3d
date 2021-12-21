import datetime

from mongoengine import *
from wormlab3d.nn.args.optimiser_args import LOSSES, OPTIMISER_ALGORITHMS


class MFParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    depth = IntField()
    n_points_total = IntField()
    window_size = IntField()
    use_master = BooleanField()
    sigmas_init = FloatField()
    masks_threshold = FloatField()

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
    optimise_intensities = BooleanField()

    loss_masks_metric = StringField(choices=LOSSES)
    loss_masks_multiscale = BooleanField()
    loss_masks = FloatField()
    loss_neighbours = FloatField()
    loss_parents = FloatField()
    loss_aunts = FloatField()
    loss_scores = FloatField()
    loss_sigmas = FloatField()
    loss_intensities = FloatField()
    loss_smoothness = FloatField()
    loss_temporal = FloatField()

    algorithm = StringField(choices=OPTIMISER_ALGORITHMS)

    lr_cam_coeffs = FloatField()
    lr_points = FloatField()
    lr_sigmas = FloatField()
    lr_intensities = FloatField()

    meta = {
        'collection': 'mf_parameters',
        'ordering': ['-created'],
    }
