import datetime

from mongoengine import *

from wormlab3d.midlines3d.args.network_args import ENCODING_MODES


class MFModelParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    n_cloud_points = IntField(required=True)
    n_curve_points = IntField(required=True)
    curve_mode = StringField(required=True, choices=ENCODING_MODES)
    n_curve_basis_fns = IntField()
    blur_sigmas_cloud_init = FloatField()
    blur_sigmas_curve_init = FloatField()
    blur_sigma_vols = FloatField()
    max_revolutions = FloatField()

    meta = {
        'collection': 'mf_model_parameters',
        'ordering': ['-created'],
    }
