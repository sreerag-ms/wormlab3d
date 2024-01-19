import datetime

from mongoengine import *

PE_ANGLE_DIST_TYPES = ['norm', 'lognorm', 'cauchy', 'levy_stable', '2norm']
PE_PAUSE_TYPES = ['linear', 'quadratic']


class PEParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    batch_size = IntField()
    duration = FloatField()
    dt = FloatField()
    n_steps = IntField()

    rate_01 = FloatField()
    rate_10 = FloatField()
    rate_02 = FloatField()
    rate_20 = FloatField()
    rate_12 = FloatField()

    speeds_0_mu = FloatField()
    speeds_0_sig = FloatField()
    speeds_1_mu = FloatField()
    speeds_1_sig = FloatField()

    theta_dist_type = StringField(choices=PE_ANGLE_DIST_TYPES)
    theta_dist_params = ListField(FloatField())
    phi_dist_type = StringField(choices=PE_ANGLE_DIST_TYPES)
    phi_dist_params = ListField(FloatField())

    delta_type = StringField(choices=PE_PAUSE_TYPES)
    delta_max = FloatField()

    meta = {
        'collection': 'pe_parameters',
        'ordering': ['-created'],
    }

    def validate(self, clean=True):
        super().validate(clean=clean)
        n_steps = self.duration / self.dt
        if int(n_steps) - n_steps != 0:
            raise ValidationError('Duration and dt must produce an even number of time steps!')

    def save(self, *args, **kwargs):
        self.n_steps = int(self.duration / self.dt)
        res = super().save(*args, **kwargs)
        return res
