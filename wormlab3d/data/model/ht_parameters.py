import datetime

from mongoengine import *


class HTParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    use_head_and_tail_optimisations = BooleanField(default=False)
    initial_head_and_tail_loss_weight = FloatField(default=0.0)
    n_steps_head_tail_refine = IntField(default=100)
    ht_freeze_length = BooleanField(default=True)
    central_freeze_applied = BooleanField(default=True)

    meta = {
        'collection': 'ht_parameters',
        'ordering': ['-created'],
    }
