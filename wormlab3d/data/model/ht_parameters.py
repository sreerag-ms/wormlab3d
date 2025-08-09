import datetime

from mongoengine import *


class HTParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    read_head_and_tail_coordinates = BooleanField(default=True)
    head_and_tail_coordinates = StringField()
    loss_head_and_tail = FloatField(default=0.0)
    n_steps_head_tail_refine = IntField(default=100)
    ht_freeze_length = BooleanField(default=True)

    meta = {
        'collection': 'ht_parameters',
        'ordering': ['-created'],
    }
