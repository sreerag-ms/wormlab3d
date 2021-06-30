import datetime

from mongoengine import *


class SwSimulationParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    worm_length = IntField(required=True)
    duration = FloatField(required=True)
    dt = FloatField(required=True)
    K = FloatField(required=True)
    K_rot = FloatField(required=True)
    A = FloatField(required=True)
    B = FloatField(required=True)
    C = FloatField(required=True)
    D = FloatField(required=True)

    meta = {
        'ordering': ['-created'],
    }
