import datetime

from mongoengine import *


class SwSimulationParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    worm_length = IntField(required=True)
    duration = FloatField(required=True)
    dt = FloatField(required=True)

    meta = {
        'ordering': ['-created'],
    }
