from mongoengine import *


class CameraShifts(Document):
    frame = ReferenceField('Frame', required=True, unique=True)
    dx = FloatField(required=True)
    dy = FloatField(required=True)
    dz = FloatField(required=True)

    # Indexes
    meta = {
        'indexes': ['frame'],
    }
