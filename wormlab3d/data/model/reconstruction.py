import datetime

from mongoengine import *
from wormlab3d.data.model.mf_parameters import MFParameters
from wormlab3d.data.model.midline3d import M3D_SOURCES


class Reconstruction(Document):
    created = DateTimeField(required=True, default=datetime.datetime.now)
    updated = DateTimeField(required=True, default=datetime.datetime.now)
    trial = ReferenceField('Trial', required=True)
    start_frame = IntField(required=True)
    end_frame = IntField(required=True)
    midlines = ListField(LazyReferenceField('Midline3D'))
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    mf_parameters = ReferenceField(MFParameters)

    meta = {
        'indexes': [
            'trial',
            'source',
            'source_file',
            'mf_parameters',
            {
                'fields': ['trial', 'source', 'source_file', 'mf_parameters'],
                'unique': True
            },
        ]
    }

    def save(self, *args, **kwargs):
        self.updated = datetime.datetime.now()
        return super().save(*args, **kwargs)
