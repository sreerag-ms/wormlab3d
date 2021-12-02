from mongoengine import *

from wormlab3d.data.model.midline3d import M3D_SOURCES
from wormlab3d.data.model.model import Model


class Reconstruction(Document):
    trial = ReferenceField('Trial', required=True)
    start_frame = IntField(required=True)
    end_frame = IntField(required=True)
    midlines = ListField(LazyReferenceField('Midline3D'), required=True)
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    model = ReferenceField(Model)

    meta = {
        'indexes': [
            'trial',
            'source',
            'source_file',
            'model',
            {
                'fields': ['trial', 'source', 'source_file', 'model'],
                'unique': True
            },
        ]
    }
