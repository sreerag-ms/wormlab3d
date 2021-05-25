import numpy as np
from mongoengine import *

from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK

M3D_SOURCE_WT3D = 'WT3D'
M3D_SOURCE_RECONST = 'reconst'
M3D_SOURCE_MODEL = 'model'

M3D_SOURCES = [
    M3D_SOURCE_WT3D,
    M3D_SOURCE_RECONST,
    M3D_SOURCE_MODEL
]


class Midline3D(Document):
    frame = ReferenceField('Frame', required=True)

    # Midline coordinates
    X = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    base_3d = NumpyField(shape=(3,), dtype=np.float32)
    error = FloatField()

    # Model/source used to generate this midline
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    model = ReferenceField(Model)

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline3d',
        'indexes': ['frame']
    }
