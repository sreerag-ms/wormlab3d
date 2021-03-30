import numpy as np
from mongoengine import *

from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK


class Midline3D(Document):
    frame = ReferenceField(Frame, required=True)

    # Midline coordinates
    X = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)

    # Model used to generate this midline
    model = ReferenceField(Model, required=True)

    # todo: what are these used for and rename fields to something more meaningful
    base_3d = NumpyField(shape=(3,), dtype=np.float32)
    seg_or = FloatField()
    and_carve = IntField()
    ex_carve = IntField()
    E = FloatField()

    # these are from the range keys
    pts3d_v2 = NumpyField()
    kpwt = NumpyField()
    yellow = NumpyField()

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline3d'
    }
