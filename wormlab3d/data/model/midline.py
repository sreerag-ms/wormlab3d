from mongoengine import *

from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField


class Midline(Document):
    frame = ReferenceField(Frame, required=True)

    # Midline coordinates
    X = NumpyField(required=True)

    # Model used to generate this midline
    model = ReferenceField(Model)

    # todo: what are these used for and rename fields to something more meaningful
    base_3d = NumpyField()
    seg_or = FloatField()
    and_carve = IntField()
    ex_carve = IntField()
    E = FloatField()

    # these are from the range keys
    pts3d_v2 = NumpyField()
    kpwt = NumpyField()
    yellow = NumpyField()
