from mongoengine import *

from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField


class Midline2D(Document):
    frame = ReferenceField('Frame', required=True)
    camera = IntField(choices=[1, 2, 3], required=True)

    # Midline coordinates
    X = NumpyField(required=True)

    # Hand-annotated
    user = StringField()

    # Model generated
    model = ReferenceField(Model)

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline2d'
    }
