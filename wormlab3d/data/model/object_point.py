from mongoengine import *

from wormlab3d.data.triplet_field import TripletField


class ObjectPoint(EmbeddedDocument):
    cameras = ReferenceField('Cameras')
    point_3d = TripletField(FloatField(), required=True)
    error = FloatField()
    source_point_idxs = TripletField()
    reprojected_points_2d = TripletField(ListField())

    # Order by descending error
    meta = {
        'ordering': ['-error']
    }
