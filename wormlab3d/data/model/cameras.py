import numpy as np
from mongoengine import *

from wormlab3d.data.numpy_field import NumpyField
from wormlab3d.data.triplet_field import TripletField


class Cameras(Document):
    experiment = ReferenceField('Experiment')
    trial = ReferenceField('Trial')
    timestamp = DateTimeField(required=True)
    wormcv_version = StringField()
    opencv_version = StringField()
    opencv_contrib_hash = StringField()
    total_calib_images = IntField()
    pattern_height = FloatField()
    pattern_width = FloatField()
    square_size = FloatField()
    flag_value = IntField()
    n_mini_matches = IntField()
    n_cameras = IntField()
    camera_type = IntField()
    reprojection_error = FloatField()
    n_images_used = TripletField(IntField(), required=True)
    pose = TripletField(NumpyField(shape=(4, 4), dtype=np.float64), required=True)
    matrix = TripletField(NumpyField(shape=(3, 3), dtype=np.float64), required=True)
    distortion = TripletField(NumpyField(shape=(5,), dtype=np.float64), required=True)

    # Indexes
    meta = {
        'indexes': ['reprojection_error'],
        'ordering': ['+reprojection_error']
    }

    def get_camera_model_triplet(self) -> 'CameraModelTriplet':
        """
        Instantiate a CameraModelTriplet parametrised by self.
        """
        from wormlab3d.toolkit.camera_model_triplet import CameraModelTriplet
        return CameraModelTriplet(
            camera_models=self
        )
