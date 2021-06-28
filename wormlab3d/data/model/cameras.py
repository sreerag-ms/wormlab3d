from typing import Tuple

import numpy as np
from mongoengine import *

from wormlab3d.data.numpy_field import NumpyField
from wormlab3d.data.triplet_field import TripletField

CAM_SOURCE_WT3D = 'WT3D'
CAM_SOURCE_ANNEX = 'ANNEX'

CAM_SOURCES = [
    CAM_SOURCE_WT3D,
    CAM_SOURCE_ANNEX,
]


class Cameras(Document):
    experiment = ReferenceField('Experiment')
    trial = ReferenceField('Trial')
    frame = ReferenceField('Frame')
    timestamp = DateTimeField()
    source = StringField(choices=CAM_SOURCES, required=True)
    source_file = StringField()
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
    n_images_used = TripletField(IntField())
    pose = TripletField(NumpyField(shape=(4, 4), dtype=np.float64), required=True)
    matrix = TripletField(NumpyField(shape=(3, 3), dtype=np.float64), required=True)
    distortion = TripletField(NumpyField(shape=(5,), dtype=np.float64), required=True)

    # Indexes
    meta = {
        'indexes': [
            'reprojection_error',
            'experiment',
            'trial',
            'frame',
        ],
        'ordering': ['+reprojection_error']
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shifts = None

    def set_shifts(self, shifts: 'CameraShifts'):
        """Link CameraShifts to this Cameras instance."""
        self.shifts = shifts

    def get_shift(self, camera_idx: int) -> Tuple[float, float]:
        """
        Return the camera shift for the given camera index.
        Assumes the default camera layout; possible bug.
        """
        if self.shifts is None:
            return 0, 0
        if camera_idx == 0:
            return self.shifts.dx, 0
        if camera_idx == 1:
            return 0, -self.shifts.dy
        if camera_idx == 2:
            return 0, self.shifts.dz

    def get_camera_model_triplet(self) -> 'CameraModelTriplet':
        """
        Instantiate a CameraModelTriplet parametrised by self.
        """
        from wormlab3d.toolkit.camera_model_triplet import CameraModelTriplet
        return CameraModelTriplet(
            camera_models=self
        )
