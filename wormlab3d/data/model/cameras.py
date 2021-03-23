from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.numpy_field import NumpyField

CAMERA_IDXS = [0, 1, 2]


class Cameras(Document):
    experiment = ReferenceField(Experiment)
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
    n_images_used = ListField(IntField(), required=True)
    pose = ListField(NumpyField(), required=True)
    matrix = ListField(NumpyField(), required=True)
    distortion = ListField(NumpyField(), required=True)
