from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.numpy_field import NumpyField
from wormlab3d.data.triplet_field import TripletField

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
    n_images_used = TripletField(IntField(), required=True)
    pose = TripletField(NumpyField(), required=True)
    matrix = TripletField(NumpyField(), required=True)
    distortion = TripletField(NumpyField(), required=True)
