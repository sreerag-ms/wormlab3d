from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.numpy_field import NumpyField


class Frame(Document):
    experiment = ReferenceField(Experiment, required=True)
    trial = ReferenceField('Trial', required=True)
    frame_num = IntField(required=True)

    # Zoomed-in/low-resolution images (we don't store high-resolution images)
    image_cam_1 = NumpyField()
    image_cam_2 = NumpyField()
    image_cam_3 = NumpyField()
    zoom_region_offset = NumpyField()

    # Indexes
    meta = {
        'indexes': [
            {
                'fields': ['trial', 'frame_num'],
                'unique': True
            }
        ]
    }

