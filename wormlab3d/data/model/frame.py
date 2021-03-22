from typing import List

from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.numpy_field import NumpyField


class Frame(Document):
    experiment = ReferenceField(Experiment, required=True)
    trial = ReferenceField('Trial', required=True)
    frame_num = IntField(required=True)

    # Triangulations
    centres_cam_1 = ListField()
    centres_cam_2 = ListField()
    centres_cam_3 = ListField()

    # Zoomed-in/low-resolution images (we don't store high-resolution images)
    image_cam_1 = NumpyField()
    image_cam_2 = NumpyField()
    image_cam_3 = NumpyField()
    zoom_region_offset = NumpyField()

    # Tags
    tags = ListField(ReferenceField(Tag))

    # Indexes
    meta = {
        'indexes': [
            {
                'fields': ['trial', 'frame_num'],
                'unique': True
            }
        ],
        'ordering': ['+trial', '+frame_num']
    }

    def get_midlines2d(self, manual_only: bool = False, generated_only: bool = False) -> List[Midline2D]:
        """
        Fetch all the 2D midlines associated with this frame.
        """
        assert not (manual_only and generated_only)

        filters = {'frame': self}
        if manual_only:
            filters['user__exists'] = True
            filters['model__exists'] = False
        if generated_only:
            filters['user__exists'] = False
            filters['model__exists'] = True

        return Midline2D.objects(**filters)
