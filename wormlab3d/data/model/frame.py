from typing import List

from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.object_point import ObjectPoint
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.numpy_field import NumpyField
from wormlab3d.data.triplet_field import TripletField

PREPARED_IMAGE_SIZE = (200, 200)


class Frame(Document):
    experiment = ReferenceField(Experiment, required=True)
    trial = ReferenceField('Trial', required=True)
    frame_num = IntField(required=True)

    # Triangulations
    centres_2d = TripletField(ListField(ListField()))
    centre_3d = EmbeddedDocumentField(ObjectPoint)

    # Prepared images (we don't store high-resolution images)
    images = TripletField(NumpyField())

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

    def get_midlines2d(
            self,
            manual_only: bool = False,
            generated_only: bool = False,
            filters: dict = None
    ) -> List[Midline2D]:
        """
        Fetch all the 2D midlines associated with this frame.
        """
        assert not (manual_only and generated_only)
        if filters is None:
            filters = {}
        filters = {'frame': self, **filters}
        if manual_only:
            filters['user__exists'] = True
            filters['model__exists'] = False
        if generated_only:
            filters['user__exists'] = False
            filters['model__exists'] = True

        return Midline2D.objects(**filters)

    def centres_2d_available(self) -> bool:
        """
        Check that we have 2d centre points available in each camera view
        """
        image_points_valid = True
        if len(self.centres_2d) != 3:
            image_points_valid = False
        else:
            for centres_2d_cam in self.centres_2d:
                if len(centres_2d_cam) == 0:
                    image_points_valid = False
                    break
        return image_points_valid
