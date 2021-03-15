from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame


class Trial(Document):
    experiment = ReferenceField(Experiment)
    date = DateTimeField(required=True)
    num_frames = IntField(required=True, default=0)
    fps = FloatField()
    quality = FloatField()
    temperature = FloatField(min_value=0)
    comments = StringField()
    camera_1_avi = StringField(required=True)
    camera_2_avi = StringField(required=True)
    camera_3_avi = StringField(required=True)
    legacy_id = IntField(unique=True)
    legacy_data = DictField()

    def get_frame(self, frame_num) -> Frame:
        return Frame.objects.get(
            trial=self,
            frame_num=frame_num
        )
