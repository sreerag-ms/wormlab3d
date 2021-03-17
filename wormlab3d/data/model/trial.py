from typing import List, Dict

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

    def get_frames(self, filter=None) -> List[Frame]:
        if filter is None:
            filter = {}
        return Frame.objects.get(
            trial=self,
            **filter
        )

    def get_clips(self, filter: Dict=None) -> List[List[Frame]]:
        """
        Return a list of clips (frame-sequences) where frames in each clip match the given filter.
        Ensures that the tags are consistent through the clip.
        If no filter is provided this should just return a single clip.
        """
        frames = self.get_frames(filter)
        clips = []
        clip = []
        prev_num = -1
        prev_tags = []
        for f in frames:
            if f.frame_num == prev_num + 1 and f.tags == prev_tags:
                clip.append(f)
                prev_num += 1
            else:
                if len(clip):
                    clips.append(clip)
                clip = [f]
                prev_num = f.frame_num
                prev_tags = f.tags

        return clips
