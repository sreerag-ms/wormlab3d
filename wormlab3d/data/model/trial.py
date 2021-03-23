from typing import List, Dict

from mongoengine import *

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.util import fix_path
from wormlab3d.preprocessing.video_reader import VideoReader

CAMERA_IDXS = [0, 1, 2]


class Trial(Document):
    id = SequenceField(primary_key=True)
    experiment = ReferenceField(Experiment)
    date = DateTimeField(required=True)
    num_frames = IntField(required=True, default=0)
    fps = FloatField()
    quality = FloatField()
    temperature = FloatField(min_value=0)
    comments = StringField()
    camera_0_avi = StringField(required=True)
    camera_1_avi = StringField(required=True)
    camera_2_avi = StringField(required=True)
    camera_0_background = StringField()
    camera_1_background = StringField()
    camera_2_background = StringField()
    legacy_id = IntField(unique=True)
    legacy_data = DictField()

    def get_frame(self, frame_num) -> Frame:
        return Frame.objects.get(
            trial=self,
            frame_num=frame_num
        )

    def get_frames(self, filters: Dict = None) -> List[Frame]:
        if filters is None:
            filters = {}
        return Frame.objects(
            trial=self,
            **filters
        )

    def get_clips(self, filters: Dict = None) -> List[List[Frame]]:
        """
        Return a list of clips (frame-sequences) where frames in each clip match the given filter.
        Ensures that the tags are consistent through the clip.
        If no filter is provided this should just return a single clip.
        """
        frames = self.get_frames(filters)
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

    def get_video_reader(self, camera_idx: int) -> VideoReader:
        assert camera_idx in CAMERA_IDXS
        vid_path = fix_path(getattr(self, f'camera_{camera_idx}_avi'))
        bg_path = fix_path(getattr(self, f'camera_{camera_idx}_background'))

        return VideoReader(
            video_path=vid_path,
            background_image_path=bg_path
        )
