from typing import List, Dict, Union

from mongoengine import *

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.data.util import fix_path
from wormlab3d.preprocessing.video_reader import VideoReader
from wormlab3d.preprocessing.video_triplet_reader import VideoTripletReader


class Trial(Document):
    id = SequenceField(primary_key=True)
    experiment = ReferenceField(Experiment)
    date = DateTimeField(required=True)
    trial_num = IntField()
    num_frames = IntField()
    n_frames = TripletField(IntField())
    n_frames_max = IntField(required=True, default=0)
    n_frames_min = IntField(required=True, default=0)
    fps = FloatField()
    quality = FloatField()
    temperature = FloatField(min_value=0)
    comments = StringField()
    videos = TripletField(StringField(), required=True)
    backgrounds = TripletField(StringField())
    legacy_id = IntField(unique=True)
    legacy_data = DictField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readers = [None] * 3
        self.triplet_reader = None

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

    def get_cameras(self, best: bool = True, fallback_to_experiment: bool = True) -> Union[Cameras, List[Cameras]]:
        """
        Fetch the camera models for this trial.
        If best=False then returns a list of all associated, otherwise picks the best according to reprojection_error.
        """
        cameras = Cameras.objects(
            trial=self
        )

        # If no cameras are found for the trial, return what we can from the experiment
        if len(cameras) == 0:
            if fallback_to_experiment:
                return self.experiment.get_cameras(best=best)
            return None

        if best:
            return cameras.first()

        return cameras

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

    def get_video_reader(self, camera_idx: int, reload: bool = False) -> VideoReader:
        """
        Instantiate a video reader for the recording taken by the target camera.
        """
        assert camera_idx in CAMERA_IDXS

        if self.readers[camera_idx] is None or reload:
            vid_path = fix_path(self.videos[camera_idx])
            if len(self.backgrounds) > 0:
                bg_path = fix_path(self.backgrounds[camera_idx])
            else:
                bg_path = None

            self.readers[camera_idx] = VideoReader(
                video_path=vid_path,
                background_image_path=bg_path
            )

        return self.readers[camera_idx]

    def get_video_triplet_reader(self, reload: bool = False) -> VideoTripletReader:
        """
        Instantiate a video-triplet reader to read all recordings in sync.
        """
        if self.triplet_reader is None or reload:
            vid_paths = [fix_path(self.videos[c]) for c in CAMERA_IDXS]
            if len(self.backgrounds) > 0:
                bg_paths = [fix_path(self.backgrounds[c]) for c in CAMERA_IDXS]
            else:
                bg_paths = None

            self.triplet_reader = VideoTripletReader(
                video_paths=vid_paths,
                background_image_paths=bg_paths
            )

        return self.triplet_reader
