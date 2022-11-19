from pathlib import Path
from typing import List, Dict, Union, Tuple

import numpy as np
from mongoengine import *

from wormlab3d import CAMERA_IDXS, TRACKING_VIDEOS_PATH, PREPARED_IMAGE_SIZE_DEFAULT
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.data.util import fix_path
from wormlab3d.preprocessing.video_reader import VideoReader
from wormlab3d.preprocessing.video_triplet_reader import VideoTripletReader

TRIAL_QUALITY_BEST = 10
TRIAL_QUALITY_GOOD = 9
TRIAL_QUALITY_MINOR_ISSUES = 7
TRIAL_QUALITY_TRACKING_ISSUES = 5
TRIAL_QUALITY_VIDEO_ISSUES = 3
TRIAL_QUALITY_BROKEN = 1

TRIAL_QUALITY_CHOICES = {
    TRIAL_QUALITY_BEST: 'Best',
    TRIAL_QUALITY_GOOD: 'Good',
    TRIAL_QUALITY_MINOR_ISSUES: 'Minor issues',
    TRIAL_QUALITY_TRACKING_ISSUES: 'Tracking issues',
    TRIAL_QUALITY_VIDEO_ISSUES: 'Video issues',
    TRIAL_QUALITY_BROKEN: 'Broken',
}


class TrialQualityChecks(EmbeddedDocument):
    fps = BooleanField(default=False)
    durations = BooleanField(default=False)
    brightnesses = BooleanField(default=False)
    triangulations = BooleanField(default=False)
    triangulations_fixed = BooleanField(default=False)
    tracking_video = BooleanField(default=False)
    syncing = BooleanField(default=False)
    crop_size = BooleanField(default=False)
    verified = BooleanField(default=False)


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
    temperature = FloatField(min_value=0)
    comments = StringField()
    videos = TripletField(StringField(), required=True)
    videos_uncompressed = TripletField(StringField(), required=True)
    backgrounds = TripletField(StringField())
    legacy_id = IntField(unique=True)
    legacy_data = DictField()
    quality = IntField()
    quality_checks = EmbeddedDocumentField(TrialQualityChecks)
    crop_size = IntField(default=PREPARED_IMAGE_SIZE_DEFAULT)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readers = [None] * 3
        self.triplet_reader = None

    def get_frame(self, frame_num: int) -> Frame:
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

    def get_cameras(self, best: bool = True, fallback_to_experiment: bool = True, source: str = None) -> Union[
        Cameras, List[Cameras]]:
        """
        Fetch the camera models for this trial.
        If best=False then returns a list of all associated, otherwise picks the best according to reprojection_error.
        """
        filters = {'trial': self}
        if source is not None:
            filters['source'] = source
        cameras = Cameras.objects(**filters)

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

    def get_video_reader(
            self,
            camera_idx: int,
            reload: bool = False,
            use_uncompressed_videos: bool = False
    ) -> VideoReader:
        """
        Instantiate a video reader for the recording taken by the target camera.
        """
        assert camera_idx in CAMERA_IDXS

        if self.readers[camera_idx] is None or reload:
            if use_uncompressed_videos:
                vid_path = fix_path(self.videos_uncompressed[camera_idx])
            else:
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

    def get_video_triplet_reader(
            self,
            reload: bool = False,
            use_uncompressed_videos: bool = False
    ) -> VideoTripletReader:
        """
        Instantiate a video-triplet reader to read all recordings in sync.
        """
        if self.triplet_reader is None or reload:
            if use_uncompressed_videos:
                vid_paths = [fix_path(self.videos_uncompressed[c]) for c in CAMERA_IDXS]
            else:
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

    def get_tracking_data(
            self,
            fixed: bool,
            prune_missing: bool = False,
            start_frame: int = None,
            end_frame: int = None,
            return_2d_points: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch the 3D tracking data.
        """
        points = []
        timestamps = []

        # Match this trial and restrict frame range.
        matches = {'trial': self.id}
        frame_num_matches = {}
        if start_frame is not None:
            frame_num_matches['$gte'] = start_frame
        if end_frame is not None:
            frame_num_matches['$lte'] = end_frame
        if len(frame_num_matches):
            matches['frame_num'] = frame_num_matches

        pipeline = [
            {'$match': matches},
            {'$project': {
                '_id': 0,
                'frame_num': 1,
                'p3d': '$centre_3d' + ('_fixed' if fixed else ''),
            }},
            {'$sort': {'frame_num': 1}},
        ]
        cursor = Frame.objects().aggregate(pipeline)

        for res in cursor:
            if 'p3d' in res and res['p3d'] is not None:
                if return_2d_points:
                    pt = res['p3d']['reprojected_points_2d']
                else:
                    pt = res['p3d']['point_3d']
            elif not prune_missing:
                if return_2d_points:
                    pt = np.array([[0, 0], [0, 0], [0, 0]])
                else:
                    pt = np.array([0, 0, 0])
            elif len(points) == 0:
                continue
            else:
                break
            points.append(pt)
            timestamps.append(res['frame_num'] / self.fps)

        points = np.array(points)
        timestamps = np.array(timestamps)

        return points, timestamps

    @property
    def num_reconstructions(self) -> int:
        from wormlab3d.data.model import Reconstruction
        return Reconstruction.objects(trial=self).count()

    @property
    def duration(self):
        from datetime import datetime
        if self.fps > 0:
            dt = datetime.fromtimestamp(round(self.n_frames_min / self.fps))
        else:
            dt = datetime.fromtimestamp(0)
        return dt

    @property
    def tracking_video_path(self) -> Path:
        return TRACKING_VIDEOS_PATH / f'{self.id:03d}.mp4'

    @property
    def has_tracking_video(self) -> bool:
        return self.tracking_video_path.exists()

    def find_next_frame_with_different_images(self, start_frame: int, threshold: float, direction: int = 1) -> Frame:
        """
        Find the next frame from the one given which has an image difference greater than the threshold.
        """
        f0 = self.get_frame(start_frame)
        if f0.images is None or len(f0.images) != 3:
            raise RuntimeError('Start frame does not have a triplet of prepared images.')
        images0 = np.stack(f0.images)
        diff = 0
        step = direction * 15
        frame_num = start_frame + step
        while diff < threshold or abs(step) > 1:
            f1 = self.get_frame(frame_num)
            if f1.images is None or len(f1.images) != 3:
                raise RuntimeError('Cannot find subsequent frame with a triplet of prepared images.')
            images1 = np.stack(f1.images)
            diff = np.sum((images0 - images1)**2)

            # If the difference is large enough step back in the with smaller steps.
            if step > 0 and diff > threshold:
                step = min(-1, -int(step / 2))

            # If we're stepping backwards and are now below threshold again step forward with smaller steps.
            elif step < 0 and diff < threshold:
                step = min(1, -int(step / 2))

            frame_num += step
        return f1
