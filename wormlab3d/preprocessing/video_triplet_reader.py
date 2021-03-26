from typing import List

import numpy as np
import pims

from wormlab3d.data.model.cameras import CAMERA_IDXS
from wormlab3d.preprocessing.video_reader import VideoReader


class VideoTripletReader:
    """
    Video triplet reader class.
    Creates 3 VideoReader objects and reads from them in sync
    """

    def __init__(
            self,
            video_paths: List[str],
            background_image_paths: List[str] = None
    ):
        if background_image_paths is None:
            background_image_paths = [None] * 3
        assert len(video_paths) == 3
        assert len(background_image_paths) == 3

        self.readers = [
            VideoReader(
                video_path=video_paths[c],
                background_image_path=background_image_paths[c]
            )
            for c in CAMERA_IDXS
        ]

        self.current_frame: int = -1

    def __iter__(self):
        return self

    def __next__(self) -> List[pims.Frame]:
        try:
            next_frame = self.current_frame + 1
            imgs = [self.readers[c][next_frame] for c in CAMERA_IDXS]
            self.current_frame = next_frame
        except IndexError:
            raise StopIteration()
        return imgs

    def __getitem__(self, idx) -> List[pims.Frame]:
        return [self.readers[c][idx] for c in CAMERA_IDXS]

    def set_frame_num(self, idx: int):
        assert 0 <= idx <= len(self.readers[0])
        self.current_frame = idx
        for r in self.readers:
            r.current_frame = idx

    def get_images(self) -> List[pims.Frame]:
        return self[self.current_frame]

    def find_contours(self, subtract_background: bool = True) -> List[List[np.ndarray]]:
        contours = []
        for c in CAMERA_IDXS:
            contours.append(
                self.readers[c].find_contours(subtract_background=subtract_background)
            )
        return contours

    def find_objects(self) -> List[np.ndarray]:
        centres = []
        for c in CAMERA_IDXS:
            centres.append(
                self.readers[c].find_objects()
            )
        return centres
