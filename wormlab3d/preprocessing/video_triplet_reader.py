from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pims

from wormlab3d import CAMERA_IDXS, logger
from wormlab3d.preprocessing.contour import CONT_THRESH_RATIO_DEFAULT
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
            imgs = self[next_frame]
            self.set_frame_num(next_frame)
        except IndexError:
            raise StopIteration()
        return imgs

    def __getitem__(self, idx: int) -> List[pims.Frame]:
        return [self.readers[c][idx] for c in CAMERA_IDXS]

    def get_frame_counts(self) -> List[int]:
        return [len(self.readers[c]) for c in CAMERA_IDXS]

    def set_frame_num(self, idx: int):
        """Set the frame number for all readers. (Skips assertion checks.)"""
        self.current_frame = idx
        for r in self.readers:
            r.current_frame = idx

    def get_images(self, invert: bool = False, subtract_background: bool = False) -> OrderedDict[int, pims.Frame]:
        images = OrderedDict()
        for c in CAMERA_IDXS:
            try:
                img = self.readers[c].get_image(
                    invert=invert,
                    subtract_background=subtract_background
                )
                images[c] = img
            except (IndexError, AssertionError) as e:
                logger.warning(f'Error reading image from camera {c}: {e}')
        return images

    def find_contours(self, subtract_background: bool = True, cont_threshold_ratios: List[float] = None) \
            -> Tuple[OrderedDict[int, List[np.ndarray]], OrderedDict[int, int]]:
        if cont_threshold_ratios is None:
            cont_threshold_ratios = [CONT_THRESH_RATIO_DEFAULT] * 3
        contours = OrderedDict()
        thresholds = OrderedDict()
        for c in CAMERA_IDXS:
            try:
                conts, final_threshold = self.readers[c].find_contours(
                    subtract_background=subtract_background,
                    cont_threshold_ratio=cont_threshold_ratios[c]
                )
                contours[c] = conts
                thresholds[c] = final_threshold
            except (IndexError, AssertionError) as e:
                logger.warning(f'Error finding contours in camera {c}: {e}')
        return contours, thresholds

    def find_objects(self, cont_threshold_ratios: List[float] = None, cam_idxs: List[int] = None) \
            -> Tuple[OrderedDict[int, list], OrderedDict[int, int]]:
        if cont_threshold_ratios is None:
            cont_threshold_ratios = [CONT_THRESH_RATIO_DEFAULT] * 3
        if cam_idxs is None:
            cam_idxs = CAMERA_IDXS
        centres = OrderedDict()
        thresholds = OrderedDict()
        for c in cam_idxs:
            try:
                objs, threshold = self.readers[c].find_objects(cont_threshold_ratio=cont_threshold_ratios[c])
                centres[c] = objs
                thresholds[c] = threshold
            except (IndexError, AssertionError) as e:
                logger.warning(f'Error finding objects in camera {c}: {e}')
                centres[c] = []
                thresholds[c] = np.inf
        return centres, thresholds

    def close(self):
        for c in CAMERA_IDXS:
            self.readers[c].close()
