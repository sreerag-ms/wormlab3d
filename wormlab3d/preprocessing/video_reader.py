from typing import List

import cv2
import numpy as np
import pims

from wormlab3d import logger
from wormlab3d.data.annex import fetch_from_annex, is_annexed_file
from wormlab3d.preprocessing.contour import CONT_THRESH_DEFAULT, contour_mask, CONT_MIN_AREA, find_contours, \
    MAX_CONTOURING_ATTEMPTS, contour_centre
from wormlab3d.preprocessing.create_bg_lp import Accumulate


class VideoReader:
    """
    Video reader class. This can read either any format supported by opencv or pims.
    Works as an iterator over video frames or can be indexed directly.
    Provides methods to:
        1. Extract contours from frames.
        2. Find centre points of any detected objects.
        3. Generate a background image from the video using a low pass temporal filter.
    """

    def __init__(
            self,
            video_path: str,
            background_image_path: str = None,
            contour_thresh: float = CONT_THRESH_DEFAULT
    ):
        # If the video is a link try and fetch it from the annex
        if is_annexed_file(video_path):
            fetch_from_annex(video_path)

        try:
            # standard video reader
            self.video = pims.PyAVReaderTimed(video_path)
        except Exception as e:
            logger.error(f'{type(e)}, {e}')
            # TODO add specialist for seq files once we have test data
            # generic video reader
            self.video = pims.open(video_path)

        # Open background image
        if background_image_path is not None:
            if is_annexed_file(background_image_path):
                fetch_from_annex(background_image_path)
            self.background = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
            if self.background is None:
                logger.error(f'Cannot open background image: {background_image_path}')
        else:
            self.background = None

        self.current_frame: int = -1
        self.contour_thresh = contour_thresh
        logger.debug(f'VideoReader(video={video_path}, bg={background_image_path})')

    @property
    def fps(self):
        """Frames per second"""
        return self.video.frame_rate

    @property
    def frame_size(self):
        shape = self.video.frame_shape
        if len(shape) == 3:
            shape = shape[:-1]
        return shape

    def __iter__(self):
        return self

    def __next__(self) -> pims.Frame:
        try:
            next_frame = self.current_frame + 1
            img = self[next_frame]
            self.current_frame = next_frame
        except IndexError:
            raise StopIteration()
        return img

    def __getitem__(self, idx):
        img = self.video[idx]
        grey = self._as_grey(img)
        return grey

    def __len__(self):
        return len(self.video)

    def set_frame_num(self, idx: int):
        assert 0 <= idx <= len(self)
        self.current_frame = idx

    @staticmethod
    @pims.pipeline
    def _as_grey(frame: pims.Frame) -> pims.Frame:
        dtype = frame.dtype
        frame = pims.as_grey(frame)
        return frame.astype(dtype)

    @staticmethod
    @pims.pipeline
    def _invert(frame: pims.Frame) -> pims.Frame:
        frame = frame.copy()
        frame.data = np.invert(frame.data)
        return frame

    def find_contours(self, subtract_background: bool = True) -> List[np.ndarray]:
        """
        Find the contours in the image.
        """
        image = self[self.current_frame].copy()

        # Invert image (white worms on black background)
        image = self._invert(image)

        # Subtract background
        if subtract_background:
            if self.background is None:
                raise ValueError('No background image available to subtract.')
            if self.background.shape != image.shape:
                raise ValueError('Image size from video does not match background image size!')
            bg_inv = self._invert(self.background)
            image = cv2.subtract(image.copy(), bg_inv)

        # Get max brightness
        max_brightness = image.max()

        # Find the contours
        contours = []
        cont_thresh = self.contour_thresh
        attempts = 0
        while len(contours) == 0:
            mask = contour_mask(
                image,
                thresh=max(3, max_brightness * cont_thresh),
                maxval=max_brightness,
                min_area=CONT_MIN_AREA
            )
            mask_dil = cv2.dilate(mask, None, iterations=10)
            contours = find_contours(
                image=mask_dil,
                min_area=CONT_MIN_AREA
            )

            # If no contours found, decrease the threshold and try again
            if len(contours) == 0:
                attempts += 1
                cont_thresh /= 2
                if attempts > MAX_CONTOURING_ATTEMPTS:
                    raise RuntimeError('Could not find any contours in image!')

        return contours

    def find_objects(self) -> np.ndarray:
        """
        Finds contours in the current frame and returns the centre point coordinates.
        """
        contours = self.find_contours()
        centres = []
        for c in contours:
            centres.append(contour_centre(c))
        centres = np.stack(centres)
        return centres

    def get_background(self) -> np.ndarray:
        """
        Create a background image by low-pass filtering the video.
        todo: redo the low-pass filter so it is more understandable
        """
        logger.info('Generating background')
        # Create the filter
        a = Accumulate(self.frame_size)
        self.current_frame = -1
        for image in self:
            a.push(image)
        bg = a.get()
        return bg
