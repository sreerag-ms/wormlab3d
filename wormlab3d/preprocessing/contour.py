from typing import List, Tuple

import cv2
import numpy as np

from wormlab3d import logger
from wormlab3d.preprocessing.contour import contour_mask, CONT_MIN_AREA, find_contours

CONT_MIN_AREA = 300
CONT_THRESH_DEFAULT = .4
MAX_CONTOURING_ATTEMPTS = 5


def find_contours(
        image: np.ndarray,
        thresh: float = 50.,
        maxval: int = 255,
        min_area: int = 100
) -> List[np.ndarray]:
    """
    Find all contours in the image.
    The image is first thresholded using `thresh` and `maxval` arguments before all contours are found.
    The possible contours are then filtered to remove any smaller than `min_area`.
    """
    thresh = int(thresh)
    assert image.dtype == np.uint8
    # image = np.uint8(image)

    # Threshold the image
    _, thresh_img = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)

    # Find the contours
    all_contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours so we only take ones larger than min_area
    contours = []
    for c in all_contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            contours.append(c)

    return contours


def contour_mask(
        image: np.ndarray,
        thresh: float = 50.,
        maxval: int = 255,
        min_area: int = 100
):
    """
    Find the contours and create a mask from them.
    """
    contours = find_contours(image, thresh=thresh, maxval=maxval, min_area=min_area)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, 0, 255, -1)

    return mask


def contour_centre(contour: np.ndarray):
    """
    Find the centre-point of a contour.
    """
    M = cv2.moments(contour)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])

    return x, y


class VideoContour:
    """
    Read a video file and find object contours for each frame.
    """

    def __init__(
            self,
            video_path: str,
            background_image_path: str,
            contour_thresh: float = CONT_THRESH_DEFAULT
    ):
        # Open video file
        self.reader = cv2.VideoCapture(video_path)
        if not self.reader.isOpened():
            raise IOError(f'Cannot open video: {video_path}')

        # Open background image
        self.background = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
        if self.background is None:
            raise IOError(f'Cannot open background image: {background_image_path}')

        self.contour_thresh = contour_thresh
        logger.debug(f'VideoContour(video={video_path}, bg={background_image_path})')

    def read(self) -> Tuple[bool, np.ndarray, List[np.ndarray]]:
        """
        Read the next frame from the video and find the contours.
        """
        success, image = self.reader.read()

        if not success:
            logger.debug('End of video stream.')
            return False, None, []

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours = self._find_contours(image)

        return success, image, contours

    def _find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find the contours in the image.
        """
        if self.background.shape != image.shape:
            raise ValueError('Image size from video does not match background image size!')

        # Subtract background image and get max brightness
        image = cv2.subtract(self.background, image.copy())
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
            contours = find_contours(mask_dil)

            # If no contours found, decrease the threshold and try again
            if len(contours) == 0:
                attempts += 1
                cont_thresh /= 2
                if attempts > MAX_CONTOURING_ATTEMPTS:
                    raise RuntimeError('Could not find any contours in image!')

        return contours
