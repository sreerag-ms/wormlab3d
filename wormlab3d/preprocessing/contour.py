from typing import List

import cv2
import numpy as np

from wormlab3d import logger

CONT_MIN_AREA = 300
CONT_MAX_AREA = 20000
CONT_THRESH_DEFAULT = .4
MAX_CONTOURING_ATTEMPTS = 5


def find_contours(
        image: np.ndarray,
        thresh: float = 50.,
        maxval: int = 255,
        min_area: int = CONT_MIN_AREA,
        max_area: int = CONT_MAX_AREA,
) -> List[np.ndarray]:
    """
    Find all contours in the image.
    The image is first thresholded using `thresh` and `maxval` arguments before all contours are found.
    The possible contours are then filtered to remove any smaller than `min_area` or larger than `max_area`.
    """
    logger.debug(f'Finding contours (thresh={thresh:.2f}, maxval={maxval}, min_area={min_area}, max_area={max_area})')
    thresh = int(thresh)
    assert image.dtype == np.uint8

    # Threshold the image
    _, thresh_img = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)

    # Find the contours
    all_contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours to only take ones larger than min_area
    contours = []
    for c in all_contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            contours.append(c)
    logger.debug(
        f'Found {len(contours)} contours of suitable size in thresholded image. '
        f'({len(all_contours)} total).'
    )

    return contours


def contour_mask(
        image: np.ndarray,
        thresh: float = 50.,
        maxval: int = 255,
        min_area: int = CONT_MIN_AREA,
        max_area: int = CONT_MAX_AREA
):
    """
    Find the contours and create a mask from them.
    """
    contours = find_contours(image, thresh=thresh, maxval=maxval, min_area=min_area, max_area=max_area)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, -1)

    return mask


def contour_centre(contour: np.ndarray):
    """
    Find the centre-point of a contour.
    """
    M = cv2.moments(contour)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])

    return [x, y]
