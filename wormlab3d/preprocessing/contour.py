from typing import List

import cv2
import numpy as np

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
