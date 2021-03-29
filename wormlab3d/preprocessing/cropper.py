from typing import Tuple

import numpy as np


def crop_image(image: np.ndarray, centre_2d: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Crop an image from the centre-out.
    """
    lb = int(centre_2d[0] - size[0] / 2)
    rb = int(centre_2d[0] + size[0] / 2)
    bb = int(centre_2d[1] - size[1] / 2)
    tb = int(centre_2d[1] + size[1] / 2)

    assert lb >= 0
    assert rb <= image.shape[0]
    assert bb >= 0
    assert tb <= image.shape[1]

    crop = image[bb:tb, lb:rb].copy()

    assert crop.shape == size

    return crop
