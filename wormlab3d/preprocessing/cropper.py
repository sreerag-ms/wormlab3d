from typing import Tuple

import numpy as np


def crop_image(
        image: np.ndarray,
        centre_2d: np.ndarray,
        size: Tuple[int, int],
        fix_overlaps: bool = False
) -> np.ndarray:
    """
    Crop an image from the centre-out.
    """
    row_from = int(centre_2d[1] - size[1] / 2)
    row_to = int(centre_2d[1] + size[1] / 2)
    col_from = int(centre_2d[0] - size[0] / 2)
    col_to = int(centre_2d[0] + size[0] / 2)

    if fix_overlaps:
        top_overlap = -row_from if row_from < 0 else 0
        row_from = max(0, row_from)
        row_to = min(image.shape[0], row_to)

        left_overlap = -col_from if col_from < 0 else 0
        col_from = max(0, col_from)
        col_to = min(image.shape[1], col_to)
    else:
        assert row_from >= 0
        assert row_to <= image.shape[0]
        assert col_from >= 0
        assert col_to <= image.shape[1]

    crop = image[row_from:row_to, col_from:col_to].copy()

    if fix_overlaps and crop.shape != size:
        print('centre_2d', centre_2d)
        print('crop.shape', crop.shape)
        print('size', size)
        print('row_from', row_from)
        print('row_to', row_to)
        print('col_from', col_from)
        print('col_to', col_to)
        print('top_overlap', top_overlap)
        print('left_overlap', left_overlap)
        crop_fixed = np.zeros(size, dtype=image.dtype)
        crop_fixed[top_overlap:top_overlap + crop.shape[0], left_overlap:left_overlap + crop.shape[1]] = crop
        crop = crop_fixed

    assert crop.shape == size

    return crop
