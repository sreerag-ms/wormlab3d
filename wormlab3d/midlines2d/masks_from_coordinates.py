from typing import Tuple

import numpy as np
from skimage.draw import line, line_aa
from skimage.filters import gaussian
from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT


def make_segmentation_mask(
        X: np.ndarray,
        blur_sigma: float = None,
        draw_mode: str = 'line_aa',
        image_size: Tuple[int] = (PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
        raise_on_empty: bool = True
) -> np.ndarray:
    """
    Turn a list of coordinates into a segmentation mask by drawing the coordinates onto a mask
    either using (anti-aliased or not) straight-line interpolations or just the individual pixels.
    Optionally apply a gaussian blur to the mask and then renormalise -- this has the effect of making the midline thicker.
    """
    X = X.round().astype(np.uint16)
    mask = np.zeros(image_size, dtype=np.float32)

    # Anti-aliased lines between coordinates
    if draw_mode == 'line_aa':
        for i in range(len(X) - 1):
            rr, cc, val = line_aa(X[i, 1], X[i, 0], X[i + 1, 1], X[i + 1, 0])
            rr = rr.clip(min=0, max=image_size[0] - 1)
            cc = cc.clip(min=0, max=image_size[1] - 1)
            mask[rr, cc] = val

    # Simpler single-pixel lines between coordinates
    elif draw_mode == 'line':
        for i in range(len(X) - 1):
            rr, cc = line(X[i, 1], X[i, 0], X[i + 1, 1], X[i + 1, 0])
            rr = rr.clip(min=0, max=image_size[0] - 1)
            cc = cc.clip(min=0, max=image_size[1] - 1)
            mask[rr, cc] = 1

    # Draw the coordinate pixels only
    elif draw_mode == 'pixels':
        mask[X[:, 1], X[:, 0]] = 1

    else:
        raise RuntimeError(f'Unrecognised draw_mode: {draw_mode}')

    # Apply a gaussian blur and then re-normalise to "fatten" the midline
    if blur_sigma is not None:
        mask = gaussian(mask, sigma=blur_sigma)

        # Normalise to [0-1] with float32 dtype
        mask_range = mask.max() - mask.min()
        if mask_range > 0:
            mask = (mask - mask.min()) / mask_range
        elif raise_on_empty:
            raise RuntimeError(f'Mask range zero!')

    return mask
