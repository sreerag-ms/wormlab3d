import numpy as np
import pytest
from PIL import Image

from tests.util import TEST_BACKGROUND_PATHS
from wormlab3d import logger
from wormlab3d.preprocessing.cropper import crop_image

"""
We have a triplet video and have triangulated the 3d centre point.
Now we want to crop each frame to a small region around the worm.
"""
ref_point_3d = np.array([-0.5, -4, 140])


def test_crop_image():
    """
    Load a single image and check that the crop returns the right size.
    """
    logger.info('Test: test_crop_image')
    test_img = Image.open(TEST_BACKGROUND_PATHS[0])
    test_img = np.asarray(test_img)

    centres_2d = [
        np.array([200, 200]),
        np.array([500, 500]),
        np.array([800, 200]),
    ]
    sizes = [
        (2, 2),
        (50, 50),
        (10, 400),
        (400, 11),
    ]

    for centre_2d in centres_2d:
        img_centre_pt = test_img[centre_2d[0], centre_2d[1]]
        for size in sizes:
            logger.debug(f'Testing crop centre = {centre_2d}, size = {size}')
            crop = crop_image(
                image=test_img,
                centre_2d=centre_2d,
                size=size
            )
            assert crop.shape == size

            # Check the centre points match up, but give a +/- 1 pixel buffer
            crop_centre_pts = crop[int(size[0] / 2):int(size[0] / 2) + 2, int(size[1] / 2):int(size[1] / 2) + 2]
            assert (img_centre_pt == crop_centre_pts).any()


def test_crop_image_bad_sizes():
    """
    Cropping with bad sizes should raise raise an assertion error.
    """
    logger.info('Test: test_crop_image_bad_sizes')
    test_img = Image.open(TEST_BACKGROUND_PATHS[0])
    test_img = np.asarray(test_img)
    centre_2d = np.array([500, 500])
    sizes = [
        (2, 10000),
        (10000, 1),
        (10000, 10000),
    ]

    for size in sizes:
        logger.debug(f'Testing bad crop size = {size}')
        with pytest.raises(AssertionError):
            crop_image(
                image=test_img,
                centre_2d=centre_2d,
                size=size
            )


def test_crop_image_bad_overlap():
    """
    Cropping with a region which would overlap outside of image should raise an assertion error.
    """
    logger.info('Test: test_crop_image_bad_overlap')
    test_img = Image.open(TEST_BACKGROUND_PATHS[0])
    test_img = np.asarray(test_img)
    size = (100, 100)
    centres_2d = [
        np.array([0, 0]),
        np.array([0, 500]),
        np.array([0, 2000]),
        np.array([500, 0]),
        np.array([2000, 0]),
        np.array([2000, 500]),
        np.array([2000, 2000]),
        np.array([2000, 2000]),
    ]

    for centre_2d in centres_2d:
        logger.debug(f'Testing overlap errors with centre = {centre_2d}')
        with pytest.raises(AssertionError):
            crop_image(
                image=test_img,
                centre_2d=centre_2d,
                size=size
            )


if __name__ == '__main__':
    test_crop_image()
    test_crop_image_bad_sizes()
    test_crop_image_bad_overlap()
