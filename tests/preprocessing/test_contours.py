from tests.util import TEST_BACKGROUND_PATHS, TEST_VIDEO_PATHS
from wormlab3d import logger
from wormlab3d.preprocessing.video_reader import VideoReader


def test_find_contours_in_video1_subtract_bg():
    """
    Load the first test video and background image and check that we can find a single contour
    for each frame using the default settings.
    """
    video = VideoReader(TEST_VIDEO_PATHS[0], TEST_BACKGROUND_PATHS[0], contour_thresh_ratio=0.4)
    for _ in video:
        logger.debug(f'Frame number = {video.current_frame}')
        contours, _ = video.find_contours(subtract_background=True)
        assert len(contours) == 1


def test_find_contours_in_video2_no_subtract_bg():
    """
    Load the second test video but don't subtract the background image.
    Also uses a higher threshold, which tests the automatic threshold reduction.
    """
    video = VideoReader(TEST_VIDEO_PATHS[1], TEST_BACKGROUND_PATHS[1], contour_thresh_ratio=0.9)
    for _ in video:
        logger.debug(f'Frame number = {video.current_frame}')
        contours, _ = video.find_contours(subtract_background=False)
        assert len(contours) == 1


if __name__ == '__main__':
    test_find_contours_in_video1_subtract_bg()
    test_find_contours_in_video2_no_subtract_bg()
