from wormlab3d import ROOT_PATH, logger
from wormlab3d.preprocessing.contour import contour_centre
from wormlab3d.preprocessing.img_util import my_imshow
from wormlab3d.preprocessing.video_reader import VideoReader

TEST_DATA_PATH = ROOT_PATH + '/data/test-data'


def test_find_contours_in_video():
    video_path = TEST_DATA_PATH + '/Cam1_firstsec.avi'
    bg_path = TEST_DATA_PATH + '/background-images/Cam1_firstsec.png'

    video = VideoReader(video_path, bg_path)
    for image in video:
        contours = video.find_contours()
        for c in contours:
            my_imshow(image, contour_centre(c))
            logger.info(f'frame #{video.current_frame}, center = {contour_centre(c)}')


if __name__ == '__main__':
    test_find_contours_in_video()
