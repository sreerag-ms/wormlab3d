from wormlab3d import ROOT_PATH, logger
from wormlab3d.preprocessing.contour import contour_centre, VideoContour
#from wormlab3d.preprocessing.img_util import my_imshow

TEST_DATA_PATH = ROOT_PATH + '/data/test-data'


def test_find_contours_in_video():
	video_path = TEST_DATA_PATH + '/Cam1_firstsec.avi'
	bg_path = TEST_DATA_PATH + '/background-images/Cam1_firstsec.png'
	vidcap = VideoContour(video_path, bg_path)

	success, image, contours = vidcap.read()
	while success:
		for c in contours:
			logger.info(f'center = {contour_centre(c)}')
			# my_imshow(f, center(c))
		success, image, contours = vidcap.read()


if __name__ == '__main__':
	test_find_contours_in_video()
	# test_triangulate_2d_video()
