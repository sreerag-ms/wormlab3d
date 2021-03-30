import cv2

from wormlab3d import ROOT_PATH
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.cameras import CAMERA_IDXS

TEST_DATA_PATH = ROOT_PATH + '/tests/test-data'
TEST_VIDEO_PATHS = [
    TEST_DATA_PATH + '/Cam1_firstsec.avi',
    TEST_DATA_PATH + '/Cam2_firstsec.avi',
    TEST_DATA_PATH + '/Cam3_firstsec.avi',
]
TEST_BACKGROUND_PATHS = [
    TEST_DATA_PATH + '/background-images/Cam1_firstsec.png',
    TEST_DATA_PATH + '/background-images/Cam2_firstsec.png',
    TEST_DATA_PATH + '/background-images/Cam3_firstsec.png',
]


def get_test_cameras():
    """
    Instantiate a Cameras model from the xml calibration file.
    """
    calib_path = TEST_DATA_PATH + '/calibration-3d.xml'
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    cams = Cameras()
    cams.wormcv_version = fs.getNode('WormCV_Version').string()
    cams.opencv_version = fs.getNode('OpenCV_Version').string()
    cams.opencv_contrib_hash = fs.getNode('OpenCV_contrib_hash').string()
    cams.total_calib_images = int(fs.getNode('total_filenames').real())
    cams.pattern_height = float(fs.getNode('pattern_height').real())
    cams.pattern_width = float(fs.getNode('pattern_width').real())
    cams.square_size = float(fs.getNode('square_size').real())
    cams.flag_value = int(fs.getNode('flag_value').real())
    cams.n_mini_matches = int(fs.getNode('n_mini_matches').real())
    cams.n_cameras = int(fs.getNode('nCameras').real())
    cams.camera_type = int(fs.getNode('camera_type').real())
    cams.reprojection_error = float(fs.getNode('meanReprojectError').real())
    cams.n_images_used = [int(fs.getNode(f'images_used_{c}').real()) for c in CAMERA_IDXS]
    cams.pose = [fs.getNode(f'camera_pose_{c}').mat() for c in CAMERA_IDXS]
    cams.matrix = [fs.getNode(f'camera_matrix_{c}').mat() for c in CAMERA_IDXS]
    cams.distortion = [fs.getNode(f'camera_distortion_{c}').mat()[0] for c in CAMERA_IDXS]

    return cams
