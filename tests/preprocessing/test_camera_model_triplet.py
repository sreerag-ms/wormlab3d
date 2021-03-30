import numpy as np

from tests.util import get_test_cameras
from wormlab3d.toolkit.camera_model_triplet import CameraModelTriplet


def test_camera_model_triplet():
    """
    Check that the test camera projects a 3d point to a reference triplet of 2d points.
    """
    cameras = get_test_cameras()
    test_point = np.array([-2.9, -4.7, 1391])

    # Test with distortion
    ref_with_distortion = np.array([
        (615.54230655, 362.32718648),
        (782.44815866, 1181.09646427),
        (913.93541325, 304.14621938)
    ])
    cams = CameraModelTriplet(camera_models=cameras)
    points_2d = cams.project_to_2d(test_point)
    assert np.allclose(ref_with_distortion, points_2d)

    # Test no distortion
    ref_no_distortion = np.array([
        (617.24060141, 365.07959539),
        (782.47674487, 1181.07777499),
        (914.280055, 306.40898828)
    ])
    cams = CameraModelTriplet(camera_models=cameras, distort=False)
    points_2d = cams.project_to_2d(test_point)
    assert np.allclose(ref_no_distortion, points_2d)


if __name__ == '__main__':
    test_camera_model_triplet()
