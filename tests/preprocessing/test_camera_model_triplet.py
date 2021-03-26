import numpy as np

from wormlab3d.data.model.cameras import Cameras
from wormlab3d.toolkit.camera_model_triplet import CameraModelTriplet


def test_camera_model_triplet():
    # These values are taken from the calibration xml file "004_0.xml"
    cameras = Cameras()

    cameras.matrix = [
        np.array([[2.25706375e+05, 0.00000000e+00, 1.02350000e+03],
                  [0.00000000e+00, 2.25706375e+05, 1.02350000e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        np.array([[1.46225344e+05, 0.00000000e+00, 1.02350000e+03],
                  [0.00000000e+00, 1.46225344e+05, 1.02350000e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        np.array([[1.81441125e+05, 0.00000000e+00, 1.02350000e+03],
                  [0.00000000e+00, 1.81441125e+05, 1.02350000e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    ]

    cameras.pose = [
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]),
        np.array([[9.97283936e-01, 9.40146492e-05, -7.36526102e-02, 1.19816086e+02],
                  [-7.36519247e-02, -3.23557993e-03, -9.97278750e-01, 1.60297961e+03],
                  [-3.32067721e-04, 9.99994755e-01, -3.21986759e-03, 9.87169678e+02],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        np.array([[5.49758039e-03, -2.34067943e-02, 9.99710917e-01, -1.60751245e+03],
                  [-3.16075310e-02, 9.99222398e-01, 2.35691722e-02, -3.83101006e+01],
                  [-9.99485254e-01, -3.17279696e-02, 4.75347461e-03, 1.20445032e+03],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    ]

    cameras.distortion = [
        np.array([3.05524902e+02, 2.20938586e-02, 0.00000000e+00, 0.00000000e+00, 5.65077755e-07]),
        np.array([9.34072723e+01, 9.62209329e-02, 0.00000000e+00, 0.00000000e+00, 6.36056029e-06]),
        np.array([2.17883606e+02, 1.80288311e-02, 0.00000000e+00, 0.00000000e+00, 6.82736697e-07])
    ]

    test_point = np.array([-1.2, 3.11, 1609])

    # Test with distortion
    ref_with_distortion = np.array([
        (854.9463418783388, 1460.3348972986387),
        (1040.2161509582995, 791.3525026573891),
        (1164.7120265844462, 1436.5251947117722)
    ])
    cams = CameraModelTriplet(camera_models=cameras)
    points_2d = cams.project_to_2d(test_point)
    assert np.allclose(ref_with_distortion, points_2d)

    # Test no distortion
    ref_no_distortion = np.array([
        (855.16709136, 1459.76278822),
        (1040.21219786, 791.40740176),
        (1164.53462731, 1436.00632695)
    ])
    cams = CameraModelTriplet(camera_models=cameras, distort=False)
    points_2d = cams.project_to_2d(test_point)
    assert np.allclose(ref_no_distortion, points_2d)


if __name__ == '__main__':
    test_camera_model_triplet()
