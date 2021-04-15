import numpy as np

from tests.util import TEST_VIDEO_PATHS, TEST_BACKGROUND_PATHS, get_test_cameras
from wormlab3d import logger
from wormlab3d.preprocessing.video_triplet_reader import VideoTripletReader
from wormlab3d.toolkit.triangulate import triangulate

test_vid_points_3d = np.array([
    [-5.69152574e-01, -4.78796549e+00, 1.39149001e+03],
    [-6.05115116e-01, -4.77588269e+00, 1.39154505e+03],
    [-6.45131832e-01, -4.73863366e+00, 1.39161961e+03],
    [-6.64644814e-01, -4.72876084e+00, 1.39164365e+03],
    [-6.73678375e-01, -4.73614400e+00, 1.39164366e+03],
    [-6.83604035e-01, -4.72963089e+00, 1.39165747e+03],
    [-6.81062851e-01, -4.73624992e+00, 1.39165087e+03],
    [-5.66720917e-01, -4.72067417e+00, 1.39167666e+03],
    [-5.15002182e-01, -4.73036505e+00, 1.39165117e+03],
    [-4.49597388e-01, -4.73294833e+00, 1.39163878e+03],
    [-4.32793807e-01, -4.73900601e+00, 1.39163479e+03],
    [-4.15729128e-01, -4.73871118e+00, 1.39163075e+03],
    [-4.26102418e-01, -4.73254696e+00, 1.39162787e+03],
    [-4.53718415e-01, -4.75379115e+00, 1.39160590e+03],
    [-4.76652473e-01, -4.68622524e+00, 1.39172741e+03],
    [-4.77413185e-01, -4.62125974e+00, 1.39183088e+03],
    [-4.86730344e-01, -4.61120996e+00, 1.39185113e+03],
    [-5.37585807e-01, -4.61579777e+00, 1.39184569e+03],
    [-5.43613605e-01, -4.61937412e+00, 1.39183241e+03],
    [-5.54898792e-01, -4.62653721e+00, 1.39182982e+03],
    [-5.38441224e-01, -4.64324332e+00, 1.39183601e+03],
    [-3.83868130e-01, -4.64787912e+00, 1.39186631e+03],
    [-3.36000369e-01, -4.63009493e+00, 1.39188827e+03],
    [-3.08676874e-01, -4.64300747e+00, 1.39188725e+03],
    [-2.87532410e-01, -4.69063629e+00, 1.39183245e+03]
])


def test_triangulate_reproject():
    """
    Create a point in 3D and get the 2D projections.
    Now test if the triangulator finds the same 3D point and 2D reprojections.
    """
    cameras = get_test_cameras()
    cameras_model = cameras.get_camera_model_triplet()
    ref_point_3d = np.array([-0.5, -4, 140])
    ref_points_2d = cameras_model.project_to_2d(ref_point_3d)
    ref_points_2d = np.array(ref_points_2d)

    # Re-order the points to group by camera as expected by triangulator
    image_points = list(np.transpose(ref_points_2d, axes=(1, 0, 2)))

    # Check that the triangulator returns the same points
    res_3d = triangulate(
        image_points=image_points,
        cameras=cameras
    )

    assert len(res_3d) == 1
    assert res_3d[0]['source_point_idxs'] == (0, 0, 0)
    assert np.allclose(res_3d[0]['pt'], ref_point_3d)
    assert np.allclose(res_3d[0]['points_2d'], ref_points_2d)


def test_triangulate_reproject_noisy():
    """
    Create a point in 3D, get the 2D projections and add different amounts of noise.
    Now test if the triangulator still finds the same 3D point and 2D reprojections.
    """
    cameras = get_test_cameras()
    cameras_model = cameras.get_camera_model_triplet()
    ref_point_3d = np.array([-0.5, -4, 140])
    ref_points_2d = cameras_model.project_to_2d(ref_point_3d)
    ref_points_2d = np.array(ref_points_2d)

    # Add noise
    noises = [1e-5, 1e-3, 1e-1, 10]
    for noise_var in noises:
        logger.debug(f'Testing triangulation reprojection with noise level = {noise_var:.1E}')
        ref_points_2d = ref_points_2d.copy() + np.random.normal(scale=noise_var, size=ref_points_2d.shape)

        # Re-order the points to group by camera as expected by triangulator
        image_points = list(np.transpose(ref_points_2d, axes=(1, 0, 2)))

        # Check that the triangulator returns the same points
        res_3d = triangulate(
            image_points=image_points,
            cameras=cameras
        )

        # Adjust the tolerance with the amount of noise
        assert np.allclose(res_3d[0]['pt'], ref_point_3d, rtol=noise_var / 2)
        assert np.allclose(res_3d[0]['points_2d'], ref_points_2d, rtol=noise_var / 2)


def test_triangulate_two_3d_points():
    """
    Create two points in 3D and get the 2D projections for each.
    Now test if the triangulator can recover both 3D points.
    """
    cameras = get_test_cameras()
    cameras_model = cameras.get_camera_model_triplet()
    ref_points_3d = np.array([[-0.5, -4, 140], [-0.2, -2, 120]])
    ref_points_2d = cameras_model.project_to_2d(ref_points_3d)
    ref_points_2d = np.array(ref_points_2d)

    # Re-order the points to group by camera as expected by triangulator
    image_points = list(np.transpose(ref_points_2d, axes=(1, 0, 2)))

    # Check that the triangulator returns the same points
    res_3d = triangulate(
        image_points=image_points,
        cameras=cameras
    )
    assert len(res_3d) == 2
    for i, r in enumerate(res_3d):
        assert np.allclose(r['pt'], ref_points_3d[i])
        assert np.allclose(r['points_2d'], ref_points_2d[i])


def test_triangulate_video():
    """
    Read the test videos and check that a single 3d point can be found by triangulating
    the 2d points found in each of the views using the centres of contoured objects.
    """
    cameras = get_test_cameras()
    reader = VideoTripletReader(
        video_paths=TEST_VIDEO_PATHS,
        background_image_paths=TEST_BACKGROUND_PATHS
    )

    points_3d = []
    for _ in reader:
        logger.debug(f'Frame number = {reader.current_frame}')
        centres, thresholds = reader.find_objects()
        res_3d = triangulate(
            image_points=centres,
            cameras=cameras
        )
        assert len(res_3d) == 1
        point_3d = res_3d[0]['pt']
        points_2d = res_3d[0]['points_2d']
        assert point_3d.shape == (3,)
        assert len(points_2d) == 3
        for p2d in points_2d:
            assert p2d.shape == (2,)
        points_3d.append(point_3d.copy())

    points_3d = np.array(points_3d)
    assert np.allclose(points_3d, test_vid_points_3d)


if __name__ == '__main__':
    test_triangulate_reproject()
    test_triangulate_reproject_noisy()
    test_triangulate_two_3d_points()
    test_triangulate_video()
