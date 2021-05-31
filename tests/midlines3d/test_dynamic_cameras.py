import torch

from tests.util import get_test_cameras
from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model import Cameras, CameraShifts
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras

dtype = torch.float32


def _get_camera_coeffs(cameras: Cameras):
    # Combine the coefficients into single tensors
    fx = torch.tensor([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
    fy = torch.tensor([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
    cx = torch.tensor([cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
    cy = torch.tensor([cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
    rotation = torch.tensor([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
    translation = torch.tensor([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
    distortion = torch.tensor([cameras.distortion[c] for c in CAMERA_IDXS])
    if cameras.shifts is not None:
        shifts = torch.tensor([cameras.shifts.dx, cameras.shifts.dy, cameras.shifts.dz])
    else:
        shifts = torch.tensor([0, 0, 0])

    cam_coeffs = torch.cat([
        fx.unsqueeze(1),
        fy.unsqueeze(1),
        cx.unsqueeze(1),
        cy.unsqueeze(1),
        rotation.reshape(3, -1),
        translation.reshape(3, -1),
        distortion.reshape(3, -1),
        shifts.unsqueeze(1),
    ], dim=1)
    cam_coeffs = cam_coeffs.to(dtype)

    return cam_coeffs


def test_dynamic_cameras():
    """
    Check that the dynamic cameras module projects batches of 3d points to batches of triplets of 2d points.
    Each item in the batch must have a triplet of camera coefficients.
    """

    # Test the same point with two different camera models, one with distortion and the other without
    batch_size = 2
    n_points = 5

    # Test with distortion
    ref_with_distortion = torch.tensor([
        (615.54230655, 362.32718648),
        (782.44815866, 1181.09646427),
        (913.93541325, 304.14621938)
    ], dtype=dtype)

    # Test no distortion
    ref_no_distortion = torch.tensor([
        (617.24060141, 365.07959539),
        (782.47674487, 1181.07777499),
        (914.280055, 306.40898828)
    ], dtype=dtype)

    # Combine points
    test_points = torch.tensor([[[-2.9, -4.7, 1391]] * n_points] * batch_size, dtype=dtype)
    ref_points = torch.stack([
        torch.stack([ref_with_distortion] * n_points),
        torch.stack([ref_no_distortion] * n_points),
    ]).permute(dims=(0, 2, 1, 3))

    # Use the test camera coefficients
    cameras = get_test_cameras()
    cam1_coeffs = _get_camera_coeffs(cameras)

    # No distortion in cams2
    cam2_coeffs = cam1_coeffs.clone()
    cam2_coeffs[:, 16:21] = 0.

    # Combine camera parameters in a batch of 2
    cam_coeffs = torch.stack([cam1_coeffs, cam2_coeffs])
    cam_coeffs = cam_coeffs.to(dtype)

    # Using the batch of coefficients and the batch of points return a batch of 2d triplets
    cams = DynamicCameras()
    points_2d = cams.forward(
        coefficients=cam_coeffs,
        points=test_points
    )

    assert points_2d.shape == ref_points.shape
    assert points_2d.shape == (batch_size, 3, n_points, 2)
    assert torch.allclose(points_2d, ref_points)


def test_camera_shifts():
    """
    Check that the dynamic cameras module applies the shifts in the same way as the pinhole cams.
    """

    # Test points
    test_points = torch.tensor([
        [-2.9, -4.7, 1391],
        [-2.5, -4.3, 1411],
    ], dtype=dtype)

    # Test shifts
    test_shifts = [(0, 0, 0), (1, 1, 1), (-10, 10, 100)]

    # Use the test camera coefficients
    cameras = get_test_cameras()
    dyn_cams = DynamicCameras()

    for shifts in test_shifts:
        cam_shifts = CameraShifts(dx=shifts[0], dy=shifts[1], dz=shifts[1])
        cameras.set_shifts(cam_shifts)
        pinhole_cams = cameras.get_camera_model_triplet()
        cam_coeffs = _get_camera_coeffs(cameras).unsqueeze(0)

        for test_point in test_points:
            pc2d = pinhole_cams.project_to_2d(test_point)
            pc2d = torch.tensor(pc2d, dtype=dtype).squeeze()
            dc2d = dyn_cams.forward(cam_coeffs, points=test_point.reshape(1, 1, 3))
            dc2d = dc2d.squeeze()
            assert torch.allclose(pc2d, dc2d)


if __name__ == '__main__':
    test_dynamic_cameras()
    test_camera_shifts()
