import torch

from tests.util import get_test_cameras
from wormlab3d import CAMERA_IDXS
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras


def test_dynamic_cameras():
    """
    Check that the dynamic cameras module projects batches of 3d points to batches of triplets of 2d points.
    Each item in the batch must have a triplet of camera coefficients.
    """
    dtype = torch.float32

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

    # Combine the coefficients into single tensors
    fx = torch.tensor([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
    fy = torch.tensor([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
    # cx = torch.tensor([cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
    # cy = torch.tensor([cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
    rotation = torch.tensor([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
    translation = torch.tensor([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
    distortion = torch.tensor([cameras.distortion[c] for c in CAMERA_IDXS])

    cameras1_parameters = torch.cat([
        fx.unsqueeze(1),
        fy.unsqueeze(1),
        rotation.reshape(3, -1),
        translation.reshape(3, -1),
        distortion.reshape(3, -1)
    ], dim=1)

    # No distortion in cams2
    cameras2_parameters = cameras1_parameters.clone()
    cameras2_parameters[:, 14:] = 0.

    # Combine camera parameters in a batch of 2
    camera_parameters = torch.stack([cameras1_parameters, cameras2_parameters])
    camera_parameters = camera_parameters.to(dtype)

    # Using the batch of coefficients and the batch of points return a batch of 2d triplets
    cams = DynamicCameras()
    points_2d = cams.forward(
        coefficients=camera_parameters,
        points=test_points
    )

    assert points_2d.shape == ref_points.shape
    assert points_2d.shape == (batch_size, 3, n_points, 2)
    assert torch.allclose(points_2d, ref_points)


if __name__ == '__main__':
    test_dynamic_cameras()
