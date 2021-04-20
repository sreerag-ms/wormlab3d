import torch
from torch import nn

N_CAM_COEFFICIENTS = 21


class DynamicCameras(nn.Module):
    """
    A dynamic camera model modelled on the pinhole camera model with distortion from here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """

    def forward(self, coefficients: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of camera-triplet parameters, project the batch of 3D points through
        each of the parametrised camera models to give a batch of triplets of 2D points.
        """
        assert len(coefficients) == len(points)
        assert coefficients.dim() == 3  # batch size
        assert coefficients.shape[1] == 3  # cameras come in triplets
        assert coefficients.shape[2] == N_CAM_COEFFICIENTS  # camera parameters
        assert points.dim() == 3
        assert points.shape[2] == 3  # points have 3 coordinates

        # Extract parameters
        fx = coefficients[:, :, 0].unsqueeze(dim=2)
        fy = coefficients[:, :, 1].unsqueeze(dim=2)
        cx = coefficients[:, :, 2].unsqueeze(dim=2)
        cy = coefficients[:, :, 3].unsqueeze(dim=2)
        rotation = coefficients[:, :, 4:13].reshape((coefficients.shape[0], 3, 3, 3))
        translation = coefficients[:, :, 13:16].unsqueeze(dim=2)
        distortion = coefficients[:, :, 16:].unsqueeze(dim=2)

        # Rotate and translate
        xyz = torch.einsum('bcij,bpj->bcpi', rotation, points) + translation

        # Project to 2D
        x1 = xyz[:, :, :, 0] / xyz[:, :, :, 2]
        y1 = xyz[:, :, :, 1] / xyz[:, :, :, 2]
        x1y1 = x1 * y1

        # Distort
        k1, k2, p1, p2, k3 = (distortion[:, :, :, i] for i in range(5))

        r2 = x1**2 + y1**2
        r4 = r2 * r2
        r6 = r4 * r2

        k_term = 1 + k1 * r2 + k2 * r4 + k3 * r6

        x2 = x1 * k_term + 2 * p1 * x1y1 + p2 * (r2 + 2 * x1**2)
        y2 = y1 * k_term + p1 * (r2 + 2 * y1**2) + 2 * p2 * x1y1

        u = fx * x2 + cx
        v = fy * y2 + cy

        out = torch.stack([u, v], dim=-1)

        return out
