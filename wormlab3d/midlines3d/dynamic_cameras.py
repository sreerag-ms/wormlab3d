import torch
from torch import nn

N_CAM_COEFFICIENTS = 21


class DynamicCameras(nn.Module):
    """
    A dynamic camera model modelled on the pinhole camera model with distortion from here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """

    def __init__(self, distort: bool = True):
        super().__init__()
        self.distort = distort
        self.register_buffer('fx', None, persistent=False)
        self.register_buffer('fy', None, persistent=False)
        self.register_buffer('translation', None, persistent=False)
        self.register_buffer('distortion', None, persistent=False)

    def extract_coefficients(self, coefficients: torch.Tensor):
        """
        Unpack the coefficients into the buffers.
        """
        assert coefficients.dim() == 3  # batch size
        assert coefficients.shape[1] == 3  # cameras come in triplets
        assert coefficients.shape[2] == N_CAM_COEFFICIENTS  # camera parameters
        bs = coefficients.shape[0]

        # Extract parameters
        self.fx = coefficients[:, :, 0].unsqueeze(dim=2)
        self.fy = coefficients[:, :, 1].unsqueeze(dim=2)
        self.cx = coefficients[:, :, 2].unsqueeze(dim=2)
        self.cy = coefficients[:, :, 3].unsqueeze(dim=2)
        self.rotation = coefficients[:, :, 4:13].reshape((bs, 3, 3, 3))
        self.translation = coefficients[:, :, 13:16]
        self.distortion = coefficients[:, :, 16:21]

    def forward(self, coefficients: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of camera-triplet parameters, project the batch of 3D points through
        each of the parametrised camera models to give a batch of triplets of 2D points.
        """
        assert points.dim() == 3
        assert points.shape[2] == 3  # points have 3 coordinates
        self.extract_coefficients(coefficients)

        # Rotate and translate
        xyz = torch.einsum('bcij,bpj->bcpi', self.rotation, points) \
              + self.translation.unsqueeze(2)

        # Project to 2D
        x = xyz[:, :, :, 0] / xyz[:, :, :, 2]
        y = xyz[:, :, :, 1] / xyz[:, :, :, 2]

        # Distort
        if self.distort:
            k1, k2, p1, p2, k3 = (self.distortion[:, :, i].unsqueeze(-1) for i in range(5))
            xy = x * y
            r2 = x**2 + y**2
            r4 = r2 * r2
            r6 = r4 * r2
            k_term = 1 + k1 * r2 + k2 * r4 + k3 * r6
            x = x * k_term + 2 * p1 * xy + p2 * (r2 + 2 * x**2)
            y = y * k_term + p1 * (r2 + 2 * y**2) + 2 * p2 * xy

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        out = torch.stack([u, v], dim=-1)

        return out
