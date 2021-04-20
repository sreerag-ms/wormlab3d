from typing import Tuple

import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import N_WORM_POINTS
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS, DynamicCameras
from wormlab3d.midlines3d.points_to_masks import PointsToMasks
from wormlab3d.nn.models.basenet import BaseNet


class EncDec(nn.Module):
    def __init__(self, encoder: BaseNet, blur_sigma: float = 0):
        super().__init__()
        self.encoder = encoder
        self.cams = DynamicCameras()
        self.register_buffer('blur_sigma', torch.tensor(blur_sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x.shape[0]
        net_output = self.encoder(x)

        # Split into coordinates and camera coefficients
        points_3d = net_output[:, :3 * N_WORM_POINTS].reshape((bs, N_WORM_POINTS, 3))
        coeffs = net_output[:, 3 * N_WORM_POINTS:].reshape((bs, 3, N_CAM_COEFFICIENTS))

        # Project 3D points to 2D and generate masks
        points_2d = self.cams.forward(coeffs, points_3d)
        masks = PointsToMasks.apply(points_2d)

        # Apply a gaussian blur to the masks
        if self.blur_sigma > 0:
            masks = gaussian_blur(masks, kernel_size=7, sigma=self.blur_sigma.item())
            mask_maxs = torch.amax(masks, dim=(2, 3), keepdim=True)
            mask_mins = torch.amin(masks, dim=(2, 3), keepdim=True)
            mask_ranges = mask_maxs - mask_mins

            masks = torch.where(
                mask_ranges > 0,
                (masks - mask_mins) / mask_ranges,
                torch.zeros_like(masks)
            )

        return points_3d, coeffs, points_2d, masks

    def get_n_params(self) -> int:
        return self.encoder.get_n_params()

    def calc_norms(self, p: int = 2) -> float:
        return self.encoder.calc_norms(p=p)
