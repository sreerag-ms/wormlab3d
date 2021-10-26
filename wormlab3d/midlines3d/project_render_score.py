from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras


def avg_pool_2d(grad, oob_grad_val=0., mode='constant'):
    # Average pooling with overlap and boundary values
    padded_grad = F.pad(grad, (1, 1, 1, 1), mode=mode, value=oob_grad_val)
    ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


def render_points(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]
    device = points.device

    # Shift to [-1,+1]
    points = (points - 100) / 100

    # Reshape points
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)

    # Reshape blur sigmas
    blur_sigmas = blur_sigmas.repeat_interleave(3, 0)[:, :, None, None]

    # Build x and y grids
    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])

    # 1 x 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Make (un-normalised) gaussian blobs centred at the coordinates
    mmp2 = m - p2
    dst = mmp2[..., 0]**2 + mmp2[..., 1]**2
    blobs = torch.exp(-(dst / (2 * blur_sigmas**2)))

    # Mask is the sum of the blobs
    masks = blobs.sum(dim=1)

    # Normalise
    masks = masks.clamp(max=1.)
    sum_ = masks.sum(dim=(1, 2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks_normed = masks / sum_

    # Reshape
    masks_normed = masks_normed.reshape(bs, 3, *PREPARED_IMAGE_SIZE)
    blobs = blobs.reshape(bs, 3, n_worm_points, *PREPARED_IMAGE_SIZE)

    return masks_normed, blobs


def render_curve(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]

    # Shift to [-1,+1]
    points = (points - 100) / 100

    # Reshape
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)
    a = points[:, :-1]
    b = points[:, 1:]

    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=a.device)

    yv, xv = torch.meshgrid([grid, grid])
    # 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]

    # B x N x 1 x 1 x 2
    a, b = a[:, :, None, None, :], b[:, :, None, None, :]
    t_min = sumprod(m - a, b - a) / \
            torch.max(sumprod(b - a, b - a), torch.tensor(1e-6, device=a.device))
    t_line = torch.clamp(t_min, 0.0, 1.0)

    # closest points on the line to every image pixel
    s = a + t_line * (b - a)

    d = sumprod(s - m, s - m, keepdim=False)

    # Blur line
    d = torch.sqrt(d + 1e-6)
    sig = (blur_sigmas[:, 1:] + blur_sigmas[:, :-1]) / 2
    lines = torch.exp(-d / (sig**2)[:, :, None, None])
    # lines = torch.exp(-(dst / (2 * blur_sigmas**2)[:, :, None, None]))

    # Sum the lines together to make the render
    masks = lines.sum(dim=1)
    masks = masks.clamp(max=1)

    # Normalise and reshape
    sum_ = masks.sum(dim=(1, 2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks = (masks / sum_).reshape(bs, 3, *PREPARED_IMAGE_SIZE)

    return masks


class ProjectRenderScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cams = DynamicCameras()

    def forward(self):
        raise RuntimeError('The forward method of this model is not enabled!')

    def forward_cloud(
            self,
            cam_coeffs: torch.Tensor,
            cloud_points: torch.Tensor,
            masks_target: torch.Tensor,
            blur_sigmas_cloud: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Render the point cloud
        cloud_points_2d = self._project_to_2d(cam_coeffs, cloud_points, points_3d_base, points_2d_base)
        masks_cloud, blobs = render_points(cloud_points_2d, blur_sigmas_cloud)

        # Normalise blobs
        sum_ = blobs.sum(dim=(2, 3), keepdim=True)
        sum_ = sum_.clamp(min=1e-8)
        blobs_normed = blobs / sum_

        # Calculate points scores
        cloud_points_scores = (blobs_normed * masks_target.unsqueeze(2)).sum(dim=(3, 4)).mean(dim=1)

        return masks_cloud, cloud_points_scores

    def forward_curve(
            self,
            cam_coeffs: torch.Tensor,
            curve_points: torch.Tensor,
            blur_sigmas_curve: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Render the curve
        curve_points_2d = self._project_to_2d(cam_coeffs, curve_points, points_3d_base, points_2d_base)
        masks_curves = render_curve(curve_points_2d, blur_sigmas_curve)

        # todo: Get the point scores for the curve points
        curve_points_scores = torch.zeros_like(blur_sigmas_curve)

        return masks_curves, curve_points_scores

    def _project_to_2d(
            self,
            cam_coeffs: torch.Tensor,
            points_3d: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ):
        bs = cam_coeffs.shape[0]
        device = cam_coeffs.device

        # Add the 3d centre point offset to centre on the camera
        points_3d = points_3d + points_3d_base[:, None]

        # Project 3D points to 2D
        points_2d = self.cams.forward(cam_coeffs, points_3d)

        # Re-centre according to 2D base points plus a (100,100) to put it in the centre of the cropped image
        image_centre_pt = torch.ones((bs, 1, 1, 2), dtype=torch.float32, device=device) * PREPARED_IMAGE_SIZE[0] / 2
        points_2d = points_2d - points_2d_base[:, :, None] + image_centre_pt

        return points_2d
