from typing import Tuple, List, Final

import torch
import torch.nn.functional as F
from torch import nn
from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras


@torch.jit.script
def render_points(
        points: torch.Tensor,
        sigmas: torch.Tensor,
        intensities: torch.Tensor,
        camera_sigmas: torch.Tensor,
        cameras_intensities: torch.Tensor,
        image_size: int = PREPARED_IMAGE_SIZE[0]
):
    """
    Render points as Gaussian blobs onto a 2D image.
    """
    bs = points.shape[0]
    N = points.shape[2]
    device = points.device

    # Shift to [-1,+1]
    s2 = int(image_size / 2)
    points = (points - s2) / s2

    # Reshape points
    points = points.reshape(bs * 3, N, 2)
    points = points.clamp(min=-1, max=1)

    # Reshape and scale sigmas
    sigmas = sigmas.repeat_interleave(3, 0)[:, :, None, None]
    sfs = camera_sigmas.reshape(bs * 3)[:, None, None, None]
    sigmas = sigmas * sfs

    # Reshape and scale intensities
    intensities = intensities.repeat_interleave(3, 0)[:, :, None, None]
    sfs = cameras_intensities.reshape(bs * 3)[:, None, None, None]
    intensities = intensities * sfs

    # Build x and y grids
    grid = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Make (un-normalised) gaussian blobs centred at the coordinates
    mmp2 = m - p2
    dst = mmp2[..., 0]**2 + mmp2[..., 1]**2
    blobs = torch.exp(-(dst / (2 * sigmas**2)))

    # The rendering is the sum of blobs where each blob is scaled by its intensity
    masks = (blobs * intensities).sum(dim=1)
    masks = masks.clamp(max=1.)
    masks = masks.reshape(bs, 3, image_size, image_size)

    # Also return the blobs for scoring points individually
    blobs = blobs.reshape(bs, 3, N, image_size, image_size)

    return masks, blobs


class ProjectRenderScoreModel(nn.Module):
    image_size: Final[int]

    def __init__(self, image_size: int = PREPARED_IMAGE_SIZE[0]):
        super().__init__()
        self.image_size = image_size
        self.cams = torch.jit.script(DynamicCameras())

    def forward(
            self,
            cam_coeffs: torch.Tensor,
            points: List[torch.Tensor],
            masks_target: List[torch.Tensor],
            sigmas: List[torch.Tensor],
            intensities: List[torch.Tensor],
            camera_sigmas: torch.Tensor,
            camera_intensities: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor]
    ]:
        """
        Project the 3D points to 2D, render the projected points as blobs on an image and score each point for overlap.
        """
        D = len(points)
        masks = []
        points_2d = []
        scores = []
        points_smoothed = []
        sigmas_smoothed = []
        intensities_smoothed = []

        # Run the parameters through the model at each scale to get the outputs
        for d in range(D):
            points_d = points[d]
            masks_target_d = masks_target[d]
            sigmas_d = sigmas[d]
            intensities_d = intensities[d]

            # Smooth the points, sigmas and intensities using average pooling convolutions.
            if d > 1:
                ks = int(d / 2) * 2 + 1
                pad_size = int(ks / 2)

                # Smooth the curve points
                cp = torch.cat([
                    torch.repeat_interleave(points_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    points_d,
                    torch.repeat_interleave(points_d[:, -1].unsqueeze(1), pad_size, dim=1)
                ], dim=1)
                cp = cp.permute(0, 2, 1)
                cps = F.avg_pool1d(cp, kernel_size=ks, stride=1, padding=0)
                points_d = cps.permute(0, 2, 1)

                # Smooth the sigmas
                sigs = torch.cat([
                    sigmas_d[:, 1:pad_size + 1].flip(dims=(1,)),
                    sigmas_d,
                    sigmas_d[:, -pad_size - 1:-1].flip(dims=(1,)),
                ], dim=1)
                sigs = sigs[:, None, :]
                sigs = F.avg_pool1d(sigs, kernel_size=ks, stride=1, padding=0)
                sigmas_d = sigs.squeeze(1)

                # Smooth the intensities
                ints = torch.cat([
                    intensities_d[:, 1:pad_size + 1].flip(dims=(1,)),
                    intensities_d,
                    intensities_d[:, -pad_size - 1:-1].flip(dims=(1,)),
                ], dim=1)
                ints = ints[:, None, :]
                ints = F.avg_pool1d(ints, kernel_size=ks, stride=1, padding=0)
                intensities_d = ints.squeeze(1)

            # Project and render
            points_2d_d = self._project_to_2d(cam_coeffs, points_d, points_3d_base, points_2d_base)
            masks_d, blobs = render_points(points_2d_d, sigmas_d, intensities_d, camera_sigmas, camera_intensities,
                                           self.image_size)

            # Normalise blobs
            sum_ = blobs.amax(dim=(2, 3), keepdim=True)
            sum_ = sum_.clamp(min=1e-8)
            blobs_normed = blobs / sum_

            # Score the points
            scores_d = (blobs_normed * masks_target_d.unsqueeze(2)).sum(dim=(3, 4)).mean(dim=1)

            masks.append(masks_d)
            points_2d.append(points_2d_d.transpose(1, 2))
            scores.append(scores_d)
            points_smoothed.append(points_d)
            sigmas_smoothed.append(sigmas_d)
            intensities_smoothed.append(intensities_d)

        return masks, points_2d, scores, points_smoothed, sigmas_smoothed, intensities_smoothed

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
        image_centre_pt = torch.ones((bs, 1, 1, 2), dtype=torch.float32, device=device) * self.image_size / 2
        points_2d = points_2d - points_2d_base[:, :, None] + image_centre_pt

        return points_2d
