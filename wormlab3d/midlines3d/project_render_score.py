from typing import Tuple, List, Final

import torch
import torch.nn.functional as F
from torch import nn

from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.data.model.mf_parameters import RENDER_MODE_GAUSSIANS, RENDER_MODES
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras


@torch.jit.script
def make_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks, device=device)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


@torch.jit.script
def render_points(
        points: torch.Tensor,
        sigmas: torch.Tensor,
        exponents: torch.Tensor,
        intensities: torch.Tensor,
        camera_sigmas: torch.Tensor,
        camera_exponents: torch.Tensor,
        cameras_intensities: torch.Tensor,
        image_size: int = PREPARED_IMAGE_SIZE[0],
        render_mode: str = RENDER_MODE_GAUSSIANS
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

    # Reshape and scale gaussian exponents
    exponents = exponents.repeat_interleave(3, 0)[:, :, None, None]
    sfs = camera_exponents.reshape(bs * 3)[:, None, None, None]
    exponents = exponents * sfs

    # Reshape and scale intensities
    intensities = intensities.repeat_interleave(3, 0)[:, :, None, None]
    sfs = cameras_intensities.reshape(bs * 3)[:, None, None, None]
    intensities = intensities * sfs

    # Build x and y grids
    grid = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Centre the grids around the coordinates
    mmp2 = m - p2
    dst = mmp2[..., 0]**2 + mmp2[..., 1]**2

    if render_mode == 'gaussians':
        # Make (un-normalised) gaussian blobs
        blobs = torch.exp(-(dst / (2 * sigmas**2))**exponents)

        # The rendering is the maximum of the intensity-scaled overlapping blobs
        masks = (blobs * intensities).amax(dim=1)

    elif render_mode == 'circles':
        # Make circles with radii equal to the sigmas
        blobs = torch.ceil(torch.relu(sigmas - dst))

        # The rendering is the intersection of all circles
        masks = blobs.sum(dim=1)

    else:
        raise RuntimeError(f'Unknown rendering mode: "{render_mode}".')

    masks = masks.clamp(max=1.)
    masks = masks.reshape(bs, 3, image_size, image_size)

    # Also return the blobs for scoring points individually
    blobs = blobs.reshape(bs, 3, N, image_size, image_size)

    return masks, blobs


class ProjectRenderScoreModel(nn.Module):
    image_size: Final[int]

    def __init__(self, image_size: int = PREPARED_IMAGE_SIZE[0], render_mode: str = RENDER_MODE_GAUSSIANS):
        super().__init__()
        self.image_size = image_size
        assert render_mode in RENDER_MODES
        self.render_mode = render_mode
        self.cams = torch.jit.script(DynamicCameras())

    def forward(
            self,
            cam_coeffs: torch.Tensor,
            points: List[torch.Tensor],
            masks_target: List[torch.Tensor],
            sigmas: List[torch.Tensor],
            exponents: List[torch.Tensor],
            intensities: List[torch.Tensor],
            camera_sigmas: torch.Tensor,
            camera_exponents: torch.Tensor,
            camera_intensities: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
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
        exponents_smoothed = []
        intensities_smoothed = []

        # Run the parameters through the model at each scale to get the outputs
        for d in range(D):
            points_d = points[d]
            masks_target_d = masks_target[d]
            sigmas_d = sigmas[d]
            exponents_d = exponents[d]
            intensities_d = intensities[d]

            # Smooth the points, sigmas and intensities using average pooling convolutions.
            if d > 1:
                ks = int(2 * d + 1)
                pad_size = int(ks / 2)

                # Smooth the curve points
                cp = torch.cat([
                    torch.repeat_interleave(points_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    points_d,
                    torch.repeat_interleave(points_d[:, -1].unsqueeze(1), pad_size, dim=1)
                ], dim=1)
                cp = cp.permute(0, 2, 1)

                # Smooth with a gaussian kernel
                k_sig = ks / 5
                k = make_gaussian_kernel(k_sig, device=cp.device)
                k = torch.stack([k] * 3)[:, None, :]
                cps = F.conv1d(cp, weight=k, groups=3)

                # Average pooling smoothing
                # cps = F.avg_pool1d(cp, kernel_size=ks, stride=1, padding=0)

                points_d = cps.permute(0, 2, 1)

                # Smooth the sigmas
                # todo: this is the wrong padding - swap to repeated edge values
                sigs = torch.cat([
                    torch.repeat_interleave(sigmas_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    sigmas_d,
                    torch.repeat_interleave(sigmas_d[:, -1].unsqueeze(1), pad_size, dim=1),
                ], dim=1)
                sigs = sigs[:, None, :]
                sigs = F.avg_pool1d(sigs, kernel_size=ks, stride=1, padding=0)
                sigmas_d = sigs.squeeze(1)

                # Smooth the exponents
                exps = torch.cat([
                    torch.repeat_interleave(exponents_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    exponents_d,
                    torch.repeat_interleave(exponents_d[:, -1].unsqueeze(1), pad_size, dim=1),
                ], dim=1)
                exps = exps[:, None, :]
                exps = F.avg_pool1d(exps, kernel_size=ks, stride=1, padding=0)
                exponents_d = exps.squeeze(1)

                # Smooth the intensities
                ints = torch.cat([
                    torch.repeat_interleave(intensities_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    intensities_d,
                    torch.repeat_interleave(intensities_d[:, -1].unsqueeze(1), pad_size, dim=1),
                ], dim=1)
                ints = ints[:, None, :]
                ints = F.avg_pool1d(ints, kernel_size=ks, stride=1, padding=0)
                intensities_d = ints.squeeze(1)

            # Project and render
            points_2d_d = self._project_to_2d(cam_coeffs, points_d, points_3d_base, points_2d_base)
            masks_d, blobs = render_points(
                points_2d_d,
                sigmas_d,
                exponents_d,
                intensities_d,
                camera_sigmas,
                camera_exponents,
                camera_intensities,
                self.image_size,
                self.render_mode
            )

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
            exponents_smoothed.append(exponents_d)
            intensities_smoothed.append(intensities_d)

        return masks, points_2d, scores, points_smoothed, sigmas_smoothed, exponents_smoothed, intensities_smoothed

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
