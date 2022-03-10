from typing import Tuple, List, Final

import torch
import torch.nn.functional as F
from torch import nn

from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.data.model.mf_parameters import RENDER_MODE_GAUSSIANS, RENDER_MODES
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras
from wormlab3d.midlines3d.mf_methods import calculate_curvature


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
    yv, xv = torch.meshgrid([grid, grid], indexing='ij')
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


@torch.jit.script
def _smooth_parameter(param: torch.Tensor, ks: int, mode: str = 'avg') -> torch.Tensor:
    """
    Apply 1D average pooling to smooth a parameter vector.
    """
    pad_size = int(ks / 2)
    p_smoothed = torch.cat([
        torch.repeat_interleave(param[:, 0].unsqueeze(1), pad_size, dim=1),
        param,
        torch.repeat_interleave(param[:, -1].unsqueeze(1), pad_size, dim=1),
    ], dim=1)

    # Average pooling smoothing
    if mode == 'avg':
        p_smoothed = p_smoothed[:, None, :]
        p_smoothed = F.avg_pool1d(p_smoothed, kernel_size=ks, stride=1, padding=0)
        p_smoothed = p_smoothed.squeeze(1)

    # Smooth with a gaussian kernel
    elif mode == 'gaussian':
        k_sig = ks / 5
        k = make_gaussian_kernel(k_sig, device=param.device)
        k = torch.stack([k] * 3)[:, None, :]
        p_smoothed = p_smoothed.permute(0, 2, 1)
        p_smoothed = F.conv1d(p_smoothed, weight=k, groups=3)
        p_smoothed = p_smoothed.permute(0, 2, 1)

    else:
        raise RuntimeError(f'Unknown smoothing mode: "{mode}".')

    return p_smoothed


@torch.jit.script
def _taper_parameter(param: torch.Tensor) -> torch.Tensor:
    """
    Ensure the parameter vector decreases from the middle-out.
    """
    N = param.shape[1]
    mp = int(N / 2)
    tapered = torch.zeros_like(param) + 1e-5
    tapered[:, mp - 1:mp + 1] = param[:, mp - 1:mp + 1]

    # Middle to head
    for i in range(mp - 2, -1, -1):
        tapered[:, i] = torch.amin(
            torch.stack([param[:, i], tapered[:, i + 1].clone()], dim=1),
            dim=1
        )

    # Middle to tail
    for i in range(mp + 1, N):
        tapered[:, i] = torch.amin(
            torch.stack([param[:, i], tapered[:, i - 1].clone()], dim=1),
            dim=1
        )

    return tapered


class ProjectRenderScoreModel(nn.Module):
    image_size: Final[int]

    def __init__(
            self,
            image_size: int = PREPARED_IMAGE_SIZE[0],
            render_mode: str = RENDER_MODE_GAUSSIANS,
            curvature_mode: bool = False
    ):
        super().__init__()
        self.image_size = image_size
        assert render_mode in RENDER_MODES
        self.render_mode = render_mode
        self.curvature_mode = curvature_mode
        self.cams = torch.jit.script(DynamicCameras())

    def forward(
            self,
            cam_coeffs: torch.Tensor,
            points: List[torch.Tensor],
            curvatures: List[torch.Tensor],
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
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor]
    ]:
        """
        Project the 3D points to 2D, render the projected points as blobs on an image and score each point for overlap.
        """
        D = len(points)
        blobs = []
        masks = []
        detection_masks = []
        points_2d = []
        scores = []
        points_smoothed = []
        curvatures_smoothed = []
        sigmas_smoothed = []
        exponents_smoothed = []
        intensities_smoothed = []

        # Run the parameters through the model at each scale to get the outputs
        for d in range(D):
            points_d = points[d]
            curvatures_d = curvatures[d]
            # masks_target_d = masks_target[d]
            sigmas_d = sigmas[d]
            exponents_d = exponents[d]
            intensities_d = intensities[d]

            N = points_d.shape[1]
            depth = int(torch.log2(torch.tensor(N)))

            # Smooth the points, sigmas and intensities using average pooling convolutions.
            if depth > 1:
                ks = int(2 * depth + 1)

                # Integrate curvature to form midline points.
                if self.curvature_mode:
                    X0 = curvatures_d[:, 0]
                    T0 = curvatures_d[:, 1]
                    h = torch.norm(T0, dim=-1)
                    K = torch.zeros_like(curvatures_d)
                    K[:, 1:-1] = curvatures_d[:, 2:]
                    K = _smooth_parameter(K, ks, mode='gaussian')
                    Kv = (K[:, 1:] + K[:, :-1]) / 2
                    T = torch.cat([T0[:, None], Kv], dim=1).cumsum(dim=1)
                    T = T / torch.norm(T, dim=-1, keepdim=True)
                    Tv = h * (T[:, 1:] + T[:, :-1]) / 2
                    points_d = torch.cat([X0[:, None], Tv], dim=1).cumsum(dim=1)
                    curvatures_d = K

                else:
                    # Distance to parent points fixed as a quarter of the mean segment-length between parents.
                    if d > 0:
                        parents = points_smoothed[d - 1]
                        x = torch.mean(torch.norm(parents[:, 1:] - parents[:, :-1], dim=-1)) / 4
                        parents_repeated = torch.repeat_interleave(parents, repeats=2, dim=1)
                        direction = points_d - parents_repeated
                        points_anchored = parents_repeated + x * (
                                direction / torch.norm(direction, dim=-1, keepdim=True))
                        points_d = torch.cat([
                            points_d[:, 0][:, None, :],
                            points_anchored[:, 1:-1],
                            points_d[:, -1][:, None, :]
                        ], dim=1)
                    points_d = _smooth_parameter(points_d, ks, mode='gaussian')
                    curvatures_d = calculate_curvature(points_d)

                    # # Collapse worm where the curvature is too great
                    # k = calculate_scalar_curvature(points_d)
                    # sl = torch.norm(points_d[:, 1:] - points_d[:, :-1], dim=-1)
                    # wl = sl.sum(dim=-1, keepdim=True)
                    # kinks = k > (2 * 2 * torch.pi) / wl
                    # intensities_d[kinks] = 0.

                # Ensure tapering of rendered worm to avoid holes
                sigmas_d = _taper_parameter(sigmas_d)
                intensities_d = _taper_parameter(intensities_d)

                # Smooth the sigmas, exponents and intensities
                sigmas_d = _smooth_parameter(sigmas_d, ks)
                exponents_d = _smooth_parameter(exponents_d, ks)
                intensities_d = _smooth_parameter(intensities_d, ks)

            else:
                curvatures_d = torch.zeros_like(points_d)

            # Project and render
            points_2d_d = self._project_to_2d(cam_coeffs, points_d, points_3d_base, points_2d_base)
            masks_d, blobs_d = render_points(
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

            # # Normalise blobs
            # sum_ = blobs_d.amax(dim=(2, 3), keepdim=True)
            # sum_ = sum_.clamp(min=1e-8)
            # blobs_normed = blobs_d / sum_
            #
            # # Score the points
            # scores_d = (blobs_normed * masks_target_d.unsqueeze(2)).sum(dim=(3, 4)).mean(dim=1)
            # if d > 1:
            #     scores_d = self._taper_parameter(scores_d)
            #
            #     # Make new render with blobs scaled by relative scores to get detection masks
            #     max_score = scores_d.amax(keepdim=True)
            #     if max_score.item() > 0:
            #         rel_scores = scores_d / max_score
            #         sf = rel_scores[:, None, :, None, None]
            #         detection_masks_d = (blobs_d * sf).amax(dim=2)
            #     else:
            #         detection_masks_d = masks_d.clone()
            # else:
            #     detection_masks_d = masks_d.clone()

            blobs.append(blobs_d)
            masks.append(masks_d)
            points_2d.append(points_2d_d.transpose(1, 2))
            points_smoothed.append(points_d)
            curvatures_smoothed.append(curvatures_d)
            sigmas_smoothed.append(sigmas_d)
            exponents_smoothed.append(exponents_d)
            intensities_smoothed.append(intensities_d)

        # Scores need to feed back up the chain to inform parent points
        for d in range(D - 1, -1, -1):
            blobs_d = blobs[d]
            masks_d = masks[d]
            masks_target_d = masks_target[d]

            # Normalise blobs
            sum_ = blobs_d.amax(dim=(2, 3), keepdim=True)
            sum_ = sum_.clamp(min=1e-8)
            blobs_normed = blobs_d / sum_

            # Score the points - take lowest score from all projections
            scores_d = (blobs_normed * masks_target_d.unsqueeze(2)).sum(dim=(3, 4)).amin(dim=1)
            if d > 1:
                scores_d = _taper_parameter(scores_d)

            # Parent points can only score the minimum of their child points
            if d < D - 1:
                scores_children = torch.amin(scores[-1].reshape((scores_d.shape[0], scores_d.shape[1], 2)), dim=-1)
                scores_d = torch.amin(torch.stack([scores_d, scores_children]), dim=0)

            if d > 1:
                # Make new render with blobs scaled by relative scores to get detection masks
                max_score = scores_d.amax(keepdim=True)
                if max_score.item() > 0:
                    rel_scores = scores_d / max_score
                    sf = rel_scores[:, None, :, None, None]
                    detection_masks_d = (blobs_d * sf).amax(dim=2)
                else:
                    detection_masks_d = masks_d.clone()
            else:
                detection_masks_d = masks_d.clone()

            scores.append(scores_d)
            detection_masks.append(detection_masks_d)
        scores = scores[::-1]
        detection_masks = detection_masks[::-1]

        return masks, detection_masks, points_2d, scores, points_smoothed, curvatures_smoothed, sigmas_smoothed, exponents_smoothed, intensities_smoothed

    def _project_to_2d(
            self,
            cam_coeffs: torch.Tensor,
            points_3d: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ) -> torch.Tensor:
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
