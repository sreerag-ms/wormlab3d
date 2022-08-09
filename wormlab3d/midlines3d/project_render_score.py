from typing import Tuple, List, Final, Optional

import torch
import torch.nn.functional as F
from torch import nn

from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT
from wormlab3d.data.model.mf_parameters import RENDER_MODE_GAUSSIANS, RENDER_MODES
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras
from wormlab3d.midlines3d.mf_methods import calculate_curvature, integrate_curvature, normalise, smooth_parameter


@torch.jit.script
def _apply_filters(blobs: torch.Tensor, filters: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply filters.
    """
    if filters is not None:
        bs = len(filters)
        blobs = blobs.reshape(bs, 3, blobs.shape[1], blobs.shape[2], blobs.shape[3])
        blobs = blobs.permute(0, 2, 1, 3, 4)
        blobs_filtered = []
        for i in range(bs):
            filters_i = filters[i].unsqueeze(1)
            blobs_i_filtered = F.conv2d(blobs[i], filters_i, padding='same', groups=3)
            blobs_i_filtered = blobs_i_filtered.clamp(min=0., max=1.)
            blobs_filtered.append(blobs_i_filtered)
        blobs_filtered = torch.stack(blobs_filtered, dim=0)
        blobs_filtered = blobs_filtered.permute(0, 2, 1, 3, 4)
        blobs_filtered = blobs_filtered.reshape(bs * 3, blobs_filtered.shape[2], blobs_filtered.shape[3],
                                                blobs_filtered.shape[4])
    else:
        blobs_filtered = blobs.clone()

    return blobs_filtered


@torch.jit.script
def render_points(
        points: torch.Tensor,
        sigmas: torch.Tensor,
        exponents: torch.Tensor,
        intensities: torch.Tensor,
        camera_sigmas: torch.Tensor,
        camera_exponents: torch.Tensor,
        cameras_intensities: torch.Tensor,
        image_size: int = PREPARED_IMAGE_SIZE_DEFAULT,
        render_mode: str = RENDER_MODE_GAUSSIANS,
        filters: Optional[torch.Tensor] = None
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

        # Apply filters
        blobs = _apply_filters(blobs, filters)

        # The rendering is the maximum of the intensity-scaled overlapping blobs
        masks = (blobs * intensities).amax(dim=1)

    elif render_mode == 'circles':
        # Make circles with radii equal to the sigmas
        blobs = torch.ceil(torch.relu(sigmas - dst))

        # Apply filters
        blobs = _apply_filters(blobs, filters)

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


@torch.jit.script
def _normalise_scale_factor(v: torch.Tensor) -> torch.Tensor:
    """
    Camera scaling factors should average 1 and not be more than 30% from the mean.
    """
    v = v / v.mean()
    sf = 0.3 / ((v - 1).abs()).amax()
    if sf < 1:
        v = (v - 1) * sf + 1
    return v


class ProjectRenderScoreModel(nn.Module):
    image_size: Final[int]

    def __init__(
            self,
            image_size: int = PREPARED_IMAGE_SIZE_DEFAULT,
            render_mode: str = RENDER_MODE_GAUSSIANS,
            second_render_prob: float = 0.5,
            filter_size: int = None,
            sigmas_min: float = 0.04,
            sigmas_max: float = 0.1,
            exponents_min: float = 0.5,
            exponents_max: float = 10.,
            intensities_min: float = 0.4,
            intensities_max: float = 10.,
            curvature_mode: bool = False,
            curvature_deltas: bool = False,
            length_min: float = 0.5,
            length_max: float = 2,
            curvature_max: float = 2.,
            dX0_limit: float = None,
            dl_limit: float = None,
            dk_limit: float = None,
            dpsi_limit: float = None,
            clamp_X0: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        assert render_mode in RENDER_MODES
        self.render_mode = render_mode
        self.second_render_prob = second_render_prob
        self.filter_size = filter_size
        self.sigmas_min = sigmas_min + 0.001
        self.sigmas_max = sigmas_max
        self.exponents_min = exponents_min
        self.exponents_max = exponents_max
        self.intensities_min = intensities_min + 0.01
        self.intensities_max = intensities_max
        self.curvature_mode = curvature_mode
        self.curvature_deltas = curvature_deltas
        self.length_min = length_min
        self.length_max = length_max
        self.curvature_max = curvature_max
        self.clamp_X0 = clamp_X0

        # The limits don't matter when not in delta-mode, but they need to be not-None for jit.
        if not curvature_deltas:
            dX0_limit = 0.
            dl_limit = 0.
            dk_limit = 0.
            dpsi_limit = 0.

        self.dX0_limit = dX0_limit
        self.dl_limit = dl_limit
        self.dk_limit = dk_limit
        self.dpsi_limit = dpsi_limit
        self.cams = torch.jit.script(DynamicCameras())

    def forward(
            self,
            cam_coeffs: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
            X0: List[torch.Tensor],
            T0: List[torch.Tensor],
            M10: List[torch.Tensor],
            length: List[torch.Tensor],
            curvatures: List[torch.Tensor],
            points: List[torch.Tensor],
            masks_target: List[torch.Tensor],
            sigmas: List[torch.Tensor],
            exponents: List[torch.Tensor],
            intensities: List[torch.Tensor],
            camera_sigmas: torch.Tensor,
            camera_exponents: torch.Tensor,
            camera_intensities: torch.Tensor,
            filters: torch.Tensor,
            length_warmup: bool
    ) -> Tuple[
        List[torch.Tensor],
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
        eps = 1e-6
        device = points[0].device
        D = len(points)
        bs = points[0].shape[0]
        blobs = []
        masks = []
        detection_masks = []
        points_raw = []
        points_2d = []
        scores = []
        points_smoothed = []
        curvatures_smoothed = []
        sigmas_smoothed = []
        exponents_smoothed = []
        intensities_smoothed = []

        # Camera scaling factors should average 1 and not be more than 20% from the mean
        camera_sigmas = _normalise_scale_factor(camera_sigmas)
        camera_exponents = _normalise_scale_factor(camera_exponents)
        camera_intensities = _normalise_scale_factor(camera_intensities)

        # Run the parameters through the model at each scale to get the outputs
        for d in range(D):
            points_d = points[d]

            # Clamp the sigmas, exponents and intensities
            sigmas_d = sigmas[d].clamp(min=self.sigmas_min, max=self.sigmas_max)
            exponents_d = exponents[d].clamp(min=self.exponents_min, max=self.exponents_max)
            intensities_d = intensities[d].clamp(min=self.intensities_min, max=self.intensities_max)

            N = points_d.shape[1]
            depth = int(torch.log2(torch.tensor(N)))

            # Smooth the points, sigmas and intensities using average pooling convolutions.
            if depth > 1:
                ks = int(2 * depth + 1)

                # Integrate curvature to form midline points.
                if self.curvature_mode:
                    if self.curvature_deltas:
                        dX0 = X0[d]
                        X0_d = torch.zeros_like(dX0)
                        X0_d[0] = dX0[0]

                        dT0 = T0[d]
                        T0_d = torch.zeros_like(dT0)
                        T0_d[0] = normalise(dT0[0])

                        dM10 = M10[d]
                        M10_d = torch.zeros_like(dM10)
                        M10_d[0] = normalise(dM10[0])

                        dl = length[d]
                        length_d = torch.zeros_like(dl)
                        length_d[0] = dl[0]

                        dK = curvatures[d]
                        curvatures_d = torch.zeros_like(dK)
                        curvatures_d[0] = dK[0]
                        k = torch.zeros(bs, N, device=device)
                        k[0] = torch.norm(curvatures_d[0].clone(), dim=-1)
                        psi = torch.zeros(bs, N, device=device)
                        psi[0] = torch.atan2(
                            curvatures_d[0, :, 0].clone(),
                            curvatures_d[0, :, 1].clone() + eps
                        )

                        for i in range(1, bs):
                            # Limit X0 change
                            dX0i = dX0[i]
                            dX0i_size = torch.norm(dX0i)
                            if self.dX0_limit is not None and dX0i_size > self.dX0_limit:
                                dX0i = dX0i / dX0i_size * self.dX0_limit
                            X0_d[i] = X0_d[i - 1] + dX0i

                            # T0 always has length 1
                            T0_d[i] = normalise(T0_d[i - 1] + dT0[i])

                            # M10 always has length 1
                            M10_d[i] = normalise(M10_d[i - 1] + dM10[i])

                            # Limit length change
                            if length_warmup:
                                length_d[i] = length_d[i - 1]
                            else:
                                length_d[i] = length_d[i - 1] + dl[i].clamp(min=-self.dl_limit, max=self.dl_limit)

                            # Limit curvature magnitude changes
                            dk = dK[i, :, 0].clone()
                            if self.dk_limit is not None:
                                dk_lim = self.dk_limit / (N - 1)
                                dk = dk.clamp(min=-dk_lim, max=dk_lim)
                            k[i] = k[i - 1].clone() + dk

                            # Limit curvature angle changes
                            dpsi = dK[i, :, 1].clone()
                            if self.dpsi_limit is not None:
                                dpsi = dpsi.clamp(min=-self.dpsi_limit, max=self.dpsi_limit)
                            psi[i] = psi[i - 1].clone() + dpsi

                            # Calculate new curvature
                            curvatures_d[i] = torch.stack([
                                k[i].clone() * torch.sin(psi[i].clone()),
                                k[i].clone() * torch.cos(psi[i].clone())
                            ], dim=-1)
                    else:
                        X0_d = X0[d]
                        T0_d = T0[d]
                        M10_d = M10[d]
                        length_d = length[d]
                        curvatures_d = curvatures[d]

                    # Add some noise..?
                    if 0:
                        X0_d = X0_d + torch.randn_like(X0_d) * 0.001
                        T0_d = T0_d + torch.randn_like(T0_d) * 0.002
                        M10_d = M10_d + torch.randn_like(M10_d) * 0.001
                        length_d = length_d + torch.randn_like(length_d) * 0.0005
                        curvatures_d = curvatures_d + torch.randn_like(curvatures_d) * 0.001

                    # Ensure that the worm does not get too long/short.
                    if not length_warmup:
                        length_d = length_d.clamp(min=self.length_min, max=self.length_max)

                    # Smooth the curvatures
                    curvatures_d = smooth_parameter(curvatures_d, ks, mode='gaussian')

                    # Ensure curvature doesn't get too large
                    k = torch.norm(curvatures_d, dim=-1)
                    k_max = self.curvature_max * 2 * torch.pi / (N - 1)
                    curvatures_d = torch.where(
                        (k > k_max)[..., None],
                        curvatures_d * (k_max / (k + eps))[..., None],
                        curvatures_d
                    )

                    # Keep X0 somewhere in the frame
                    if self.clamp_X0:
                        X0_d = X0_d.clamp(min=-0.5, max=0.5)

                    # Integrate the curvature to get the midline coordinates
                    Xc_d, Tc_d, M1c_d = integrate_curvature(X0_d, T0_d, length_d, curvatures_d, M10_d)

                    # Rebuild it again from the head and the tail
                    Xh_d, Th_d, M1h_d = integrate_curvature(Xc_d[:, 1], Tc_d[:, 1], length_d, curvatures_d, M1c_d[:, 1], start_idx=1)
                    Xt_d, Tt_d, M1t_d = integrate_curvature(Xc_d[:, -2], Tc_d[:, -2], length_d, curvatures_d, M1c_d[:, -2], start_idx=N - 2)

                    # Use the average of the head and tail curves
                    points_d = (Xh_d + Xt_d) / 2

                    # Log centre, head and tail curves
                    points_raw_d = torch.stack([Xc_d, Xh_d, Xt_d], dim=1)

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
                    points_d = smooth_parameter(points_d, ks, mode='gaussian')
                    points_raw_d = torch.zeros_like(points_d)
                    curvatures_d = calculate_curvature(points_d)

                # Prepare sigmas, exponents and intensities
                N5 = int(N / 5)

                # Sigmas should be equal in the middle section but taper towards the ends
                sigma_d = sigmas_d[:, None].clamp(min=self.sigmas_min)
                slopes = (sigma_d - self.sigmas_min) / N5 * torch.arange(N5, device=device)[None, :] + self.sigmas_min
                sigmas_d = torch.cat([
                    slopes,
                    torch.ones(bs, N - 2 * N5, device=device) * sigma_d,
                    slopes.flip(dims=(1,))
                ], dim=1)

                # Make exponents equal everywhere
                exponents_d = torch.ones(bs, N, device=device) * exponents_d[:, None]

                # Intensities should be equal in the middle section but taper towards the ends
                int_d = intensities_d[:, None]
                slopes = (int_d - self.intensities_min) / N5 \
                         * torch.arange(N5, device=device)[None, :] + self.intensities_min
                intensities_d = torch.cat([
                    slopes,
                    torch.ones(bs, N - 2 * N5, device=device) * int_d,
                    slopes.flip(dims=(1,))
                ], dim=1)

            else:
                points_raw_d = torch.zeros_like(points_d)
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
                self.render_mode,
                filters if self.filter_size is not None else None
            )

            blobs.append(blobs_d)
            masks.append(masks_d)
            points_raw.append(points_raw_d)
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
            N = blobs_d.shape[2]

            # Normalise blobs
            sum_ = blobs_d.amax(dim=(3, 4), keepdim=True)
            sum_ = sum_.clamp(min=1e-8)
            blobs_normed = blobs_d / sum_

            # Score the points - look at projections in each view and check how well each blob matches against the lowest intensity image
            scores_d = (blobs_normed * masks_target_d.unsqueeze(2)).sum(dim=(3, 4)).amin(dim=1) \
                       / intensities_smoothed[d].detach() \
                       / sigmas_smoothed[d].detach()  # Scale scores by sigmas and intensities
            scores_d_untapered = scores_d.clone()
            if N > 2:
                scores_d = _taper_parameter(scores_d)
            max_score = scores_d.amax(dim=1, keepdim=True)

            # Parent points can only score the minimum of their child points
            if d < D - 1:
                scores_children = torch.amin(scores[-1].reshape((scores_d.shape[0], scores_d.shape[1], 2)), dim=-1)
                scores_d = torch.amin(torch.stack([scores_d, scores_children]), dim=0)

            if scores_d.shape[1] > 1:
                # Make new render with blobs scaled by relative scores to get detection masks
                rel_scores = torch.where(max_score > 0, scores_d / max_score, torch.ones_like(scores_d))
                sf = rel_scores[:, None, :, None, None]
                dmd = (blobs_normed * sf).amax(dim=2)
                dmd[dmd > 0.1] = 1
                dmd[dmd < 0.1] = 0.2

                # Generate head and tail detection regions
                head_blobs = blobs_normed[:, :, 0].clone()
                tail_blobs = blobs_normed[:, :, -1].clone()
                head_blobs[head_blobs > 0.001] = 1
                tail_blobs[tail_blobs > 0.001] = 1

                # Add head and tail booster regions only if no gaps detected
                if 0:
                    dmd = dmd + torch.where(rel_scores[:, 0][:, None, None, None] > 0.1, head_blobs,
                                            torch.zeros_like(head_blobs))
                    dmd = dmd + torch.where(rel_scores[:, -1][:, None, None, None] > 0.1, tail_blobs,
                                            torch.zeros_like(tail_blobs))

                elif 1:
                    scores_rel_tapered = (scores_d - scores_d_untapered).abs() \
                                         / (torch.amax(torch.stack([scores_d, scores_d_untapered], dim=2), dim=2) + eps)
                    head_consistent = (scores_rel_tapered[:, 0] < 0.2) & (scores_d[:, 0] > 250)
                    tail_consistent = (scores_rel_tapered[:, -1] < 0.2) & (scores_d[:, -1] > 250)
                    dmd = dmd + torch.where(head_consistent[:, None, None, None], head_blobs, torch.zeros_like(head_blobs))
                    dmd = dmd + torch.where(tail_consistent[:, None, None, None], tail_blobs, torch.zeros_like(tail_blobs))

                # Add head and tail booster regions regardless
                else:
                    dmd = dmd + head_blobs + tail_blobs
                dmd = dmd.clamp(max=1.)
                detection_masks_d = dmd

                # The second rendering is the maximum of the intensity-and-score-scaled overlapping blobs
                if torch.rand(1)[0] < self.second_render_prob:
                    masks2 = (blobs_d * intensities_smoothed[d][:, None, :, None, None] * sf).amax(dim=2)
                    masks[d] = masks2
            else:
                detection_masks_d = masks_d.clone()

            scores.append(scores_d)
            detection_masks.append(detection_masks_d)
        scores = scores[::-1]
        detection_masks = detection_masks[::-1]

        return masks, detection_masks, points_raw, points_2d, scores, curvatures_smoothed, points_smoothed, \
               sigmas_smoothed, exponents_smoothed, intensities_smoothed

    def _project_to_2d(
            self,
            cam_coeffs: torch.Tensor,
            points_3d: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
            clamp: bool = True
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

        # Ensure points land inside image boundaries otherwise gradient breaks
        if clamp:
            points_2d = points_2d.clamp(min=1., max=self.image_size - 1)

        return points_2d
