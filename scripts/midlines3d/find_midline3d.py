import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import RMSprop, SGD, LBFGS, AdamW, Adam
from torch.optim.lr_scheduler import CyclicLR
from torchvision.transforms.functional import gaussian_blur
from wormlab3d.toolkit.plot_utils import interactive_plots
from wormlab3d import logger, CAMERA_IDXS, N_WORM_POINTS, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import SegmentationMasks, Midline3D
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_DELTA_ANGLES_BASIS, MAX_DECAY_FACTOR, ENCODING_MODE_POINTS
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras
from wormlab3d.midlines3d.points_to_masks import PointsToMasks
from wormlab3d.toolkit.util import is_bad, to_numpy
from wormlab3d import logger, LOGS_PATH


if 1 and torch.cuda.is_available():
    logger.info('USING GPU!')
    device = 'cuda'
else:
    logger.info('USING CPU')
    device = 'cpu'

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')

VOL_BOUNDS = (-0.4, 0.4)
VOL_SIZE = (11, 11, 11)

cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'



class CloudToCurve(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            curve_points_3d: torch.Tensor,
            cloud_points_3d: torch.Tensor,
            cloud_points_scores: torch.Tensor,
            blur_sigma: torch.Tensor,
            decay_factor: torch.Tensor
    ) -> torch.Tensor:
        # Make volume from the point cloud
        vol_cloud = _points_3d_to_volume(cloud_points_3d, cloud_points_scores, blur_sigma)

        # Score the curve points against the point cloud volume
        curve_idxs = _get_idxs_3d(curve_points_3d)
        points_scores = vol_cloud[[*curve_idxs]].reshape(curve_points_3d.shape[:-1])

        ctx.save_for_backward(
            curve_points_3d,
            vol_cloud,
            curve_idxs,
            points_scores,
            torch.tensor(blur_sigma),
            decay_factor
        )

        return points_scores

    @staticmethod
    def backward(ctx: Any, points_scores_grad: torch.Tensor) -> torch.Tensor:
        curve_points_3d, vol_cloud, curve_idxs, points_scores, blur_sigma, decay_factor = ctx.saved_tensors
        n_curve_points = curve_points_3d.shape[-2]
        decay = torch.exp(-torch.arange(n_curve_points, device=curve_points_3d.device) / n_curve_points * decay_factor)

        # Make volume from the curve points
        vol_curve = _points_3d_to_volume(curve_points_3d, blur_sigma=blur_sigma, decay_factor=decay_factor)
        vol_diff = vol_cloud - vol_curve

        # Calculate the gradient surface approximations
        J_diff = _calculate_gradient_surface_3d(vol_diff)
        J_target = _calculate_gradient_surface_3d(vol_cloud)

        # Get the directional gradients both towards the target and to spread the points out
        grads_diff = _get_directional_gradients_3d(J_diff, curve_idxs, n_curve_points)
        grads_target = _get_directional_gradients_3d(J_target, curve_idxs, n_curve_points)

        # Gradients are combination of towards-the-target when far away and cover-up-remainder when close
        points_scores = points_scores.unsqueeze(-1)
        points_3d_grad = grads_diff * points_scores + grads_target * (1 - points_scores)
        points_3d_grad = -points_3d_grad * decay.reshape(1, n_curve_points, 1)

        assert not is_bad(points_3d_grad)

        return points_3d_grad, None, None, None, None


class PointsToMasks(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            points_2d: torch.Tensor,
            blur_sigma: torch.Tensor,
            masks_target: torch.Tensor
    ) -> torch.Tensor:
        # Generate masks from 2D points
        masks = _points_to_masks(points_2d, blur_sigma)

        # Check how close each 2D point is to the target mask
        idxs = _get_idxs_2d(points_2d)
        points_scores = masks_target[[*idxs]].reshape(points_2d.shape[:-1])

        ctx.save_for_backward(points_2d, masks, masks_target, idxs, points_scores)

        return masks, points_scores

    @staticmethod
    def backward(ctx: Any, masks_grad: torch.Tensor, points_scores_grad: torch.Tensor) -> torch.Tensor:
        points_2d, masks_out, masks_target, idxs, points_scores = ctx.saved_tensors
        n_points = points_2d.shape[2]

        # Balance the overall masks output
        ratio = masks_target.sum(dim=(2,3), keepdim=True)/masks_out.sum(dim=(2,3), keepdim=True)
        masks_out = masks_out * ratio
        masks_diff = masks_target - masks_out

        # Calculate the gradient surface approximations
        J_diff = _calculate_gradient_surface_2d(masks_diff)
        J_target = _calculate_gradient_surface_2d(masks_target)

        # Get the directional gradients both towards the target and to spread the points out
        grads_diff = _get_directional_gradients_2d(J_diff, idxs, n_points)
        grads_target = _get_directional_gradients_2d(J_target, idxs, n_points)

        # grads_noise = torch.normal(0, std=torch.ones_like(grads_diff)/grads_diff.mean())
        # grads_diff = grads_diff + grads_noise
        # grads_noise = torch.normal(0, std=torch.stack([grads_target.norm(dim=-1), grads_target.norm(dim=-1)], dim=-1))
        # grads_diff = grads_noise * 100

        # How well is the target covered in each view?

        # grads_noise = torch.normal(0, std=torch.ones_like(grads_target))
        # coverage = torch.sqrt((masks_diff**2).mean(dim=(2,3)))
        # coverage = coverage.reshape(grads_target.shape[0], 3, 1, 1)
        # grads_diff = grads_noise * coverage

        # # Normalise the grads
        # nf = grads_target.norm() / grads_diff.norm()
        # grads_diff = grads_diff * nf

        # Gradients are combination of towards-the-target when far away and cover-up-remainder when close
        points_scores = points_scores.unsqueeze(-1)
        mean_scores = points_scores.mean(dim=1, keepdim=True) #.unsqueeze(-1)
        # max_scores = points_scores.max(dim=1, keepdim=True)[0]
        min_scores = points_scores.min(dim=1, keepdim=True)[0]
        # sf_max = (points_scores == max_scores).to(torch.float32)
        sf_min = (points_scores == min_scores).to(torch.float32)
        # grads_diff = grads_diff * sf_max  # only move towards diff in best views
        # grads_diff = grads_diff * (1-sf_min)  # only move towards diff in best views
        # grads_target = grads_target * sf_min  # only move towards target in worst views

        points_2d_grad = grads_diff * points_scores + grads_target * (1 - points_scores)
        # points_2d_grad = grads_diff * mean_scores + grads_target * (1 - mean_scores)
        # points_2d_grad = grads_diff + grads_target
        # points_2d_grad = grads_target
        # points_2d_grad = -points_2d_grad

        # Points scoring low => far away from target => take bigger steps
        # points_2d_grad = (-points_2d_grad) * (-torch.log(points_scores+1e-10))
        gs = points_2d_grad.abs().sum(dim=-1, keepdim=True)
        points_2d_grad = (-points_2d_grad) * (-torch.log(gs+1e-10))

        # if a point has a high score in one view and very low score in another
        # then it is difficult to resolve properly
        # maybe first we need to ignore high scores and focus solely on the low ones
        # ie, only move each point in the view in which it has the lowest score
        # ie, set grads to zero in the other views

        points_2d_grad = torch.where(
            mean_scores < 0.7,
            points_2d_grad * sf_min,
            points_2d_grad  #* sf_max,
        )
        # points_2d_grad = points_2d_grad * sf_min

        assert not is_bad(points_2d_grad)

        return points_2d_grad, None, None, None, None


def _get_idxs_2d(points_2d: torch.Tensor):
    bs = points_2d.shape[0]
    n_points = points_2d.shape[2]
    device = points_2d.device
    point_idxs_2d = points_2d.round().to(torch.long)  # <-- this operation is non-differentiable!
    point_idxs_2d = point_idxs_2d.clamp(min=0, max=PREPARED_IMAGE_SIZE[0] - 1)
    idxs = [
        torch.arange(bs, device=device).repeat_interleave(3 * n_points),
        torch.arange(3, device=device).repeat_interleave(n_points).repeat(bs),
        point_idxs_2d[:, :, :, 0].flatten(),
        point_idxs_2d[:, :, :, 1].flatten(),
    ]
    idxs = torch.stack([ix for ix in idxs])
    return idxs


def _calculate_gradient_surface_2d(masks_target: torch.Tensor):
    # Calculate gradient surface
    grad0 = -masks_target
    grads = [grad0]
    g = grad0
    while g.shape[-1] > 1:
        g2 = _avg_pool_2d(g, oob_grad_val=0.1)
        grads.append(g2)
        g = g2

    # Got all the grad averages, now add them together and average
    grad_sum = torch.zeros_like(grad0)
    for i, g in enumerate(grads):
        grad_sum += F.interpolate(grads[i], PREPARED_IMAGE_SIZE, mode='bilinear', align_corners=False)
    grad_avg = grad_sum / len(grads)

    # Calculate directional gradients
    gapx = F.pad(grad_avg, (0, 0, 1, 1), mode='replicate')
    gx = (gapx[:, :, :-2] - gapx[:, :, 2:]) / 2
    gapy = F.pad(grad_avg, (1, 1, 0, 0), mode='replicate')
    gy = (gapy[:, :, :, :-2] - gapy[:, :, :, 2:]) / 2
    J = torch.stack([gx, gy])

    return J


def _avg_pool_2d(grad, oob_grad_val=0., mode='constant'):
    # Average pooling with overlap and boundary values
    padded_grad = F.pad(grad, (1, 1, 1, 1), mode=mode, value=oob_grad_val)
    ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


def _get_directional_gradients_2d(J: torch.Tensor, idxs: torch.Tensor, n_points: int):
    # Determine direction of minimum gradient from each sample coordinate
    coord_shape = J.shape[1], 3, n_points
    pixel_grads_x = J[0][[*idxs]].reshape(coord_shape)
    pixel_grads_y = J[1][[*idxs]].reshape(coord_shape)
    points_2d_grad = torch.stack([pixel_grads_x, pixel_grads_y], dim=-1)
    return points_2d_grad


def _get_idxs_3d(points_3d: torch.Tensor):
    bs = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    point_idxs_3d = ((points_3d - VOL_BOUNDS[0]) * VOL_SIZE[0]).round().to(torch.long)
    point_idxs_3d = point_idxs_3d.clamp(min=0, max=VOL_SIZE[0] - 1)
    idxs = [
        torch.arange(bs, device=device).repeat_interleave(n_points),
        point_idxs_3d[:, :, 0].flatten(),
        point_idxs_3d[:, :, 1].flatten(),
        point_idxs_3d[:, :, 2].flatten(),
    ]
    idxs = torch.stack([ix for ix in idxs])
    return idxs


def _calculate_gradient_surface_3d(vol_grad: torch.Tensor):
    # Calculate gradient surface
    grad0 = -vol_grad.unsqueeze(1)
    grads = [grad0]
    g = grad0
    while g.shape[-1] > 1:
        g2 = _avg_grad_3d(g, oob_grad_val=0.1)
        grads.append(g2)
        g = g2

    # Got all the grad averages, now add them together and average
    grad_sum = torch.zeros_like(grad0)
    for i, g in enumerate(grads):
        grad_sum += F.interpolate(grads[i], vol_grad.shape[-3:], mode='trilinear', align_corners=False)
    grad_avg = grad_sum / len(grads)

    # Calculate directional gradients
    gapx = F.pad(grad_avg, (0, 0, 0, 0, 1, 1), mode='replicate')
    gx = (gapx[:, :, :-2] - gapx[:, :, 2:]) / 2
    gapy = F.pad(grad_avg, (0, 0, 1, 1, 0, 0), mode='replicate')
    gy = (gapy[:, :, :, :-2] - gapy[:, :, :, 2:]) / 2
    gapz = F.pad(grad_avg, (1, 1, 0, 0, 0, 0), mode='replicate')
    gz = (gapz[:, :, :, :, :-2] - gapz[:, :, :, :, 2:]) / 2
    J = torch.stack([gx, gy, gz])
    J = J.squeeze(2)

    return J


def _avg_grad_3d(grad, oob_grad_val=0.):
    # Average pooling with overlap and boundary values
    padded_grad = F.pad(grad, (1, 1, 1, 1, 1, 1), mode='constant', value=oob_grad_val)
    ag = F.avg_pool3d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


def _get_directional_gradients_3d(J: torch.Tensor, idxs: torch.Tensor, n_points: int):
    # Determine direction of minimum gradient from each sample coordinate
    coord_shape = J.shape[1], n_points
    pixel_grads_x = J[0][[*idxs]].reshape(coord_shape)
    pixel_grads_y = J[1][[*idxs]].reshape(coord_shape)
    pixel_grads_z = J[2][[*idxs]].reshape(coord_shape)
    points_3d_grad = torch.stack([pixel_grads_x, pixel_grads_y, pixel_grads_z], dim=-1)
    return points_3d_grad


def _points_to_masks(points_2d: torch.Tensor, blur_sigma: float = 1, decay_factor: float = None) -> torch.Tensor:
    bs = points_2d.shape[0]
    idxs = _get_idxs_2d(points_2d)

    # Write ones at the indexed locations
    if decay_factor is None:
        mw = torch.ones(np.prod(points_2d.shape[:-1]), device=device)
    else:
        n_points = points_2d.shape[2]
        mw = torch.exp(-torch.arange(n_points) / n_points * decay_factor).repeat(bs * 3)
    ms = torch.sparse_coo_tensor(
        indices=idxs,
        values=mw,
        size=(bs, 3, *PREPARED_IMAGE_SIZE),
        device=device
    )
    masks = ms.to_dense()
    masks = masks.clamp(max=1)

    # Apply a gaussian blur to the masks
    if blur_sigma > 0:
        mask_maxs_original = torch.amax(masks, dim=(2, 3), keepdim=True)
        ks = int(blur_sigma * 5)
        if ks % 2 == 0:
            ks += 1
        masks = gaussian_blur(masks, kernel_size=ks, sigma=blur_sigma)
        mask_maxs = torch.amax(masks, dim=(2, 3), keepdim=True)
        masks = torch.where(
            mask_maxs > 0,
            masks / mask_maxs,
            torch.zeros_like(masks)
        )
        masks = masks * mask_maxs_original

    return masks


def render_points(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]
    device = points.device

    # Shift to [-1,+1]
    points = (points - 100) / 100

    # Reshape
    # points = points.reshape(bs, 3, 2, n_worm_points)
    # points = points.permute(0, 1, 3, 2)
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)

    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])

    # 1 x 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Centre points
    mmp2 = m - p2
    dst = mmp2[..., 0]**2+mmp2[..., 1]**2
    blobs = torch.exp(-(dst / (2 * blur_sigmas**2)[None, :, None, None]))

    # # Normalise blobs
    # sum_ = blobs.sum(dim=(2,3), keepdim=True)
    # sum_ = sum_.clamp(min=1e-8)
    # blobs_normed = blobs/sum_
    # blobs_normed = blobs

    # Mask is sum of the blobs
    masks = blobs.sum(dim=1)

    # Normalise
    masks = masks.clamp(max=1.)
    sum_ = masks.sum(dim=(1,2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks_normed = masks/sum_

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
    # points = points.reshape(bs, 3, 2, n_worm_points)
    # points = points.permute(0, 1, 3, 2)
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
    sig = (blur_sigmas[1:] + blur_sigmas[:-1]) / 2
    d_norm = torch.exp(-d / (sig**2)[None, :, None, None])

    # Normalise
    # max_d = d_norm.amax(dim=(2, 3), keepdim=True)
    # d_norm = torch.where(max_d <= 0, torch.zeros_like(d), d_norm / max_d)

    # d_norm = d_norm / torch.sum(d_norm, (2, 3), keepdim=True)
    d_norm = d_norm.sum(dim=1)
    d_norm = d_norm.clamp(max=1)

    # Normalise
    sum_ = d_norm.sum(dim=(1,2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    d_norm = (d_norm/sum_).reshape(bs, 3, *PREPARED_IMAGE_SIZE)

    # Reshape back to the triplets
    # d_norm = d_norm.reshape(bs, 3, *PREPARED_IMAGE_SIZE)

    return d_norm


def render_points_3d(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]
    device = points.device

    # Reshape
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)

    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])

    # 1 x 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Centre points
    mmp2 = m - p2
    dst = mmp2[..., 0]**2+mmp2[..., 1]**2
    blobs = torch.exp(-(dst / (2 * blur_sigmas**2)[None, :, None, None]))

    # # Normalise blobs
    # sum_ = blobs.sum(dim=(2,3), keepdim=True)
    # sum_ = sum_.clamp(min=1e-8)
    # blobs_normed = blobs/sum_
    # blobs_normed = blobs

    # Mask is sum of the blobs
    masks = blobs.sum(dim=1)

    # Normalise
    masks = masks.clamp(max=1.)
    sum_ = masks.sum(dim=(1,2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks_normed = masks/sum_

    # Reshape
    masks_normed = masks_normed.reshape(bs, 3, *PREPARED_IMAGE_SIZE)
    blobs = blobs.reshape(bs, 3, n_worm_points, *PREPARED_IMAGE_SIZE)

    return masks_normed, blobs

def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks, device=device)
    kernel = torch.exp(- (ts / sigma)**2 / 2)
    kernel /= kernel.max()

    return kernel


def _points_3d_to_volume(points_3d: torch.Tensor, cloud_points_scores=None, blur_sigma: float = 1,
                         decay_factor: float = None) -> torch.Tensor:
    bs = points_3d.shape[0]
    idxs = _get_idxs_3d(points_3d)
    if cloud_points_scores is None:
        weights = torch.ones(np.prod(points_3d.shape[:-1]), device=device)
    else:
        # Weight the points by the scores, but only take the top 10%, and only if > 0.5.
        # weights = cloud_points_scores.mean(dim=1).flatten()
        weights = cloud_points_scores.flatten()
        # min_weight = max(0.5, weights.sort(descending=True)[0][:int(len(weights) * 0.1)][-1])
        # min_weight = 0.5
        # weights[weights < min_weight] = 0

    if decay_factor is not None:
        n_points = points_3d.shape[1]
        decay = torch.exp(-torch.arange(n_points, device=device) / n_points * decay_factor).repeat(bs)
        weights = weights * decay

    # Write ones at the indexed locations
    ms = torch.sparse_coo_tensor(
        indices=idxs,
        values=weights,
        size=(bs, *VOL_SIZE),
        device=device
    )
    vol = ms.to_dense()
    vol = vol.clamp(max=1)
    if vol.max() > 0:
        vol = vol / vol.max()

    # Apply a gaussian blur to the volume
    if blur_sigma > 0:
        k = make_gaussian_kernel(blur_sigma)
        k1d = k.reshape(1, 1, len(k), 1, 1)
        vol_in = vol.unsqueeze(1)
        for i in range(3):
            vol_in = vol_in.permute(0, 1, 4, 2, 3)
            vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
        vol = vol_in.squeeze(1)

    if vol.max() > 0:
        vol = vol / vol.max()

    return vol


def parameters_to_curve_coordinates(
        parameters,
        mode,
        n_points,
        worm_length,
        max_revolutions,
        bs=1,
        decay_factor=None,
):
    if parameters.ndim == 1:
        parameters = parameters.unsqueeze(0)

    if mode == ENCODING_MODE_POINTS:
        # Parameters are the curve coordinates
        return parameters.reshape((bs, n_points, 3))

    # First 3 parameters are the offset
    offset = parameters[:, :3]
    parameters = parameters[:, 3:]

    if mode == ENCODING_MODE_DELTA_VECTORS:
        # Remaining parameters are the delta vectors (de0s)
        delta_vectors = parameters.reshape((bs, n_points, 3))

        # Scale the ds's so that neighbouring points will be equidistant
        e0s = F.normalize(delta_vectors, dim=2)

    else:
        # Initial angles are unconstrained
        # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/11

        # theta: inclination - angle wrt z-axis: (0, pi)
        # phi: azimuth - rotation angle from x-y: (-pi, pi)
        pre_angles = parameters[:, :4]
        # pre_angles = F.hardtanh(pre_angles, min_val=-1, max_val=1)
        theta0 = (torch.atan2(pre_angles[:, 0], pre_angles[:, 1]) + np.pi) / 2
        phi0 = torch.atan2(pre_angles[:, 2], pre_angles[:, 3])
        parameters = parameters[:, 4:]

        # Determine maximum delta-angle
        max_delta_angle = max_revolutions * 2 * np.pi / n_points

        # Remaining parameters are the delta angles
        delta_angles = torch.tanh(parameters.reshape((bs, 2, -1))) * max_delta_angle

        # # Apply decay to delta angles so they go to 0 (ie, straight lines)
        # if decay_factor is not None:
        #     decay = torch.exp(-torch.arange(n_points, device=device) / n_points * decay_factor).repeat(bs)
        #     delta_angles = delta_angles * decay[1:].reshape((1, 1, n_points - 1))

        # Sum the initial angles with the delta angles to give the progression
        delta_thetas = torch.cat([theta0.unsqueeze(1), delta_angles[:, 0]], dim=-1)
        delta_phis = torch.cat([phi0.unsqueeze(1), delta_angles[:, 1]], dim=-1)
        thetas = torch.cumsum(delta_thetas, dim=-1)
        phis = torch.cumsum(delta_phis, dim=-1)

        # Convert to cartesian coordinates to find the e0 unit vectors
        e0s = torch.stack([
            torch.cos(phis) * torch.sin(thetas),
            torch.sin(phis) * torch.sin(thetas),
            torch.cos(thetas),
            ], dim=-1)

    # Scale the e0s (which have unit length) so the arc length is fixed
    e0s_scaled = e0s * worm_length / n_points

    # Start at the offset and add the scaled e0s's to form the curve
    curve_coordinates = offset + torch.cumsum(e0s_scaled, dim=1)

    return curve_coordinates


class EncDec(nn.Module):
    def __init__(
            self,
            n_cloud_points: int,
            n_curve_points: int = N_WORM_POINTS,
            worm_length: float = 1.,
            max_revolutions: float = 2,
            blur_sigma_masks: float = 0,
            blur_sigma_masks_curve: float = 0,
            blur_sigma_vols: float = 0,
            mode: str = ENCODING_MODE_DELTA_ANGLES,
            n_basis_fns: int = 4
    ):
        super().__init__()
        self.n_cloud_points = n_cloud_points
        self.n_curve_points = n_curve_points
        self.worm_length = worm_length
        self.max_revolutions = max_revolutions
        self.blur_sigma_masks = blur_sigma_masks
        self.blur_sigma_masks_curve = blur_sigma_masks_curve
        self.blur_sigma_vols = blur_sigma_vols
        self.mode = mode
        self.n_basis_fns = n_basis_fns
        self.cams = DynamicCameras()

        # Grow the worm out by slowly letting more gradients through
        self.decay_factor = torch.tensor(MAX_DECAY_FACTOR, dtype=torch.float32)

        if self.mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
            # Fix frequencies
            ws = [1 / 4, ]  # base frequency in units 2pi
            for n in range(self.n_basis_fns - 1):
                ws.append(ws[-1] * 2)
            w_n = torch.tensor(ws) * 2 * np.pi
            self.register_buffer('w_n', w_n)

            # Sample point locations
            t = torch.linspace(0, 1, n_curve_points - 1)
            self.register_buffer('w*t', torch.einsum('n,t->nt', w_n, t))

    def forward(
            self,
            shifts: torch.Tensor,
            parameters: torch.Tensor,
            camera_coeffs: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
            masks_target: torch.Tensor,
            blur_sigmas_cloud: torch.Tensor,
            blur_sigmas_curve: torch.Tensor
    ):
        bs = parameters.shape[0]
        device = parameters.device

        # Extract point cloud coordinates
        cloud_points = parameters[:, :self.n_cloud_points * 3].reshape((bs, self.n_cloud_points, 3))
        parameters = parameters[:, self.n_cloud_points * 3:]

        # Remaining parameters define the curve
        curve_points = parameters_to_curve_coordinates(
            parameters=parameters,
            mode=self.mode,
            n_points=self.n_curve_points,
            worm_length=self.worm_length,
            max_revolutions=self.max_revolutions,
            bs=bs,
            decay_factor=self.decay_factor
        )

        # Add the 3d centre point offset to centre on the camera
        cloud_points_3d = points_3d_base.unsqueeze(1) + cloud_points
        curve_points_3d = points_3d_base.unsqueeze(1) + curve_points

        # Apply translation shift adjustments
        # camera_coeffs[:, :, 21] = camera_coeffs[:, :, 21] + shifts

        # Project 3D points to 2D
        cloud_points_2d = self.cams.forward(camera_coeffs, cloud_points_3d)
        curve_points_2d = self.cams.forward(camera_coeffs, curve_points_3d)

        # Re-centre according to 2D base points plus a (100,100) to put it in the centre of the cropped image
        image_centre_pt = torch.ones((bs, 1, 1, 2), dtype=torch.float32, device=device) * PREPARED_IMAGE_SIZE[0] / 2
        cloud_points_2d_net = cloud_points_2d - points_2d_base.unsqueeze(2) + image_centre_pt
        curve_points_2d_net = curve_points_2d - points_2d_base.unsqueeze(2) + image_centre_pt

        # Generate the masks from the 2D image points and get the scores for the cloud points
        # masks_cloud, cloud_points_scores = PointsToMasks.apply(cloud_points_2d_net, self.blur_sigma_masks, masks_target)
        # idxs = _get_idxs_2d(cloud_points_2d_net)
        # cloud_points_scores = masks_target[[*idxs]].reshape(cloud_points_2d_net.shape[:-1])
        masks_cloud, blobs = render_points(cloud_points_2d_net, blur_sigmas_cloud)

        # Normalise blobs
        sum_ = blobs.sum(dim=(2,3), keepdim=True)
        sum_ = sum_.clamp(min=1e-8)
        blobs_normed = blobs/sum_

        points_scores_individual = (blobs_normed*masks_target.unsqueeze(2)).sum(dim=(3,4)).mean(dim=1)
        cloud_points_scores = points_scores_individual  #+ points_scores_combined
        # points_scores_combined = (blobs*(masks_target-masks_cloud).unsqueeze(2)).sum(dim=(3,4)).mean(dim=1)
        # cloud_points_scores = points_scores_combined

        # masks_cloud = render_points(cloud_points_2d_net, self.blur_sigma_masks)
        # masks_curves = _points_to_masks(curve_points_2d_net, self.blur_sigma_masks_curve, decay_factor=self.decay_factor)
        masks_curves = render_curve(curve_points_2d_net, blur_sigmas_curve)

        # # Balance the overall masks output
        # ratio_cloud = masks_target.sum(dim=(2,3), keepdim=True)/masks_cloud.sum(dim=(2,3), keepdim=True)
        # ratio_curves = masks_target.sum(dim=(2,3), keepdim=True)/masks_curves.sum(dim=(2,3), keepdim=True)
        # masks_cloud = masks_cloud * ratio_cloud
        # masks_curves = masks_curves * ratio_curves

        # Get the point scores for the curve points
        # curve_points_scores = CloudToCurve.apply(curve_points, cloud_points, cloud_points_scores, self.blur_sigma_vols,
        #                                          self.decay_factor)
        curve_points_scores = cloud_points_scores

        return masks_cloud, masks_curves, cloud_points_scores, curve_points_scores


def plot_mask(X, title=None, points=None, show=True):
    if isinstance(X, torch.Tensor):
        X = to_numpy(X)
    while X.ndim > 2:
        X = X.squeeze(0)
    m = plt.imshow(X)
    plt.gcf().colorbar(m)
    if title is not None:
        plt.gca().set_title(title)

    if points is not None:
        if isinstance(points, torch.Tensor):
            points = to_numpy(points)
        while points.ndim > 2:
            points = points.squeeze(0)
        plt.scatter(y=points[:, 1], x=points[:, 0], s=20, c='red', marker='x', zorder=10)

    if show:
        plt.show()


def find_midline3d(
        masks_id: str,
        n_cloud_points: int = 1000,
        n_worm_points: int = 10,
        blur_sigma_masks_cloud_init: float = 1,
        blur_sigma_masks_curve_init: float = 1,
        blur_sigma_vols: float = 1,
        max_revolutions: int = 2,
        mode: str = ENCODING_MODE_DELTA_ANGLES,
        n_basis_fns: int = 4,
        n_steps: int = 2000,
        n_warmup_steps: int = 100,
        n_straight_steps: int = 100,
        checkpoint_every_n_steps: int = -1,
        resume_from_run: str = None,
        resume_from_step: int = None,
        save_plots:bool =False,
        show_plots:bool =True
):
    # interactive_plots()
    masks: SegmentationMasks = SegmentationMasks.objects.get(id=masks_id)
    trial = masks.trial
    images = masks.get_images()
    frame = masks.frame
    masks_target = torch.from_numpy(masks.X)
    masks_target[masks_target > 0.4] = 1
    masks_target /= masks_target.sum(axis=(1,2), keepdim=True)
    # masks_target /= masks_target.max()
    point_3d_base = torch.tensor(frame.centre_3d.point_3d)
    points_2d_base = torch.tensor(frame.centre_3d.reprojected_points_2d)
    # cameras = frame.centre_3d.cameras
    cameras = frame.get_cameras()
    worm_length = masks.trial.experiment.worm_length
    logger.debug(f'Worm length = {worm_length}')
    azim = -60
    n_adjustment_steps = n_steps - n_warmup_steps
    max_revolutions_absolute = max_revolutions
    max_revolutions_init = 0
    max_revolutions = max_revolutions_init
    shifts_reg = 1e-3

    if resume_from_run is not None:
        assert resume_from_step is not None
        logs_path = LOGS_PATH + '/find_midline3d/' + resume_from_run
    else:
        logs_path = LOGS_PATH + '/find_midline3d/' + START_TIMESTAMP
    if checkpoint_every_n_steps != -1:
        os.makedirs(logs_path, exist_ok=True)
    if save_plots:
        os.makedirs(logs_path + '/3d', exist_ok=True)
        os.makedirs(logs_path + '/2d', exist_ok=True)
        os.makedirs(logs_path + '/sigmas', exist_ok=True)
        os.makedirs(logs_path + '/scores', exist_ok=True)

    # Extract camera coefficients
    fx = np.array([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
    fy = np.array([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
    cx = np.array([cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
    cy = np.array([cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
    R = np.array([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
    t = np.array([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
    d = np.array([cameras.distortion[c] for c in CAMERA_IDXS])
    if cameras.shifts is not None:
        s = np.array([cameras.shifts.dx, cameras.shifts.dy, cameras.shifts.dz])
    else:
        s = np.array([0,0,0])
    print('shifts=', s)
    cam_coeffs = np.concatenate([
        fx.reshape(3, 1), fy.reshape(3, 1), cx.reshape(3, 1), cy.reshape(3, 1), R.reshape(3, 9), t, d, s.reshape(3, 1)
    ], axis=1).astype(np.float32)
    cam_coeffs = torch.from_numpy(cam_coeffs)

    # make test 1 - 1 point
    if 0:
        p3d1 = np.array(frame.centre_3d.point_3d)
        p3d1[0] += 0.1
        p3d1[1] += 0.1
        b2d = np.array(frame.centre_3d.reprojected_points_2d)
        ct = cameras.get_camera_model_triplet()
        object_points=np.array([p3d1])
        print('object_points', object_points)
        print('object_points.shape', object_points.shape)
        t2d = np.array(ct.project_to_2d(object_points))  # [0]
        masks_target = []
        for c in CAMERA_IDXS:
            test_points = np.array(t2d[:, c] - b2d[c]) + np.array([[100, 100]])
            print(test_points.shape)
            print(test_points)
            masks_target.append(make_segmentation_mask(test_points, draw_mode='pixels', blur_sigma=3))
        masks_target = np.array(masks_target)
        masks_target = torch.from_numpy(masks_target)
        masks_target /= masks_target.max()

        dc = DynamicCameras()
        test_points_2 = dc.forward(cam_coeffs.unsqueeze(0), torch.from_numpy(object_points).unsqueeze(0).to(torch.float32))

        print('t2d.shape', t2d.shape)
        test_points_1 = torch.from_numpy(t2d)
        test_points_1 = test_points_1.permute(1, 0, 2)
        test_points_1 = test_points_1.unsqueeze(0).to(torch.float32)
        print('test_points_1.shape', test_points_1.shape)
        print('test_points_2.shape', test_points_2.shape)
        assert torch.allclose(test_points_1, test_points_2)

    # make test 2 - 2-point line
    if 0:
        p3d1 = np.array(frame.centre_3d.point_3d)
        p3d1[0] -= 0.1
        p3d1[2] += 0.1
        p3d2 = np.array(frame.centre_3d.point_3d)
        p3d2[0] += 0.1
        p3d2[1] -= 0.1
        b2d = np.array(frame.centre_3d.reprojected_points_2d)
        ct = cameras.get_camera_model_triplet()
        object_points=np.array([p3d1, p3d2])
        print('object_points', object_points)
        print('object_points.shape', object_points.shape)
        t2d = np.array(ct.project_to_2d(object_points))
        masks_target = []
        for c in CAMERA_IDXS:
            test_points = np.array(t2d[:, c] - b2d[c]) + np.array([[100, 100]])
            print(test_points.shape)
            print(test_points)
            masks_target.append(make_segmentation_mask(test_points, draw_mode='pixels', blur_sigma=3))
        masks_target = np.array(masks_target)
        masks_target = torch.from_numpy(masks_target)

    # make test 3 - find a pre-computed 3d midline and use the projections as the targets
    if 0:
        m3d = Midline3D.objects(frame=frame)[0]
        masks_target = m3d.get_segmentation_masks(blur_sigma=3)
        masks_target = np.array(masks_target)
        masks_target = torch.from_numpy(masks_target)
        masks_target /= masks_target.max()

    # Set initial parameter vectors to be optimised
    # cam_coeffs_adj = torch.zeros_like(cam_coeffs)
    cam_coeffs_adj = cam_coeffs.clone().detach()
    shifts = torch.zeros(3)

    # Distribute initial cloud points randomly on surface of a sphere
    mean = torch.zeros(n_cloud_points, 3)
    x = torch.normal(mean=mean, std=1)
    x = x / torch.norm(x, dim=-1, keepdim=True) * 0.4
    cloud0 = x.flatten()
    p0 = [cloud0]

    if mode == ENCODING_MODE_POINTS:
        cc0 = torch.linspace(-np.sqrt(3)/6, np.sqrt(3)/6, n_worm_points).repeat_interleave(3)
        # mean = torch.zeros(3 * n_worm_points)
        # x = torch.normal(mean=mean, std=0.01)
        # cc0 = x.flatten()
        p0.append(cc0)
    elif mode == ENCODING_MODE_DELTA_VECTORS:
        offset0 = torch.zeros(3)
        p0.append(offset0)
        delta_vectors0 = torch.normal(mean=torch.zeros(3 * n_worm_points), std=0.2)
        p0.append(delta_vectors0)
    else:
        offset0 = torch.zeros(3)
        p0.append(offset0)
        pre_angles0 = torch.rand(size=(4,)) * 2 - 1
        p0.append(pre_angles0)

        if mode == ENCODING_MODE_DELTA_ANGLES:
            delta_angles0 = torch.normal(mean=torch.zeros(2 * (n_worm_points - 1)), std=0.1)
            p0.append(delta_angles0)
        elif mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
            amps0 = torch.normal(mean=torch.zeros(2 * n_basis_fns), std=1)
            phases0 = torch.normal(mean=torch.zeros(2 * n_basis_fns), std=1)
            p0.append(amps0)
            p0.append(phases0)
    px = torch.cat(p0)

    # Add batch dims and put on correct device
    cam_coeffs = cam_coeffs.unsqueeze(0).to(device)
    point_3d_base = point_3d_base.unsqueeze(0).to(device)
    points_2d_base = points_2d_base.unsqueeze(0).to(device)
    masks_target = masks_target.unsqueeze(0).to(device)
    masks_cloud: torch.Tensor
    masks_curve: torch.Tensor
    shifts = shifts.to(device)
    shifts.requires_grad = True
    px = px.to(device)
    px.requires_grad = True
    cam_coeffs_adj = cam_coeffs_adj.to(device)
    cam_coeffs_adj.requires_grad = True

    blur_sigmas_cloud = torch.ones(n_cloud_points) * blur_sigma_masks_cloud_init
    blur_sigmas_cloud = blur_sigmas_cloud.to(device)
    blur_sigmas_cloud.requires_grad = True

    blur_sigmas_curve = torch.ones(n_worm_points) * blur_sigma_masks_curve_init
    blur_sigmas_curve = blur_sigmas_curve.to(device)
    blur_sigmas_curve.requires_grad = True

    # Build modules
    encdec = EncDec(
        n_cloud_points=n_cloud_points,
        n_curve_points=n_worm_points,
        worm_length=worm_length,
        max_revolutions=max_revolutions_init,
        blur_sigma_masks=blur_sigma_masks_cloud_init,
        blur_sigma_masks_curve=blur_sigma_masks_curve_init,
        blur_sigma_vols=blur_sigma_vols,
        mode=mode,
        n_basis_fns=n_basis_fns
    )

    optimiser = Adam(
        [
            {'params':(px,)},
            {'params':(cam_coeffs_adj,), 'lr': 1e-5},
            {'params':(blur_sigmas_cloud,), 'lr': 1e-4},
            # {'params':(blur_sigmas_curve,), 'lr': 0}  # 0.0001},
        ],
        lr=0.01,
        amsgrad=True,
        weight_decay=0
        # momentum=0.9
    )
    # optimiser = RMSprop(
    #     [
    #         # {'params':(shifts,), 'lr': 0.0001, 'momentum': 0},  #.9},
    #         {'params':(px,)},
    #         # {'params':(cam_coeffs_adj,), 'lr': 0.00002},
    #    ] ,
    #     lr=0.1,
    #     momentum=0.9
    # )
    optimiser_curve = RMSprop(
        params=(px,),
        lr=0.001,
    )

    # optimiser = LBFGS(
    #     params=(px,), #cam_coeffs_adj),
    #     lr=0.1,
    #     max_iter=100,
    #     history_size=5000,
    #     tolerance_grad=1e-12,
    #     # line_search_fn='strong_wolfe'
    #     line_search_fn=None
    # )

    # lr_scheduler = CyclicLR(
    #     optimizer=optimiser,
    #     base_lr=0.0001,
    #     max_lr=0.001,
    #     step_size_up=1000,
    #     mode='triangular2',
    #     gamma=0.99,
    #     # cycle_momentum=True
    # )

    if resume_from_step is not None:
        path =logs_path + f'/{resume_from_step}.chkpt'
        logger.info(f'Resuming from {path}')
        state = torch.load(path, map_location=device)
        shifts.data = state['shifts']
        px.data = state['px']
        optimiser.load_state_dict(state['optim_sd'])
        optimiser_curve.load_state_dict(state['optim_curve_sd'])
        start_step = resume_from_step
    else:
        start_step = 0

    encdec.decay_factor = torch.tensor(0., device=device)

    # Optimise
    for i in range(start_step, start_step+n_steps):
        # Set decay factor
        # if i > n_warmup_steps:
        #     encdec.decay_factor = torch.tensor(max(0, 1 - i / (n_adjustment_steps / 2)) * MAX_DECAY_FACTOR)

        # Set max curvature
        if i > (n_warmup_steps+n_straight_steps):
            x = (i-n_warmup_steps-n_straight_steps) / (n_steps - n_warmup_steps-n_straight_steps)
            max_revolutions = torch.tensor(min(1, max(0, x))) * max_revolutions_absolute
            # max_revolutions = torch.tensor(min(1, max(0, (i-n_warmup_steps) / 2))) * max_revolutions_absolute
            # encdec.decay_factor = torch.tensor(max(0, 1 - i / (n_adjustment_steps / 2)) * MAX_DECAY_FACTOR)
            encdec.max_revolutions = max_revolutions

        #
        # # Generate masks from parameters
        # masks_cloud, masks_curve, cloud_points_scores, curve_points_scores = encdec.forward(
        #     shifts=shifts.unsqueeze(0),
        #     parameters=px.unsqueeze(0),
        #     # camera_coeffs=cam_coeffs.clone().detach(),  #_i.unsqueeze(0),
        #     # camera_coeffs=cam_coeffs_i,
        #     camera_coeffs=cam_coeffs_adj.unsqueeze(0),
        #     points_3d_base=point_3d_base,
        #     points_2d_base=points_2d_base,
        #     masks_target=masks_target
        # )

        curve_loss = 0
        arc_length = 0
        sl_loss = 0
        al_loss = 0
        angles_loss = 0
        cc = None

        inputs = {
            'shifts':shifts.unsqueeze(0),
            'parameters':px.unsqueeze(0),
            'camera_coeffs':cam_coeffs_adj.unsqueeze(0),
            'points_3d_base':point_3d_base,
            'points_2d_base':points_2d_base,
            'masks_target':masks_target,
            'blur_sigmas_cloud': blur_sigmas_cloud,
            'blur_sigmas_curve': blur_sigmas_curve
        }

        def forward(inputs_):
            # Generate masks from parameters
            outputs_ = encdec.forward(**inputs_)
            return outputs_

        def calculate_loss(outputs_):
            masks_cloud, masks_curve, cloud_points_scores, curve_points_scores = outputs_

            # loss = -cloud_points_scores.sum()

            # loss = torch.sum((torch.log(1+masks_cloud) - torch.log(1+ masks_target))**2)
            # loss += torch.sum((torch.log(1+masks_curve) - torch.log(1+ masks_cloud.detach()))**2)
            # loss += torch.sum((torch.log(1+masks_curve) - torch.log(1+ masks_cloud))**2)
            # loss = -torch.sum(cloud_points_scores)

            # if i < n_warmup_steps:  # or curve_loss > 1:
            #     loss = -(cloud_points_scores.sum())
            # else:
            #     loss = -(cloud_points_scores.sum() + curve_points_scores.sum())
            # # loss += curve_loss
            # loss += curve_loss
            #
            # # Prefer small shifts
            # loss += shifts_reg * (shifts**2).sum()
            # assert not is_bad(loss)
            #
            # Multiscale loss
            ms_loss_type = 'logs'
            # ms_loss_type = 'mse'
            # ms_loss_type = 'kl'
            loss_ms = 0
            cloud_rep = masks_cloud.clone()
            curve_rep = masks_curve.clone()
            target_rep = masks_target.clone()
            k = 1
            while cloud_rep.shape[-1] > 1:
                if ms_loss_type == 'mse':
                    rep_loss_ct = F.mse_loss(cloud_rep, target_rep)
                    # rep_loss_cc = F.mse_loss(curve_rep, cloud_rep.detach())
                elif ms_loss_type == 'kl':
                    rep_loss_ct = F.kl_div(cloud_rep, target_rep)
                    # rep_loss_cc = F.kl_div(curve_rep, cloud_rep.detach())
                elif ms_loss_type == 'logs':
                    rep_loss_ct = torch.sum((torch.log(1+cloud_rep) - torch.log(1+ target_rep))**2)
                    rep_loss_cc = torch.sum((torch.log(1+curve_rep) - torch.log(1+ cloud_rep.detach()))**2)
                    # rep_loss_cc = 0

                if i < n_warmup_steps:
                    loss_ms += rep_loss_ct / k
                else:
                    loss_ms += (rep_loss_ct+ rep_loss_cc) / k
                # cloud_rep = _avg_pool_2d(cloud_rep, mode='replicate')
                # curve_rep = _avg_pool_2d(curve_rep, mode='replicate')
                # target_rep = _avg_pool_2d(target_rep, mode='replicate')
                cloud_rep = _avg_pool_2d(cloud_rep, oob_grad_val=0)
                curve_rep = _avg_pool_2d(curve_rep, oob_grad_val=0)
                target_rep = _avg_pool_2d(target_rep, oob_grad_val=0)
                k +=1
            loss = loss_ms

            #
            # # Curves
            # curve_rep = masks_curve.clone()
            # # target_rep = masks_target.clone()
            # target_rep = masks_cloud.clone()
            # while curve_rep.shape[-1] > 1:
            #     rep_loss = F.mse_loss(curve_rep, target_rep)
            #     loss_ms += rep_loss
            #     curve_rep = _avg_pool_2d(curve_rep, mode='replicate')
            #     target_rep = _avg_pool_2d(target_rep, mode='replicate')

            # loss += F.mse_loss(masks_cloud, masks_target, reduction='sum')
            # loss = F.mse_loss(masks_cloud, masks_curve)
            # loss = F.kl_div(masks_cloud, masks_target, reduction='batchmean')

            # sigmas_uniform = torch.ones_like(blur_sigmas_cloud) / n_cloud_points
            # reg = F.kl_div(blur_sigmas_cloud, sigmas_uniform)

            # loss += blur_sigmas_cloud.var()

            # loss += -curve_points_scores.sum()

            return loss

        def get_curve_loss():
            nonlocal curve_loss, arc_length, sl_loss, al_loss, angles_loss

            # Losses for curve length and curvatures
            if mode != ENCODING_MODE_POINTS:
                return 0
            max_delta_angle = max_revolutions * 2 * np.pi / n_worm_points
            l = worm_length / (n_worm_points-1)

            cc = px[-n_worm_points * 3:].reshape((n_worm_points, 3))
            assert not is_bad(cc)

            scale_sl = 50
            segment_lengths = torch.norm(cc[1:] - cc[:-1], dim=-1)
            sl_loss = scale_sl*((segment_lengths - l)**2).sum()
            sl_loss = torch.exp(scale_sl*((segment_lengths - l).abs())).mean()

            # sl_loss = segment_lengths.var()

            arc_length = segment_lengths.sum()
            # al_loss = (arc_length - worm_length)**2
            # print('arc_length', arc_length)

            eps = 1e-5
            # a = cc[1:-1] - cc[:-2]
            a = cc[:-2]- cc[1:-1]
            b = cc[2:] - cc[1:-1]
            # a = cc[:-2] - cc[1:-1]
            # b = cc[1:-1] - cc[2:]
            # an = torch.norm(a, dim=-1)
            # bn = torch.norm(b, dim=-1)
            # adotb = (a * b).sum(dim=-1)
            # # adotb[adotb < 0.1] = 0
            # acos_arg = adotb / (an * bn)
            # acos_arg = acos_arg.clamp(min=-1 + eps, max=1 - eps)
            # angles = torch.acos(acos_arg)
            #
            # # angles[angles > np.pi] = angles[angles > np.pi] - 2*np.pi
            # # angles[angles < 0.01] = 0.01
            # # assert not is_bad(angles)
            #
            # if max_delta_angle > 0:
            #     angles_loss = (((angles - np.pi) / max_delta_angle)**2).sum()
            # else:
            #     angles_loss = ((angles - np.pi)**2).sum()

            scale_al = 50
            min_dist_1hop = l * np.sqrt(2*(1-np.cos(np.pi - max_delta_angle)))
            angles_loss = torch.exp(scale_al * (min_dist_1hop - torch.norm(a-b, dim=-1))).sum()
            # angles_loss = (dists_loss/n_worm_points).sum()

            # angles_loss = ((angles.abs() - max_delta_angle)**2).sum()
            # angles_loss = ((angles - max_delta_angle)**2).sum()
            # angles_loss = ((angles / max_delta_angle)**4).sum() / 50000
            # if max_delta_angle > 0:
            #     angles_loss = torch.clamp((angles / max_delta_angle)**4, max=100).mean()
            # else:
            #     angles_loss = torch.clamp(angles**2, max=100).mean() * 10000
            # angles_loss = angles[angles>max_delta_angle].sum() / 50000
            # curve_loss = sl_loss
            curve_loss = sl_loss + angles_loss
            # curve_loss = angles_loss + al_loss
            # curve_loss = al_loss
            # loss += sl_loss  # + angles_loss

            return curve_loss

        def closure():
            if torch.is_grad_enabled():
                optimiser.zero_grad()
            outputs = forward(inputs)
            loss = calculate_loss(outputs)
            if loss.requires_grad:
                loss.backward()
            return loss

        optimiser.step(closure)
        masks_cloud, masks_curve, cloud_points_scores, curve_points_scores = forward(inputs)
        loss = closure()

        # Ensure sigmas are positive
        with torch.no_grad():
            min_sig = 0.01
            blur_sigmas_cloud.clamp_(min=min_sig)
            blur_sigmas_curve.clamp_(min=min_sig)

            if 1:
                min_score = 0.0001

                # Every so often, any points with low scores should clone
                # to somewhere nearby a point with a better score
                max_turnover = int(n_cloud_points * 0.01)
                # low_sig_idxs = (blur_sigmas_cloud <= min_sig).nonzero()

                # low_score_idxs = (cloud_points_scores.squeeze() <= min_score).sum()
                n_scored_too_low = (cloud_points_scores.squeeze() <= min_score).sum()

                # n_to_relocate = min(max_turnover, len(low_sig_idxs))
                n_to_relocate = min(max_turnover, n_scored_too_low)
                if n_to_relocate > 0:
                    scored_idxs = torch.argsort(cloud_points_scores[0], descending=True)
                    logger.debug(f'relocating {n_to_relocate} points')
                    # src_idxs = low_sig_idxs.squeeze(dim=1)[:n_to_relocate]
                    src_idxs = scored_idxs[-n_to_relocate:]

                    # dest_idxs = torch.argsort(blur_sigmas_cloud, descending=True)[:len(low_sig_idxs)]
                    dest_idxs = scored_idxs[:n_to_relocate]

                    # Randomise destinations
                    random_idxs = torch.randperm(n_to_relocate)
                    dest_idxs = dest_idxs[random_idxs]

                    # Relocate points
                    cloud_points = px[:n_cloud_points * 3].reshape((n_cloud_points, 3))
                    cloud_points[src_idxs] = torch.normal(
                        mean=cloud_points[dest_idxs],
                        std=blur_sigmas_cloud[dest_idxs][:, None].expand_as(cloud_points[dest_idxs])
                    )
                    px[:n_cloud_points * 3] = cloud_points.reshape(n_cloud_points * 3)

                    # Update sigmas
                    blur_sigmas_cloud[dest_idxs] = blur_sigmas_cloud[dest_idxs] # / np.sqrt(2)
                    blur_sigmas_cloud[src_idxs] = blur_sigmas_cloud[dest_idxs]   # / np.sqrt(2)

        # # Take an optimisation step
        # optimiser.zero_grad()
        # loss.backward()
        # optimiser.step()
        # # lr_scheduler.step()

        # Fix curve
        if 1:
            optimiser_curve.zero_grad()
            curve_loss = get_curve_loss()
            cl_threshold = 5.
            curve_fix_count = 0
            max_curve_fix_steps = 100
            while (sl_loss > cl_threshold or angles_loss > cl_threshold) and curve_fix_count < max_curve_fix_steps:
                curve_fix_count += 1
                # Keep optimising until the curve losses are within the threshold
                optimiser_curve.zero_grad()
                curve_loss.backward()
                optimiser_curve.step()
                assert not is_bad(px)
                curve_loss = get_curve_loss()
                # print(f'curve_loss={curve_loss}')
            if curve_fix_count > 0:
                logger.info(f'Fixed curve in {curve_fix_count} steps.')

        if 1 or i % 10 == 0:   # f'lr={lr_scheduler.get_last_lr()[0]:.5f}. ' \
            log = f'Step {i}. ' \
                  f'Loss={loss:.6f}. ' \
                  f'df={encdec.decay_factor:.2f}. ' \
                  f'cloud={cloud_points_scores.sum():.3f}. ' \
                  f'curve={curve_points_scores.sum():.3f}. ' \
                  f'shifts={",".join([f"{s.item():.2f}" for s in shifts])}. ' \
                  f'rev={max_revolutions:.3f}.' \
                  f'cc_adj={cam_coeffs_adj.sum():.3f}.'
            # log = f'Step {i}. ' \
            #       f'rev={max_revolutions:.3f}.'
            if mode == ENCODING_MODE_POINTS:
                log += f' al_loss={al_loss:.4f} ' \
                       f'sl_loss={sl_loss:.4f} ' \
                       f'angles={angles_loss:.4f} ' \
                       f'cl={curve_loss:.4f} ' \
                       f'arc_length={arc_length:.3f}'
            logger.info(log)

        def plot_check():
            nonlocal  cam_coeffs, points_2d_base, point_3d_base
            cloud_points = px[:n_cloud_points * 3].reshape((n_cloud_points, 3))
            curve_points = torch.from_numpy(object_points - np.array(frame.centre_3d.point_3d))
            dc = DynamicCameras()
            cam_coeffs = cam_coeffs.clone()
            cp = torch.tensor(frame.centre_3d.point_3d)
            ip = torch.tensor([100,100])
            curve_points_2d_orig_shifts = dc.forward(cam_coeffs, (curve_points+cp).unsqueeze(0).to(torch.float32)) - points_2d_base.unsqueeze(2) + ip
            cam_coeffs[:, :, 21] = cam_coeffs[:, :, 21] + shifts
            curve_points_2d = dc.forward(cam_coeffs, (curve_points+cp).unsqueeze(0).to(torch.float32)) - points_2d_base.unsqueeze(2) + ip
            cloud_points_2d = dc.forward(cam_coeffs, (cloud_points+cp).unsqueeze(0).to(torch.float32)) - points_2d_base.unsqueeze(2) + ip

            curve_masks_os = _points_to_masks(curve_points_2d_orig_shifts, blur_sigma=4)
            curve_masks = _points_to_masks(curve_points_2d, blur_sigma=4)
            cloud_masks = _points_to_masks(cloud_points_2d)

            fig, axes = plt.subplots(3, figsize=(6, 8))
            image_triplet = np.concatenate(images, axis=1)
            curve_triplet_os = np.concatenate(to_numpy(curve_masks_os[0]), axis=1)
            curve_triplet = np.concatenate(to_numpy(curve_masks[0]), axis=1)
            cloud_triplet = np.concatenate(to_numpy(cloud_masks[0]), axis=1)

            ax = axes[0]
            ax.set_title('Curve')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = curve_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

            ax = axes[1]
            ax.set_title('Curve OS')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = curve_triplet_os.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(curve_triplet_os, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

            ax = axes[2]
            ax.set_title('Cloud')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = cloud_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(cloud_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)


            ax.set_title(f'Step {i}')
            fig.tight_layout()

            plt.show()
            plt.close(fig)

            exit()

        def plot_3d(cloud_point_threshold=0):
            nonlocal azim, cam_coeffs, points_2d_base, point_3d_base
            azim += 15
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(azim=azim)
            # cloud_points = px[9:9 + n_cloud_points * 3].reshape((n_cloud_points, 3))
            # cloud_points = px[3:3 + n_cloud_points * 3].reshape((n_cloud_points, 3))
            cloud_points = px[:n_cloud_points * 3].reshape((n_cloud_points, 3))
            curve_points = parameters_to_curve_coordinates(
                # parameters=px[9 + n_cloud_points * 3:],
                # parameters=px[3 + n_cloud_points * 3:],
                parameters=px[n_cloud_points * 3:],
                mode=mode,
                n_points=n_worm_points,
                worm_length=worm_length,
                max_revolutions=max_revolutions,
                decay_factor=encdec.decay_factor
            )[0]

            # curve_points = torch.from_numpy(object_points) - np.array(frame.centre_3d.point_3d)

            # scores = cloud_points_scores.mean(dim=1)[0]
            scores = cloud_points_scores[0]
            if cloud_point_threshold > 0:
                above_threshold = scores>cloud_point_threshold
                scores = scores[above_threshold]
                cloud_points = cloud_points[above_threshold]
            x, y, z = (to_numpy(cloud_points[:, j]) for j in range(3))
            s1 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='autumn_r', s=20, alpha=0.4)
            fig.colorbar(s1)
            x, y, z = (to_numpy(curve_points[:, j]) for j in range(3))
            scores = curve_points_scores[0]
            # s2 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='YlGnBu', s=50, marker='x', alpha=0.9)
            s2 = ax.scatter(x, y, z, color='black', s=75, marker='x', alpha=0.9)
            fig.colorbar(s2)

            ax.set_title(f'Step {i}')
            fig.tight_layout()

            if save_plots:
                save_dir = f'{logs_path}/3d'
                if cloud_point_threshold > 0:
                    save_dir += f'/T={cloud_point_threshold:.1f}'
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_dir + f'/{i:06d}.png')
            if show_plots:
                plt.show()
            plt.close(fig)

        def plot_attempt():
            X_target = to_numpy(masks_target[0])
            X_cloud = to_numpy(masks_cloud[0])
            X_curve = to_numpy(masks_curve[0])

            nrows = 3
            fig, axes = plt.subplots(nrows, figsize=(6, 8))
            fig.suptitle(
                f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
                f'Frame: {frame.frame_num}. Step = {i}'
            )

            # Stitch images and masks together
            image_triplet = np.concatenate(images, axis=1)
            X_target_triplet = np.concatenate(X_target, axis=1) / X_target.max()
            X_cloud_triplet = np.concatenate(X_cloud, axis=1) / X_cloud.max()
            X_curve_triplet = np.concatenate(X_curve, axis=1) / X_curve.max()

            ax = axes[0]
            ax.set_title('Target')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = X_target_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X_target_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

            ax = axes[1]
            ax.set_title('Cloud')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = X_cloud_triplet.copy()
            # alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X_cloud_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

            ax = axes[2]
            ax.set_title('Curve')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = X_curve_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

            fig.tight_layout()

            if save_plots:
                fn = f'{logs_path}/2d/{i:06d}.png'
                plt.savefig(fn)
            if show_plots:
                plt.show()
            plt.close(fig)


        def plot_sigmas():
            sigmas_cloud = to_numpy(blur_sigmas_cloud)
            sigmas_curve = to_numpy(blur_sigmas_curve)

            nrows = 2
            ncols = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8))
            fig.suptitle(
                f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
                f'Frame: {frame.frame_num}. Step = {i}'
            )

            ax = axes[0, 0]
            ax.plot(sigmas_cloud)
            ax.set_title('blur_sigmas_cloud')

            ax = axes[0,1]
            ax.plot(np.sort(sigmas_cloud))
            ax.set_title('sort(blur_sigmas_cloud)')

            ax = axes[1, 0]
            ax.plot(sigmas_curve)
            ax.set_title('blur_sigmas_curve')

            ax = axes[1,1]
            ax.plot(np.sort(sigmas_curve))
            ax.set_title('sort(blur_sigmas_curve)')

            fig.tight_layout()

            if save_plots:
                fn = f'{logs_path}/sigmas/{i:06d}.png'
                plt.savefig(fn)
            if show_plots:
                plt.show()
            plt.close(fig)


        def plot_scores():
            point_scores = to_numpy(cloud_points_scores[0])
            nrows = 2
            ncols = 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8))
            fig.suptitle(
                f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
                f'Frame: {frame.frame_num}. Step = {i} \n'
                f'sum(point_scores)={point_scores.sum():.4f}'
            )

            ax = axes[0]
            ax.plot(point_scores)
            ax.set_title('point_scores')

            ax = axes[1]
            ax.plot(np.sort(point_scores))
            ax.set_title('sort(point_scores)')

            fig.tight_layout()

            if save_plots:
                fn = f'{logs_path}/scores/{i:06d}.png'
                plt.savefig(fn)
            if show_plots:
                plt.show()
            plt.close(fig)

        # plot_check()

        # Plot
        if i % 100 == 0 or i == 0 or i == n_steps - 1:
            # interactive_plots()
            plot_3d()
            # plot_3d(cloud_point_threshold=0.3)
            # plot_3d(cloud_point_threshold=0.6)
            # plot_3d(cloud_point_threshold=0.9)

        # Plot
        if i % 20 == 0 or i == 0 or i == n_steps - 1:
            plot_attempt()
            plot_sigmas()
            plot_scores()

        # Checkpoint
        if checkpoint_every_n_steps != -1 and (i+1)%checkpoint_every_n_steps == 0:
            path = f'{logs_path}/{i}.chkpt'
            logger.info(f'Saving checkpoint to {path}')
            torch.save({
                'shifts':shifts,
                'px':px,
                'optim_sd':optimiser.state_dict(),
                'optim_curve_sd':optimiser_curve.state_dict()
            }, path)


if __name__ == '__main__':
    find_midline3d(
        # masks_id='60801a42f782c04c8abf6ed0',
        # masks_id='608019f0f782c04c8abf6c41',
        masks_id='6080190ef782c04c8abf6b02',  # cloud works @ 20210831_2215  !
        # masks_id='607ffb53f782c04c8abd4e71',   # cloud works @ 20210831_1922  !
        n_cloud_points=500,
        n_worm_points=20,
        blur_sigma_masks_cloud_init=0.01,
        blur_sigma_masks_curve_init=0.1,
        blur_sigma_vols=6,
        max_revolutions=.5,
        mode=ENCODING_MODE_POINTS,
        # mode=ENCODING_MODE_DELTA_VECTORS,
        # mode=ENCODING_MODE_DELTA_ANGLES,
        # mode=ENCODING_MODE_DELTA_ANGLES_BASIS,
        n_basis_fns=4,
        n_steps=500000,
        n_warmup_steps=300,
        n_straight_steps=0,
        checkpoint_every_n_steps=10000,
        # resume_from_run='20210603_1127',
        # resume_from_step=279999,
        save_plots=True,
        show_plots=False,
        # save_plots=False,
        # show_plots=True,
    )
