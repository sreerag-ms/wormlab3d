from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import RMSprop
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT, N_WORM_POINTS
from wormlab3d.midlines3d.args.network_args import MAX_DECAY_FACTOR
from wormlab3d.toolkit.util import to_numpy


class PointsToMasks(nn.Module):
    def __init__(
            self,
            blur_sigma: float = 0,
            n_points: int = N_WORM_POINTS,
            image_size: Tuple[int] = (PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
            max_n_optimisation_steps: int = 10,
            oob_grad_val: float = 1e-5,
            noise_std: float = 0.5,
    ):
        super().__init__()
        self.register_buffer('blur_sigma', torch.tensor(blur_sigma, dtype=torch.float32))
        self.n_points = n_points
        self.image_size = image_size
        self.max_n_optimisation_steps = max_n_optimisation_steps
        self.oob_grad_val = oob_grad_val
        self.noise_std = noise_std

        # Grow the worm out by slowly letting more gradients through
        self.register_buffer('decay_factor', torch.tensor(MAX_DECAY_FACTOR, dtype=torch.float32), persistent=True)

    def set_decay_factor(self, decay_factor: float):
        """Set the decay factor, requires a method as this net may be copied and distributed across devices"""
        self.decay_factor.fill_(decay_factor)

    def forward(
            self,
            points_2d: torch.Tensor,
            masks_target: torch.Tensor
    ):
        """
        Takes a batch of 2D points in coordinate form and generates 2D segmentation masks.
        Creating the indices from the points involves casting to int/long which is non-differentiable,
        so we approximate the gradient surface with central differences.
        """

        # Optimise the points to better fit the target mask
        p_opt = self._optimise_points(points_2d, masks_target)

        # Generate masks from both the input points and the optimal points
        masks_out = self._points_to_masks(points_2d)
        masks_opt = self._points_to_masks(p_opt)

        return masks_out, masks_opt, p_opt

    def _optimise_points(
            self,
            points_2d: torch.Tensor,
            masks_target: torch.Tensor
    ):
        decay = torch.exp(-torch.arange(self.n_points, device=points_2d.device) / self.n_points * self.decay_factor)
        coord_shape = points_2d.shape[0], 3, self.n_points

        def get_pixel_losses(idxs):
            return (1 - masks_target[[*idxs]].reshape(coord_shape)) * decay

        # Calculate the gradient surface approximation
        masks_out = self._points_to_masks(points_2d)
        masks_diff = masks_target - 1e-2 * masks_out
        J = self._calculate_gradient_surface(masks_diff)

        # Set initial coordinates
        p0 = points_2d.clone().detach()
        idxs = self._get_idxs(p0)

        # Set up the optimiser
        # optimiser = AdamW(params=(p0,), lr=0.1)
        optimiser = RMSprop(params=(p0,), lr=0.2)

        # Take some optimisation steps
        i = 0
        pixel_losses = get_pixel_losses(idxs)
        while pixel_losses.max() > 0.1 and i < self.max_n_optimisation_steps:
            grads = self._get_directional_gradients(J, idxs)
            optimiser.zero_grad()
            p0.grad = -grads * pixel_losses.unsqueeze(-1)
            optimiser.step()
            idxs = self._get_idxs(p0)
            pixel_losses = get_pixel_losses(idxs)
            i += 1

        return p0

    def _calculate_gradient_surface(self, mask_target: torch.Tensor):
        # Calculate gradient surface
        grad0 = -mask_target
        grads = [grad0]
        g = grad0
        while g.shape[-1] > 1:
            g2 = self._avg_grad(g)
            grads.append(g2)
            g = g2

        # Got all the grad averages, now add them together and average
        grad_sum = torch.zeros_like(grad0)
        for i, g in enumerate(grads):
            grad_sum += F.interpolate(grads[i], self.image_size, mode='bilinear', align_corners=False)
        grad_avg = grad_sum / len(grads)

        # Calculate directional gradients
        gapx = F.pad(grad_avg, (0, 0, 1, 1), mode='replicate')
        gx = (gapx[:, :, :-2] - gapx[:, :, 2:]) / 2
        gapy = F.pad(grad_avg, (1, 1, 0, 0), mode='replicate')
        gy = (gapy[:, :, :, :-2] - gapy[:, :, :, 2:]) / 2
        J = torch.stack([gx, gy])

        return J

    def _avg_grad(self, grad):
        # Average pooling with overlap and boundary values
        padded_grad = F.pad(grad, (1, 1, 1, 1), mode='constant', value=self.oob_grad_val)
        ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
        return ag

    def _get_idxs(self, points_2d: torch.Tensor):
        bs = points_2d.shape[0]
        device = points_2d.device
        point_idxs_2d = points_2d.round().to(torch.long)  # <-- this operation is non-differentiable!
        point_idxs_2d = point_idxs_2d.clamp(min=0, max=self.image_size[0] - 1)
        idxs = [
            torch.arange(bs, device=device).repeat_interleave(3 * self.n_points),
            torch.arange(3, device=device).repeat_interleave(self.n_points).repeat(bs),
            point_idxs_2d[:, :, :, 1].flatten(),
            point_idxs_2d[:, :, :, 0].flatten(),
        ]
        idxs = torch.stack([ix for ix in idxs])
        return idxs

    def _get_directional_gradients(self, J: torch.Tensor, idxs: torch.Tensor):
        # Determine direction of minimum gradient from each sample coordinate
        coord_shape = J.shape[1], 3, self.n_points
        pixel_grads_x = J[0][[*idxs]].reshape(coord_shape)
        pixel_grads_y = J[1][[*idxs]].reshape(coord_shape)
        points_2d_grad = torch.stack([pixel_grads_y, pixel_grads_x], dim=-1)

        # Add some noise
        if self.noise_std > 0:
            noise = torch.normal(mean=torch.zeros_like(points_2d_grad), std=self.noise_std)
            points_2d_grad += noise

        return points_2d_grad

    def _points_to_masks(self, points_2d: torch.Tensor) -> torch.Tensor:
        device = points_2d.device
        bs = points_2d.shape[0]
        idxs = self._get_idxs(points_2d)

        # Write ones at the indexed locations.
        # Note - duplicate indices are intentionally not summed as the target mask only shows the likelihood
        # of the worm being in this pixel and in many cases the worm may be oriented directly at the camera.
        # todo: needs fixing now duplicated pixels have different values, we should use the max...
        masks = torch.zeros((bs, 3, *self.image_size), device=device)
        mw = torch.exp(-torch.arange(self.n_points, device=device) / self.n_points * self.decay_factor).repeat(bs * 3)
        # masks[[*idxs]] = mw
        masks[[*idxs]] = 1

        # Apply a gaussian blur to the masks
        if self.blur_sigma > 0:
            ks = int(self.blur_sigma * 5)
            if ks % 2 == 0:
                ks += 1
            masks = gaussian_blur(masks, kernel_size=ks, sigma=self.blur_sigma.item())
            mask_maxs = torch.amax(masks, dim=(2, 3), keepdim=True)
            masks = torch.where(
                mask_maxs > 0,
                masks / mask_maxs,
                torch.zeros_like(masks)
            )

        return masks


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
