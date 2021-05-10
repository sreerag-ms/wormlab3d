from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _reverse_repeat_tuple
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import PREPARED_IMAGE_SIZE, N_WORM_POINTS
from wormlab3d.midlines3d.args.network_args import MAX_DECAY_FACTOR
from wormlab3d.toolkit.util import is_bad


class PointsToMasks(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            points_2d: torch.Tensor,
            blur_sigma: torch.Tensor,
            decay_factor: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes a batch of 2D points in coordinate form and generates 2D segmentation masks.
        Creating the indices from the points involves casting to int/long which is non-differentiable,
        so we approximate the gradient surface with central differences.
        """
        device = points_2d.device
        bs = points_2d.shape[0]
        point_idxs_2d = points_2d.round().to(torch.long)  # <-- this operation is non-differentiable!
        point_idxs_2d = point_idxs_2d.clamp(min=0, max=PREPARED_IMAGE_SIZE[0] - 1)

        # Make segmentation masks out of the 2d coordinates
        idxs = [
            torch.arange(bs).repeat_interleave(3 * N_WORM_POINTS),
            torch.arange(3).repeat_interleave(N_WORM_POINTS).repeat(bs),
            point_idxs_2d[:, :, :, 0].flatten(),
            point_idxs_2d[:, :, :, 1].flatten(),
        ]
        idxs = torch.stack([ix.to(device) for ix in idxs])

        # Write ones at the indexed locations.
        # Note - duplicate indices are intentionally not summed as the target mask only shows the likelihood
        # of the worm being in this pixel and in many cases the worm may be oriented directly at the camera.
        # todo: needs fixing now duplicated pixels have different values, we should use the max...
        masks = torch.zeros((bs, 3, *PREPARED_IMAGE_SIZE), device=device)
        mw = torch.exp(-torch.arange(N_WORM_POINTS, device=device) / N_WORM_POINTS * decay_factor).repeat(bs * 3)
        masks[[*idxs]] = mw

        # Apply a gaussian blur to the masks
        if blur_sigma > 0:
            ks = int(blur_sigma * 5)
            if ks % 2 == 0:
                ks += 1
            masks = gaussian_blur(masks, kernel_size=ks, sigma=blur_sigma.item())
            mask_maxs = torch.amax(masks, dim=(2, 3), keepdim=True)
            masks = torch.where(
                mask_maxs > 0,
                masks / mask_maxs,
                torch.zeros_like(masks)
            )

        # Save the idxs for the gradient calculation
        ctx.save_for_backward(idxs, decay_factor)

        return masks

    @staticmethod
    def backward(ctx: Any, masks_grad: torch.Tensor) -> torch.Tensor:
        """
        Given gradients calculated on the masks, calculate directional-derivatives in the x and y directions
        on the image using an approximation of central differences. Sample these gradients at the indices
        corresponding to the image points to give the gradients with respect to the coordinates.
        """
        idxs = ctx.saved_tensors[0]
        decay_factor = ctx.saved_tensors[1]
        bs = masks_grad.shape[0]

        # Adjust the gradient approximation kernels depending on the decay factor
        decay_factor_scaled = decay_factor / MAX_DECAY_FACTOR  # now between 0 and 1
        layers = []
        n_layers = 3 if decay_factor_scaled < 0.5 else 4
        w_max = 1
        w_min = 0.1
        ds_min = max(3, int(7 * decay_factor_scaled))
        ds_max = max(ds_min, int(50 * decay_factor_scaled))
        ss_min = max(1, int(5 * decay_factor_scaled))
        ss_max = max(ss_min, int(50 * decay_factor_scaled))

        # If decay factor is high then we really just need to find the worm, so use larger kernels
        for l in range(n_layers):
            layers.append({
                'weight': w_min + (w_max - w_min) / (n_layers - 1) * (n_layers - 1 - l),
                'directional_sigma': int(ds_min + (ds_max - ds_min) / (n_layers - 1) * l),
                'symmetric_sigma': int(ss_min + (ss_max - ss_min) / (n_layers - 1) * l),
            })

        # Convolve the image gradient with an anti-symmetric gaussian-derivative kernel to approximate
        # the directional derivatives in the x- and y- directions. Do this a number of times with
        # increasing sigmas to help build a smooth gradient map and take a weighted sum.
        mgx = torch.zeros_like(masks_grad)
        mgy = torch.zeros_like(masks_grad)
        for l_spec in layers:
            mgx_l = convolve_grad(
                masks_grad,
                directional_sigma=l_spec['directional_sigma'],
                directional_dim=0,
                symmetric_sigma=l_spec['symmetric_sigma']
            )
            mgy_l = convolve_grad(
                masks_grad,
                directional_sigma=l_spec['directional_sigma'],
                directional_dim=1,
                symmetric_sigma=l_spec['symmetric_sigma']
            )
            mgx = mgx + l_spec['weight'] * mgx_l
            mgy = mgy + l_spec['weight'] * mgy_l

        # Scale the gradients to account for rotations out of the plane
        grad_sums_per_view = masks_grad.sum(dim=(2, 3))
        grad_means_per_triplet = grad_sums_per_view.mean(dim=1, keepdim=True)
        grad_rel_differences = grad_sums_per_view - grad_means_per_triplet
        grad_rel_differences = F.normalize(grad_rel_differences, dim=1)
        scale_factors = torch.exp(-grad_rel_differences)
        scale_factors = scale_factors.reshape((bs, 3, 1, 1))
        scaled_masks_grad = masks_grad * scale_factors
        mgx = mgx * scaled_masks_grad
        mgy = mgy * scaled_masks_grad

        # Find the gradients at the coordinates
        coord_shape = bs, 3, N_WORM_POINTS
        pixel_grads_x = mgx[[*idxs]].reshape(coord_shape)
        pixel_grads_y = mgy[[*idxs]].reshape(coord_shape)
        points_2d_grad = torch.stack([pixel_grads_y, pixel_grads_x], dim=-1)

        # # Add some noise
        # noise = torch.normal(mean=torch.zeros_like(points_2d_grad), std=points_2d_grad.std() / 10)
        # points_2d_grad += noise

        # Adjust weightings so the network prioritises head-first
        decay = torch.exp(-torch.arange(N_WORM_POINTS, device=points_2d_grad.device) / N_WORM_POINTS * decay_factor)
        # decay_mid = torch.exp(-torch.arange(N_WORM_POINTS/2, device=points_2d_grad.device)/N_WORM_POINTS*20)
        # decay_mid = torch.cat([-decay_mid, decay_mid])
        # decay2 = 1-torch.exp(-torch.arange(N_WORM_POINTS, device=points_2d_grad.device)/N_WORM_POINTS*15)
        points_2d_grad_head = points_2d_grad * decay.reshape((1, 1, N_WORM_POINTS, 1))
        # points_2d_grad_tail = points_2d_grad * (1-decay.reshape((1, 1, N_WORM_POINTS, 1)))
        # points_2d_grad_mid = points_2d_grad * decay_mid.reshape((1, 1, N_WORM_POINTS, 1))
        # points_2d_grad = points_2d_grad_head + points_2d_grad_tail + points_2d_grad_mid
        points_2d_grad = points_2d_grad_head

        max_grad = 0.1
        points_2d_grad = points_2d_grad.clamp(min=-max_grad, max=max_grad)

        assert not is_bad(points_2d_grad)

        return points_2d_grad, None, None


def convolve_grad(
        grad: torch.Tensor,
        directional_dim: int,
        directional_sigma: int,
        symmetric_sigma: int
) -> torch.Tensor:
    """
    Central-difference approximation using convolution with a derivative-kernel.
    """

    # Make directional (anti-symmetric) kernel
    kernel_base = make_derivative_kernel(sigma=directional_sigma)
    kb = kernel_base.unsqueeze(directional_dim)
    kernel = torch.stack([kb, kb, kb])
    kernel = kernel.unsqueeze(1)

    # Make gaussian (symmetric) kernel
    if symmetric_sigma > 0:
        sym_kernel_base = make_gaussian_kernel(sigma=symmetric_sigma)
        skb = sym_kernel_base.unsqueeze(1 - directional_dim)
        sym_kernel = torch.stack([skb, skb, skb])
        sym_kernel = sym_kernel.unsqueeze(1)
        kernel = kernel * sym_kernel

    # Pad the gradient with a high value to push out-of-bound/boundary points towards the centre
    ks = kernel.shape
    padding = (ks[2] // 2, ks[3] // 2)
    _reversed_padding_repeated_twice = _reverse_repeat_tuple(padding, 2)
    oob_grad = grad.max()
    padded_grad = F.pad(grad, _reversed_padding_repeated_twice, mode='constant', value=oob_grad)

    # Do the convolution
    grad_conv = F.conv2d(
        input=padded_grad,
        weight=kernel.to(grad.device),
        stride=1,
        padding=0,
        groups=3
    )

    assert grad_conv.shape == grad.shape

    return grad_conv


def make_derivative_kernel(sigma: int):
    """Build the derivative kernel."""
    ks = max(3, sigma * 4)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    kernel = ts / sigma * torch.exp(- (ts / sigma)**2 / 2)
    kernel /= kernel.max()

    return kernel


def make_gaussian_kernel(sigma: int):
    """Build the gaussian kernel."""
    ks = max(3, sigma * 2)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    kernel = torch.exp(- (ts / sigma)**2 / 2)
    kernel /= kernel.max()

    return kernel


def plot_gaussian_kernel(sigma):
    kernel = make_gaussian_kernel(sigma)
    pad = len(kernel) // 2
    ts = np.arange(-pad, pad + 1)
    plt.plot(ts, kernel)

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid()
    plt.show()


def plot_derivative_kernel(sigma):
    kernel = make_derivative_kernel(sigma)
    pad = len(kernel) // 2
    ts = np.arange(-pad, pad + 1)
    plt.plot(ts, kernel)

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_kernel(sigma=0.1)
    plot_gaussian_kernel(sigma=1)
    plot_gaussian_kernel(sigma=2)
    plot_gaussian_kernel(sigma=5)
    plot_gaussian_kernel(sigma=10)
    plot_derivative_kernel(sigma=0.1)
    plot_derivative_kernel(sigma=1)
    plot_derivative_kernel(sigma=2)
    plot_derivative_kernel(sigma=5)
    plot_derivative_kernel(sigma=10)
