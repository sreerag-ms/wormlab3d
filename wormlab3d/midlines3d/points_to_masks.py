from typing import Any

import torch

from wormlab3d import PREPARED_IMAGE_SIZE, N_WORM_POINTS


class PointsToMasks(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, points_2d: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of 2D points in coordinate form and generates 2D segmentation masks.
        Creating the indices from the points involves casting to int/long which is non-differentiable,
        so we approximate the gradient with central differences.
        """
        bs = points_2d.shape[0]
        point_idxs_2d = points_2d.round().to(torch.long)  # this operation is non-differentiable!
        point_idxs_2d = point_idxs_2d.clamp(min=0, max=PREPARED_IMAGE_SIZE[0] - 1)

        # Make segmentation masks out of the 2d coordinates
        masks = torch.zeros((bs, 3, *PREPARED_IMAGE_SIZE))
        idxs = [
            torch.arange(bs).repeat_interleave(3 * N_WORM_POINTS),
            torch.arange(3).repeat_interleave(N_WORM_POINTS).repeat(bs),
            point_idxs_2d[:, :, :, 0].flatten(),
            point_idxs_2d[:, :, :, 1].flatten(),
        ]
        idxs = torch.stack(idxs)
        masks[[*idxs]] = 1
        ctx.save_for_backward(points_2d, idxs)

        return masks

    @staticmethod
    def backward(ctx: Any, masks_grad: torch.Tensor) -> torch.Tensor:
        """
        Given gradients calculated on the masks, calculate directional-derivatives in the x and y
        directions on the image using central differences. Then sample these gradients at the indices
        corresponding to the image points to give the coordinate-derivatives.
        """
        points_2d, idxs = ctx.saved_tensors
        bs = masks_grad.shape[0]
        img_size = masks_grad.shape[-1]

        # Calculate gradients in x- and y-directions using central difference formula
        pad_x = torch.zeros((bs, 3, 1, img_size))
        mgx_minus = torch.cat([pad_x, masks_grad[:, :, :-1]], dim=2)
        mgx_plus = torch.cat([masks_grad[:, :, 1:], pad_x], dim=2)
        masks_grad_x = N_WORM_POINTS * (mgx_plus - mgx_minus) / 2

        pad_y = torch.zeros((bs, 3, img_size, 1))
        mgy_minus = torch.cat([pad_y, masks_grad[:, :, :, :-1]], dim=3)
        mgy_plus = torch.cat([masks_grad[:, :, :, 1:], pad_y], dim=3)
        masks_grad_y = N_WORM_POINTS * (mgy_plus - mgy_minus) / 2

        # Find the gradients at the coordinates
        coord_shape = bs, 3, N_WORM_POINTS
        pixel_grads_x = masks_grad_x[[*idxs]].reshape(coord_shape)
        pixel_grads_y = masks_grad_y[[*idxs]].reshape(coord_shape)
        points_2d_grad = torch.stack([pixel_grads_x, pixel_grads_y], dim=-1)

        return points_2d_grad
