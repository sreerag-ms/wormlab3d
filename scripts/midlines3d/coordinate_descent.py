import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import *
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask
from wormlab3d.toolkit.util import to_numpy


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
        plt.scatter(points[:, 0], points[:, 1], s=20, c='red', marker='x')

    if show:
        plt.show()


def avg_grad(grad):
    # Cascade average pooling
    oob_grad = 0  #grad.max() + 1e-5
    padded_grad = F.pad(grad, (1, 1, 1, 1), mode='constant', value=oob_grad)
    # print('padded_grad.shape', padded_grad.shape)
    ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    # print('grad1.shape', grad1.shape)
    plot_mask(ag)
    return ag


def calculate_gradient_surface(mask_target):
    image_size = mask_target.shape[-2:]

    # Calculate gradient surface
    # grad0 = -mask_target
    grad0 = -torch.log(mask_target.clamp(min=1e-5))
    # grad0 = grad0.reshape((1, 1, *grad0.shape))
    plot_mask(grad0, 'grad0')
    # plot_mask(-torch.log(grad0), '-log(grad0)')
    # exit()

    grads = [grad0]
    g = grad0
    while g.shape[-1] > 1:
        g2 = avg_grad(g)
        # print('g2.shape', g2.shape)
        # plot_mask(g2, g2.shape)
        grads.append(g2)
        g = g2

    # Got all the grad averages, now add them together and average
    grad_sum = torch.zeros_like(grad0)
    for i, g in enumerate(grads):
        grad_sum += F.interpolate(grads[i], image_size, mode='bilinear', align_corners=False)
    grad_avg = grad_sum / len(grads)
    plot_mask(grad_avg, f'grad avg')
    # grad_avg = gaussian_blur(grad_avg, kernel_size=7, sigma=1)
    # grad_avg.zero_()

    # Calculate directional gradients
    gapx = F.pad(grad_avg, (0, 0, 1, 1), mode='replicate')
    gx = (gapx[:, :, :-2] - gapx[:, :, 2:]) / 2
    plot_mask(gx, 'gx')
    gapy = F.pad(grad_avg, (1, 1, 0, 0), mode='replicate')
    gy = (gapy[:, :, :, :-2] - gapy[:, :, :, 2:]) / 2
    plot_mask(gy, 'gy')

    J = torch.stack([gx, gy])

    return grad0, J


def get_idxs(points_2d, image_size):
    point_idxs_2d = points_2d.round().to(torch.long)  # <-- this operation is non-differentiable!
    point_idxs_2d = point_idxs_2d.clamp(min=0, max=image_size[0] - 1)
    n_points = points_2d.shape[2]
    idxs = [
        torch.arange(1).repeat_interleave(n_points),
        torch.arange(1).repeat_interleave(n_points),
        point_idxs_2d[:, :, :, 0].flatten(),
        point_idxs_2d[:, :, :, 1].flatten(),
    ]
    idxs = torch.stack([ix for ix in idxs])
    return idxs


def get_gradient_val(grad0, idxs, n_points):
    # Now determine direction of minimum gradient from each sample coordinate
    coord_shape = 1, 1, n_points
    pixel_grads_x = grad0[[*idxs]].reshape(coord_shape)
    pixel_grads_y = grad0[[*idxs]].reshape(coord_shape)
    points_2d_grad = torch.stack([pixel_grads_y, pixel_grads_x], dim=-1)
    # print('points_2d_grad.shape', points_2d_grad.shape)
    # print('points_2d_grad', points_2d_grad)

    return points_2d_grad


def get_directional_gradients(J, idxs, n_points):
    # Now determine direction of minimum gradient from each sample coordinate
    coord_shape = 1, 1, n_points
    pixel_grads_x = J[0][[*idxs]].reshape(coord_shape)
    pixel_grads_y = J[1][[*idxs]].reshape(coord_shape)
    points_2d_grad = torch.stack([pixel_grads_y, pixel_grads_x], dim=-1)
    # print('points_2d_grad.shape', points_2d_grad.shape)
    # print('points_2d_grad', points_2d_grad)

    return points_2d_grad


def test_points_to_masks():
    """
    """
    image_size = PREPARED_IMAGE_SIZE
    n_points = 3

    # Make target mask
    X_target = torch.tensor([
        [50, 50],
        [25, 100],
        [50, 150],
    ], dtype=torch.float32)
    mask_target = make_segmentation_mask(  # uses first coordinate as column and second as row
        X=X_target.numpy(),
        blur_sigma=1,
        draw_mode='line_aa',
        image_size=image_size
    )
    mask_target = torch.from_numpy(mask_target).reshape(1, 1, *image_size)
    # plot_mask(mask_target)

    # Make initial attempt
    p0 = torch.tensor([
        [25, 50],
        [50, 100],
        [25, 150],
    ], dtype=torch.float32).reshape(1, 1, 3, 2)

    # Calculate the gradient surface approximation
    grad0, J = calculate_gradient_surface(mask_target)

    # Set up the optimiser
    # optimiser = Adam(params=(p0,), lr=0.5)  # decent
    # optimiser = AdamW(params=(p0,), lr=0.5, betas=(0.1, 0.9),
    #                   amsgrad=False)  # seems to find a better trajectory than Adam
    # optimiser = Adamax(params=(p0,), lr=1)  # slower, similar traj to adam
    # optimiser = Adadelta(params=(p0,), lr=100)  # goes nowhere
    # optimiser = ASGD(params=(p0,), lr=2, lambd=1e-2, alpha=0.25)  # looks ok but crazy high lr and still only half way
    optimiser = SGD(params=(p0,), lr=1000)  # goes nowhere
    # optimiser = RMSprop(params=(p0,), lr=0.5)  # pretty good, goes quickly
    # optimiser = Rprop(params=(p0,), lr=0.5)  # decent
    # optimiser = SGD(params=(p0,), lr=1, momentum=0.9, nesterov=True)  # barely goes
    sf = torch.ones_like(p0)

    for i in range(1):
        # print(p0)
        mask_attempt = make_segmentation_mask(  # uses first coordinate as column and second as row
            X=p0.numpy().squeeze(),
            blur_sigma=1,
            draw_mode='line_aa',
            image_size=image_size
        ).reshape(1, 1, *image_size)
        mask_attempt = torch.from_numpy(mask_attempt)
        plot_mask(mask_target)
        mask_diff = mask_attempt - mask_target  # .numpy()
        plot_mask(mask_diff, f'grad{i}', p0, show=False)

        for j in range(500):
            idxs = get_idxs(p0, image_size)
            grads = get_directional_gradients(J, idxs, n_points)
            optimiser.zero_grad()
            p0.grad = -grads   #* sf
            optimiser.step()

            # update sf
            # print(grads)
            # gv = get_gradient_val(mask_diff, idxs, n_points)
            # print(gv)
            # sf = torch.exp(gv)
            # print(sf)
            # exit()

            # Add noise

            # Plot the new points
            p0n = to_numpy(p0).squeeze()
            plt.scatter(p0n[:, 0], p0n[:, 1], s=20, c='red', marker='x')

        plt.show()


if __name__ == '__main__':
    test_points_to_masks()
