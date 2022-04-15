import functools
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Uniform

from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras, N_CAM_COEFFICIENTS
from wormlab3d.nn.models.basenet import BaseNet


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def render(
        points_in,
        size,
        blur_sigma
):
    # Reshape
    points = points_in.clone()

    bs = points.shape[0]
    n_worm_points = points.shape[-1]

    points = points.reshape(bs, 3, 2, n_worm_points)
    points = points.permute(0, 1, 3, 2)
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)
    a = points[:, :-1]
    b = points[:, 1:]

    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    grid = torch.linspace(-1.0, 1.0, size, dtype=torch.float32, device=a.device)

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
    d_norm = torch.exp(-d / (blur_sigma**2))

    # Normalise
    max_d = d_norm.amax(dim=(2, 3), keepdim=True)
    d_norm = torch.where(max_d <= 0, torch.zeros_like(d), d_norm / max_d)

    # d_norm = d_norm / torch.sum(d_norm, (2, 3), keepdim=True)
    d_norm = d_norm.sum(dim=1)
    d_norm = d_norm.clamp(max=1)

    # Reshape back to the triplets
    d_norm = d_norm.reshape(bs, 3, size, size)

    return d_norm


def _avg_pool_2d(X_, oob_val=0.):
    # Average pooling with overlap and boundary values
    padded_grad = F.pad(X_, (1, 1, 1, 1), mode='constant', value=oob_val)
    ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


def _avg_surface_2d(X_: torch.Tensor):
    G = -X_
    GS = [G]
    g = G
    while g.shape[-1] > 1:
        g2 = _avg_pool_2d(g, oob_val=0)
        GS.append(g2)
        g = g2

    # Got all the grad averages, now add them together and average
    G_sum = torch.zeros_like(G)
    for i, g in enumerate(GS):
        G_sum += F.interpolate(GS[i], (PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
                               mode='bilinear', align_corners=False)
    G_avg = G_sum  # / len(GS)

    return G_avg


class RenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            points_in,
            render_target,
            size,
            blur_sigma
    ):
        # ctx.points_in = points_in
        # ctx.render_target = render_target
        # ctx.size = size
        # ctx.blur_sigma = blur_sigma

        device = points_in.device
        ctx.save_for_backward(points_in, render_target, torch.tensor(size, device=device),
                              torch.tensor(blur_sigma, device=device))

        render_out = render(points_in, size, blur_sigma)

        return render_out

    @staticmethod
    def backward(ctx, render_grad):
        bs = render_grad.shape[0]
        device = render_grad.device
        w = render_grad.abs().mean(dim=(2, 3))
        points_in, render_target, size, blur_sigma = ctx.saved_tensors
        if w.sum() == 0:
            return torch.zeros_like(points_in), None, None, None

        points_opt = points_in.detach()
        # blur_sigma = ctx.blur_sigma
        # blur_sigma = torch.tensor(ctx.blur_sigma, device=device)

        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        N = points_opt.shape[-1]
        # cmap = cm.get_cmap('plasma')
        # fc = cmap((np.arange(N) + 0.5) / N)

        # points_opt.requires_grad = True
        n_opt_steps = 10
        patch_sizes = [5, 11, 23, 47]

        # render_target = ctx.render_target.detach()
        render_target = -1 * _avg_surface_2d(render_target.detach())
        # rts = render_target.sum(dim=(2,3))

        for s in range(n_opt_steps):
            render_out = render(points_opt, size, blur_sigma)
            render_out = -1 * _avg_surface_2d(render_out)

            # grad_out = (render_out-render_target)**2
            diff = render_out - render_target

            # find minimum points for each point
            p2 = torch.round(points_opt * 100 + 100).to(torch.long)
            p2 = p2.reshape(bs, 3, 2, N)
            point_targets = torch.zeros_like(p2)
            for i in range(bs):
                for j in range(3):
                    if w[i, j] == 0:
                        continue
                    for k in range(N):
                        best = np.inf
                        best_loc = torch.tensor([0, 0], device=device)
                        for p in patch_sizes:
                            p2a = p2[i, j, :, k]

                            patch = diff[i, j,
                                    max(0, p2a[1] - p // 2):min(201, p2a[1] + p // 2 + 1),
                                    max(0, p2a[0] - p // 2):min(201, p2a[0] + p // 2 + 1)]
                            try:
                                p_min = patch.min()
                            except Exception as e:
                                continue

                            if p_min < best:
                                best = p_min
                                rel_pos = (patch == p_min).nonzero()[0] - torch.tensor([p // 2, p // 2], device=device)
                                best_loc = torch.flip(rel_pos, dims=(0,)) + p2a

                            if best < -0.3:
                                break

                        point_targets[i, j, :, k] = best_loc

            point_targets_fixed = ((point_targets - 100) / 100).reshape(bs, 6, N)
            points_grad = points_opt - point_targets_fixed
            # loss = grad_out.sum()

            # if s % 5 == 0:
            #     fig, axes = plt.subplots(3, figsize=(7, 12))
            #
            #     ax = axes[0]
            #     ax.imshow(render_target[0,0].detach().numpy())
            #     ax.set_title('target')
            #     ax.axis('off')
            #
            #     ax = axes[2]
            #     ax.imshow(render_out[0,0].detach().numpy())
            #     ax.scatter(points[0,0].detach().numpy()*100+100, points[0,1].detach().numpy()*100+100, s=15, label='orig', c=fc, marker='o')
            #     ax.scatter(point_targets_fixed[0,0].detach().numpy()*100+100, point_targets_fixed[0,1].detach().numpy()*100+100, s=15, label='targ', c='red', marker='x')
            #     ax.legend()
            #     ax.set_title('output')
            #     ax.axis('off')
            #
            #     ax = axes[1]
            #     ax.imshow(diff[0,0].detach().numpy())
            #     # plt.imshow(ctx.render_target[0,0].detach().numpy())
            #     # ax.scatter(points[0,0].detach().numpy()*100+100, points[0,1].detach().numpy()*100+100, s=3, label='pre', c=fc, marker='o')
            #     ax.axis('off')
            #     ax.set_title(f'i={s} loss={loss:.3f} sig={blur_sigma:.4f}')

            # Take gradient step
            points_opt = points_opt.detach() - 1e-1 * points_grad

            # if s % 5 == 0:
            #     ax.scatter(points[0,0].detach().numpy()*100+100, points[0,1].detach().numpy()*100+100, s=15, label='post', c=fc, marker='+')
            #     # ax.legend()
            #     fig.tight_layout()
            #     plt.show()

        w_exp = torch.repeat_interleave(w, 2, dim=1)
        points_in_grad = (points_in - points_opt) * w_exp[:, :, None]

        grads = {
            'points_in': -points_in_grad,
            'render_target': None,
            'size': None,
            'blur_sigma': None
        }

        return tuple(grads.values())


class RotAENet(nn.Module):
    def __init__(
            self,
            c2d_net: BaseNet,
            c3d_net: BaseNet,
            n_worm_points: int,
            cam_coeffs_mean: torch.Tensor,
            cam_coeffs_range: torch.Tensor,
            p3d_mean: torch.Tensor,
            p3d_range: torch.Tensor,
            p2d_mean: torch.Tensor,
            p2d_range: torch.Tensor,
            distorted_cameras: bool = True,
            blur_sigma: float = 0.2,
            max_rotation: float = 0.,
    ):
        super().__init__()
        self.c2d_net = c2d_net
        self.c3d_net = c3d_net
        self.n_worm_points = n_worm_points
        self.register_buffer('cam_coeffs_mean', cam_coeffs_mean)
        self.register_buffer('cam_coeffs_range', cam_coeffs_range)
        self.register_buffer('p3d_mean', p3d_mean)
        self.register_buffer('p3d_range', p3d_range)
        self.register_buffer('p2d_mean', p2d_mean)
        self.register_buffer('p2d_range', p2d_range)
        self.cams = DynamicCameras(distort=distorted_cameras)
        self.register_buffer('rng_low', torch.zeros(3))
        self.register_buffer('rng_high', torch.ones(3) * 2 * np.pi / 360)
        self.euler_angles = None
        self.rotation_matrix = None
        self.size = PREPARED_IMAGE_SIZE_DEFAULT
        self.blur_sigma = blur_sigma
        self.max_rotation = max_rotation

    def _get_rng(self) -> Uniform:
        return Uniform(
            low=self.rng_low,
            high=self.rng_high * self.max_rotation
        )

    def forward(
            self,
            X0: torch.Tensor,
            camera_coeffs: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use the neural network to generate flattened coordinates from individual images.
        Reshape these into 2D coordinates and centre on the image.
        """
        bs = X0.shape[0]

        # Normalise the camera coefficients and base points
        cc = (camera_coeffs - self.cam_coeffs_mean) / (self.cam_coeffs_range + 1e-7)
        p3d = (points_3d_base - self.p3d_mean) / (self.p3d_range + 1e-7)
        p2d = (points_2d_base - self.p2d_mean) / (self.p2d_range + 1e-7)

        # Include the camera coeffs and base points as input
        setup = torch.cat([cc.view(bs, -1), p3d.view(bs, -1), p2d.view(bs, -1)], dim=1)
        setup_exp = setup[..., None, None]
        setup_exp = setup_exp.expand(bs, setup.shape[1], X0.shape[-2], X0.shape[-1])
        c2d_input = torch.cat([X0, setup_exp], dim=1)

        # Use c2d network to generate 2D coordinates (X1) and setup adjustments
        c2d_output = self.c2d_net(c2d_input)
        X1a = c2d_output[:, :3].reshape(bs, 3 * 2, self.n_worm_points)  # [-1,+1]
        X1 = X1a * X0.shape[-1] / 2  # [-100,100]

        # Render coordinates
        W0 = self.render(X1 / (X0.shape[-1] / 2))
        # W0 = render(X1 / (X0.shape[-1] / 2), self.size, self.blur_sigma)
        # W0 = RenderFunction.apply(
        #     X1 / (X0.shape[-1] / 2),  # points_in
        #     X0,
        #     self.size,
        #     self.blur_sigma
        # )

        # Update setup
        setup_adj = c2d_output[:, 3:].mean(dim=(2, 3))
        setup = setup + torch.tanh(setup_adj) * 1e-3
        cc2 = setup[:, :N_CAM_COEFFICIENTS * 3].reshape_as(camera_coeffs)
        cc2 = cc2 * (self.cam_coeffs_range + 1e-7) + self.cam_coeffs_mean
        p3d2 = setup[:, N_CAM_COEFFICIENTS * 3:N_CAM_COEFFICIENTS * 3 + 3].reshape_as(points_3d_base)
        p3d2 = p3d2 * (self.p3d_range + 1e-7) + self.p3d_mean
        p2d2 = setup[:, -3 * 2:].reshape_as(points_2d_base)
        p2d2 = p2d2 * (self.p2d_range + 1e-7) + self.p2d_mean

        # Expand the new setup along the worm length
        setup2 = torch.cat([cc2.view(bs, -1), p3d2.view(bs, -1), p2d2.view(bs, -1)], dim=1)
        setup_exp2 = setup2[..., None]
        setup_exp2 = setup_exp2.expand(bs, setup2.shape[1], self.n_worm_points)

        # Lift the 2D coordinates to 3D
        c3d_input = torch.cat([X1, setup_exp2], dim=1)
        X2 = self.c3d_net(c3d_input)

        # Rotate
        if self.max_rotation > 0:
            Z2 = self.rotate(X2)
        else:
            Z2 = X2

        # Project 3D points to 2D using adjusted cameras
        Z2a = Z2 + p3d2.unsqueeze(2)
        Z1a = self.cams.forward(cc2, Z2a.permute(0, 2, 1))
        Z1b = Z1a - p2d2.unsqueeze(2)
        Z1c = Z1b.permute(0, 1, 3, 2)
        Z1 = Z1c.reshape(bs, 3 * 2, self.n_worm_points)

        # Lift back into 3D
        c3d_input = torch.cat([Z1, setup_exp2], dim=1)
        Z2 = self.c3d_net(c3d_input)

        # Un-rotate to recover reconstructed X2
        if self.max_rotation > 0:
            Y2 = self.unrotate(Z2)
        else:
            Y2 = Z2

        # Project to get the reconstructed X1
        Y2a = Y2 + p3d2.unsqueeze(2)
        Y1a = self.cams.forward(cc2, Y2a.permute(0, 2, 1))
        Y1b = Y1a - p2d2.unsqueeze(2)
        Y1c = Y1b.permute(0, 1, 3, 2)
        Y1 = Y1c.reshape(bs, 3 * 2, self.n_worm_points)

        # Render coordinates to get reconstructed X0
        Y0 = self.render(Y1 / (X0.shape[-1] / 2))  # - 1)
        # Y0 = RenderFunction.apply(
        #     Y1 / (X0.shape[-1] / 2),  # points_in
        #     X0,
        #     self.size,
        #     self.blur_sigma
        # )

        return W0, X1, X2, Y0, Y1, Y2

    def rotate(self, points_3d: torch.Tensor) -> torch.Tensor:
        # Generate Euler rotation angles and matrix
        rng = self._get_rng()
        self.euler_angles = rng.sample((points_3d.shape[0],))
        self.rotation_matrix = euler_angles_to_matrix(self.euler_angles, convention='XYZ')

        # Rotate
        return torch.einsum('buv,bvn->bun', self.rotation_matrix, points_3d)

    def unrotate(self, points_3d: torch.Tensor) -> torch.Tensor:
        return torch.einsum('buv,bvn->bun', self.rotation_matrix.transpose(1, 2), points_3d)

    def render(self, points):
        # Reshape
        bs = points.shape[0]
        points = points.reshape(bs, 3, 2, self.n_worm_points)
        points = points.permute(0, 1, 3, 2)
        points = points.reshape(bs * 3, self.n_worm_points, 2)
        points = points.clamp(min=-1, max=1)
        a = points[:, :-1]
        b = points[:, 1:]

        def sumprod(x, y, keepdim=True):
            return torch.sum(x * y, dim=-1, keepdim=keepdim)

        grid = torch.linspace(-1.0, 1.0, self.size, dtype=torch.float32, device=a.device)

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
        d_norm = torch.exp(-d / (self.blur_sigma**2))

        # Normalise
        max_d = d_norm.amax(dim=(2, 3), keepdim=True)
        d_norm = torch.where(max_d <= 0, torch.zeros_like(d), d_norm / max_d)

        # d_norm = d_norm / torch.sum(d_norm, (2, 3), keepdim=True)
        d_norm = d_norm.sum(dim=1)
        d_norm = d_norm.clamp(max=1)

        # Reshape back to the triplets
        d_norm = d_norm.reshape(bs, 3, self.size, self.size)

        return d_norm

    def get_n_params(self) -> int:
        """Return from the encoder network."""
        return self.c2d_net.get_n_params()

    def calc_norms(self, p: int = 2) -> float:
        """Return from the encoder network."""
        return self.c2d_net.calc_norms(p=p)


def check_rotation():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from simple_worm.plot3d import interactive

    N = 50
    rng = Uniform(low=torch.zeros(3), high=torch.ones(3) * 2 * np.pi)

    # Midline as a spiral
    x0 = torch.zeros((3, N))
    x0[0] = torch.sin(2 * np.pi * torch.linspace(start=0, end=1, steps=N)) / 10
    x0[1] = torch.cos(2 * np.pi * torch.linspace(start=0, end=1, steps=N)) / 10
    x0[2] = torch.linspace(start=1 / np.sqrt(3), end=0, steps=N)
    x0 = x0.transpose(0, 1)

    # Generate Euler rotation angles and matrix
    euler_angles = rng.sample()
    rotation_matrix = euler_angles_to_matrix(euler_angles, convention='XYZ')

    # Rotate
    x1 = torch.einsum('uv,bv->bu', rotation_matrix, x0)

    # Unrotate
    x2 = torch.einsum('uv,bv->bu', rotation_matrix.transpose(0, 1), x1)

    assert np.allclose(x0, x2, atol=1e-7)

    interactive()
    fig = plt.figure()

    # Colourmap / facecolors
    cmap = cm.get_cmap('rainbow')
    fc = cmap((np.arange(N) + 0.5) / N)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    x0 = x0.transpose(0, 1)
    ax.scatter(x0[0], x0[1], x0[2], c=fc)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    x1 = x1.transpose(0, 1)
    ax.scatter(x1[0], x1[1], x1[2], c=fc)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    x2 = x2.transpose(0, 1)
    ax.scatter(x2[0], x2[1], x2[2], c=fc)

    plt.show()


def check_rendering():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from wormlab3d.nn.models.fcnet import FCNet

    N = 50

    c2d_net = FCNet(
        input_shape=(3, 100, 100),
        output_shape=(3, N, 2),
        layers_config=(20, 20),
        build_model=False,
        act_out=None
    )

    c3d_net = FCNet(
        input_shape=(3, N, 2),
        output_shape=(N, 3),
        layers_config=(20, 20),
        build_model=False,
        act_out=None
    )

    net = RotAENet(
        c2d_net=c2d_net,
        c3d_net=c3d_net,
        n_worm_points=N,
        blur_sigma=0.1
    )

    # Dummy coords
    Y1 = torch.zeros((3, N, 2))
    for i in range(3):
        Y1[i, :, i % 2] = torch.linspace(start=-1, end=1, steps=N)
        Y1[i, :, (i + 1) % 2] = torch.sin(2 * np.pi * torch.linspace(start=0, end=1, steps=N))

    # Render
    Y0 = net.render(Y1.unsqueeze(0))

    cmap = cm.get_cmap('rainbow')
    fc = cmap((np.arange(N) + 0.5) / N)

    fig, axes = plt.subplots(2, 3)

    for i in range(3):
        ax = axes[0, i]
        x = Y1.squeeze()[i]
        ax.scatter(x[:, 0], x[:, 1], c=fc)

        ax = axes[1, i]
        x = Y0.squeeze()[i]
        ax.matshow(x)

    plt.show()


if __name__ == '__main__':
    # check_rotation()
    check_rendering()
