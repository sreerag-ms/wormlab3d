from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wormlab3d import N_WORM_POINTS, PREPARED_IMAGE_SIZE_DEFAULT
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, \
    ENCODING_MODE_DELTA_ANGLES, ENCODING_MODE_DELTA_ANGLES_BASIS, MAX_DECAY_FACTOR
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras
from wormlab3d.midlines3d.points_to_masks import PointsToMasks
from wormlab3d.nn.models.rednet import RedNet


class EncDec(nn.Module):
    def __init__(
            self,
            net: RedNet,
            blur_sigma: float = 0,
            distorted_cameras: bool = True,
            mode: str = ENCODING_MODE_DELTA_ANGLES,
            n_basis_fns: int = 0
    ):
        super().__init__()
        self.net = net
        self.cams = DynamicCameras(distort=distorted_cameras)
        self.mode = mode
        self.n_basis_fns = n_basis_fns

        # Points to masks todo: parametrise better
        self.pointsToMasks = PointsToMasks(
            blur_sigma=blur_sigma,
            n_points=N_WORM_POINTS,
            image_size=(PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
            max_n_optimisation_steps=500,
            oob_grad_val=1e-2,
            noise_std=0.
        )

        # Shifts
        self.register_buffer('shifts', None, persistent=False)

        # Basis coefficients
        self.register_buffer('basis_phi_A', None, persistent=False)
        self.register_buffer('basis_phi_p', None, persistent=False)
        self.register_buffer('basis_theta_A', None, persistent=False)
        self.register_buffer('basis_theta_p', None, persistent=False)

        # Delta angles
        self.register_buffer('delta_phis', None, persistent=False)
        self.register_buffer('delta_thetas', None, persistent=False)

        # Angles
        self.register_buffer('pre_angles', None, persistent=False)
        self.register_buffer('theta0', None, persistent=False)
        self.register_buffer('phi0', None, persistent=False)
        self.register_buffer('phis', None, persistent=False)
        self.register_buffer('thetas', None, persistent=False)

        if self.mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
            # Fix frequencies
            ws = [1 / 4, ]  # base frequency in units 2pi
            for n in range(self.n_basis_fns - 1):
                ws.append(ws[-1] * 2)
            w_n = torch.tensor(ws) * 2 * np.pi
            self.register_buffer('w_n', w_n)

            # Sample point locations
            t = torch.linspace(0, 1, N_WORM_POINTS - 1)
            self.register_buffer('w*t', torch.einsum('n,t->nt', w_n, t))

        # Determines which decoding method to use
        self.register_buffer('use_approx', torch.tensor(0, dtype=torch.bool), persistent=False)

        # Grow the worm out by slowly letting more gradients through
        self.register_buffer('decay_factor', torch.tensor(MAX_DECAY_FACTOR, dtype=torch.float32), persistent=True)

        # Track the best loss... so we know when to grow the worm more (wip)
        self.register_buffer('best_loss', torch.tensor(1e6, dtype=torch.float32), persistent=True)

    def set_decoder_mode(self, mode: str):
        """Set the decoder mode, requires a method as this net may be copied and distributed across devices"""
        assert mode in ['original', 'approx']
        self.use_approx.fill_(0 if mode == 'original' else 1)

    def set_decay_factor(self, decay_factor: float):
        """Set the decay factor, requires a method as this net may be copied and distributed across devices"""
        self.decay_factor.fill_(decay_factor)
        self.pointsToMasks.set_decay_factor(decay_factor)

    def forward(
            self,
            X: torch.Tensor,
            camera_coeffs: torch.Tensor,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use the neural network to generate an encoding (3D-coordinates) from the triplets of segmentation masks.
        Decode these coordinates by projecting down to triplets of 2D points and generating output masks.
        """
        bs = X.shape[0]
        device = X.device
        z_mu, z_log_var, z, disc, parameters, X_approx = self.net(X)

        # If using approximation then detach the parameters here
        if self.use_approx:
            parameters = parameters.detach().clone()

        # Extract xy shift/offset for each view
        shifts = parameters[:, :3 * 2].reshape((bs, 3, 2))
        parameters = parameters[:, 3 * 2:]

        # Determine maximum delta-angle
        max_delta_angle = 4 * np.pi / N_WORM_POINTS  # 2 complete revolutions
        # max_delta_angle = 2 * np.pi / N_WORM_POINTS  # 1 complete revolutions
        # max_delta_angle = 1 * np.pi / N_WORM_POINTS  # 1/2 complete revolutions
        worm_len = 0.7  # todo: learn or fetch from experiment

        if self.mode == ENCODING_MODE_DELTA_VECTORS:
            # Remaining parameters are the delta vectors (de0s)
            delta_vectors = parameters.reshape((bs, N_WORM_POINTS, 3))

            # Scale the ds's so that neighbouring points will be equidistant
            e0s = F.normalize(delta_vectors, dim=2) / N_WORM_POINTS

            # We don't know the delta angles
            delta_angles = None
        else:
            # Initial angles are unconstrained
            # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/11

            # theta: inclination - angle wrt z-axis: (0, pi)
            # phi: azimuth - rotation angle from x-y: (-pi, pi)
            pre_angles = parameters[:, :4]
            self.pre_angles = pre_angles
            # pre_angles = F.hardtanh(pre_angles, min_val=-1, max_val=1)
            theta0 = (torch.atan2(pre_angles[:, 0], pre_angles[:, 1]) + np.pi) / 2
            phi0 = torch.atan2(pre_angles[:, 2], pre_angles[:, 3])
            self.theta0 = theta0
            self.phi0 = phi0
            parameters = parameters[:, 4:]

            if self.mode == ENCODING_MODE_DELTA_ANGLES:
                # Remaining parameters are the delta angles
                delta_angles = torch.tanh(parameters.reshape((bs, 2, -1))) * max_delta_angle
            elif self.mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
                # Remaining parameters are the basis function coefficients; amplitudes and phases
                basis_coeffs = parameters.reshape((bs, 2, 2, -1))

                # Amplitudes can be positive or negative
                A_n = torch.tanh(basis_coeffs[:, 0])

                # Phases should be in range (-w/2, +w/2)
                p_n = torch.tanh(basis_coeffs[:, 1]) / self._buffers['w_n'].unsqueeze(0) / 2

                # Sum the basis functions to give the delta angles
                delta_angles = torch.sum(
                    A_n.unsqueeze(-1) * torch.cos(self._buffers['w*t'] + p_n.unsqueeze(-1)),
                    dim=2
                )

                # Rescale to ensure delta-angles are within [-1, +1]
                max_sizes = torch.amax(torch.abs(delta_angles), dim=-1, keepdim=True)
                scale_factors = max_sizes.clamp(min=1)
                delta_angles = delta_angles / scale_factors * max_delta_angle

                self.basis_theta_A = A_n[:, 0]
                self.basis_theta_p = p_n[:, 0]
                self.basis_phi_A = A_n[:, 1]
                self.basis_phi_p = p_n[:, 1]

            # Apply decay to delta angles so they go to 0 (ie, straight lines)
            decay = torch.exp(-torch.arange(N_WORM_POINTS, device=device) / N_WORM_POINTS * self.decay_factor)
            delta_angles = delta_angles * decay[1:].reshape((1, 1, N_WORM_POINTS - 1))

            # Sum the initial angles with the delta angles to give the progression
            delta_thetas = torch.cat([theta0.unsqueeze(1), delta_angles[:, 0]], dim=-1)
            delta_phis = torch.cat([phi0.unsqueeze(1), delta_angles[:, 1]], dim=-1)
            thetas = torch.cumsum(delta_thetas, dim=-1)
            phis = torch.cumsum(delta_phis, dim=-1)

            self.delta_phis = delta_phis
            self.delta_thetas = delta_thetas
            self.phis = phis
            self.thetas = thetas

            # Convert to cartesian coordinates to find the e0 unit vectors
            e0s = torch.stack([
                torch.cos(phis) * torch.sin(thetas),
                torch.sin(phis) * torch.sin(thetas),
                torch.cos(thetas),
            ], dim=-1)

        # Scale the e0s (which have unit length) so the arc length is fixed
        e0s_scaled = e0s * worm_len / N_WORM_POINTS

        # Start at (0,0,0) and add the scaled e0s's to form the curve
        curve_coordinates = torch.cumsum(e0s_scaled, dim=1)

        # Add the 3d centre point offset to centre it on the camera
        points_3d = points_3d_base.unsqueeze(1) + curve_coordinates

        # Project 3D points to 2D
        points_2d = self.cams.forward(camera_coeffs, points_3d)

        # Re-centre according to 2D base points plus a (100,100) to put it in the centre of the cropped image
        image_centre_pt = torch.ones((bs, 1, 1, 2), dtype=torch.float32, device=device) * PREPARED_IMAGE_SIZE_DEFAULT / 2
        points_2d_net = points_2d - points_2d_base.unsqueeze(2) + image_centre_pt

        # Apply 2D offsets, max shift allowed is within centre 50%
        self.shifts = shifts
        points_2d_net = points_2d_net + self.shifts.unsqueeze(2)

        # Generate the masks from the 2D image points and find a better set of points
        X_out, X_opt, points_2d_opt = self.pointsToMasks.forward(points_2d_net, X)

        return disc, camera_coeffs, points_3d, points_2d_net, points_2d_opt, X_out, X_opt, z_mu, z_log_var, e0s_scaled, delta_angles, X_approx

    def get_n_params(self) -> int:
        """Return from the encoder network."""
        return self.net.get_n_params()

    def calc_norms(self, p: int = 2) -> float:
        """Return from the encoder network."""
        return self.net.calc_norms(p=p)
