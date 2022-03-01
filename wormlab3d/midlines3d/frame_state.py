from typing import Dict, Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import CAMERA_IDXS, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import Cameras, MFParameters, Frame
from wormlab3d.midlines3d.mf_methods import make_rotation_matrix

PARAMETER_NAMES = [
    'cam_intrinsics',
    'cam_rotation_preangles',
    'cam_translations',
    'cam_distortions',
    'cam_shifts',
    'points',
    'sigmas',
    'exponents',
    'intensities',
    'camera_sigmas',
    'camera_exponents',
    'camera_intensities',
]

CAM_PARAMETER_NAMES = [
    'intrinsics',
    'rotation_preangles',
    'translations',
    'distortions',
    'shifts',
]

BUFFER_NAMES = [
    'masks_target',
    'masks_target_residuals',
    'cam_rotations',
    'points_2d',
    'masks_curve',
    'scores',
    'curve_lengths'
]

TRANSIENTS_NAMES = [
    'images',
    'cam_coeffs_db',
    'points_3d_base',
    'points_2d_base',
]

BINARY_DATA_KEYS = []


def _extract_camera_coefficients(cameras: Cameras) -> torch.Tensor:
    """
    Load the camera coefficients from the database object.
    """
    fx = np.array([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
    fy = np.array([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
    cx = np.array([cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
    cy = np.array([cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
    R = np.array([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
    t = np.array([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
    d = np.array([cameras.distortion[c] for c in CAMERA_IDXS])
    s = np.array([[0, ]] * 3)
    cam_coeffs = np.concatenate([
        fx.reshape(3, 1), fy.reshape(3, 1), cx.reshape(3, 1), cy.reshape(3, 1), R.reshape(3, 9), t, d, s
    ], axis=1).astype(np.float32)
    return torch.from_numpy(cam_coeffs)


class FrameState(nn.Module):
    def __init__(
            self,
            frame: Frame,
            parameters: MFParameters,
            prev_frame_state: 'FrameState' = None,
            master_frame_state: 'FrameState' = None,
    ):
        super().__init__()
        self.parameters = parameters
        self.frame = frame
        self.frame_num = frame.frame_num
        self.is_frozen = False

        # Register buffers
        self._init_images()
        self._init_cameras()
        self._init_base_points()
        self._init_masks_targets()

        # If we are using a master frame state then just reference those parameters.
        if master_frame_state is not None:
            for k in PARAMETER_NAMES:
                self.register_parameter(k, master_frame_state.get_state(k))

        # Otherwise, initialise fully
        else:
            self._init_parameters()
            self._init_cam_coeffs()

        # Initialise outputs
        self._init_outputs()
        self.stats: Dict[str, torch.Tensor] = {}

        # Copy state over from previous state if passed
        if prev_frame_state is not None:
            with torch.no_grad():
                for k in PARAMETER_NAMES:
                    self.set_state(k, prev_frame_state.get_state(k))
                for k in ['masks_curve', 'points_2d', 'curve_lengths', 'scores']:
                    self.set_state(k, prev_frame_state.get_state(k))

    def _init_images(self):
        """
        Load images.
        """
        if self.frame.images is None or len(self.frame.images) != 3:
            raise RuntimeError('Frame does not have a triplet of prepared images. Aborting!')
        images = torch.from_numpy(np.stack(self.frame.images))
        self.register_buffer('images', images)

    def _init_cameras(self):
        """
        Load cameras.
        """
        cameras: Cameras = self.frame.get_cameras()
        self.register_buffer('cam_coeffs_db', _extract_camera_coefficients(cameras))

    def _init_base_points(self):
        """
        Load the 3D and reprojected 2D base points.
        """
        # Use the fixed (interpolated and smoothed) centre point if available
        if self.frame.centre_3d_fixed is not None:
            p3d = self.frame.centre_3d_fixed
        else:
            p3d = self.frame.centre_3d
        self.register_buffer('points_3d_base', torch.tensor(p3d.point_3d))
        self.register_buffer('points_2d_base', torch.tensor(p3d.reprojected_points_2d))

    def _init_masks_targets(self):
        """
        Generate the target masks at all depths.
        """

        # Threshold images to make the full-resolution target masks
        masks_fr = self.images / self.images.amax(dim=(1, 2), keepdim=True)
        masks_fr[self.images < self.parameters.masks_threshold] = 0
        masks_fr = masks_fr.unsqueeze(0)

        # Linearly interpolate target mask resolutions from 8x8 to 200x200
        sizes = torch.linspace(8, PREPARED_IMAGE_SIZE[0], self.parameters.depth).to(torch.int32)

        # Generate downsampled target masks
        for d in range(self.parameters.depth):
            image_size = sizes[d]

            if d < self.parameters.depth - 1:
                # Downsample the full resolution version to the reduced size
                masks_ds = F.interpolate(masks_fr, (image_size, image_size), mode='nearest')

                # Upsample to restore the target size
                masks_rs = F.interpolate(masks_ds, PREPARED_IMAGE_SIZE, mode='bilinear', align_corners=False)

                # Add a gaussian-blur, smaller at lower depths.
                blur_sigma = 1 / (2**(d + 1))
                ks = int(blur_sigma * 5)
                if ks % 2 == 0:
                    ks += 1
                masks_rs = gaussian_blur(masks_rs, kernel_size=ks, sigma=blur_sigma)
            else:
                masks_rs = masks_fr

            masks_rs = masks_rs.squeeze(0)
            self.register_buffer(f'masks_target_{d}', masks_rs)

    def _init_parameters(self):
        """
        Initialise the multiscale curve parameters.
        """
        mp = self.parameters

        # Initialise the points perfectly spaced along a line of random length.
        curve_points = []
        for d in range(mp.depth):
            if d == 0:
                x = torch.normal(mean=torch.zeros(3), std=1 / (2**(d + 4)))
                x = x.unsqueeze(0)
            elif d == 1:
                x = torch.normal(mean=torch.cat([x, x]), std=1 / (2**(d + 4)))
            else:
                r = (x[1:] - x[:-1]).mean(dim=0, keepdim=True)
                r = r[0] / 3
                x = torch.stack([
                    torch.linspace(float(x[0, j] - r[j]), float(x[-1, j] + r[j]), 2**d)
                    for j in range(3)
                ], axis=1)
            x = nn.Parameter(x, requires_grad=True)
            curve_points.append(x)
        self.register_parameter('points', curve_points)

        # Initialise the sigmas equally at each level, decreasing with depth
        sigmas = []
        for d in range(mp.depth):
            sigmas_d = torch.ones(2**d) * mp.sigmas_init / (2**d)
            sigmas_d = nn.Parameter(sigmas_d, requires_grad=mp.optimise_sigmas)
            sigmas.append(sigmas_d)
        self.register_parameter('sigmas', sigmas)

        # Initialise the exponents all to 1
        exponents = []
        for d in range(mp.depth):
            exponents_d = torch.ones(2**d)
            exponents_d = nn.Parameter(exponents_d, requires_grad=mp.optimise_exponents)
            exponents.append(exponents_d)
        self.register_parameter('exponents', exponents)

        # Initialise the intensities all to 1
        intensities = []
        for d in range(mp.depth):
            intensities_d = torch.ones(2**d)
            intensities_d = nn.Parameter(intensities_d, requires_grad=mp.optimise_intensities)
            intensities.append(intensities_d)
        self.register_parameter('intensities', intensities)

        # Camera sigma, exponent and intensity scale factors
        self.register_parameter(
            'camera_sigmas',
            nn.Parameter(torch.ones(3), requires_grad=mp.optimise_sigmas)
        )
        self.register_parameter(
            'camera_exponents',
            nn.Parameter(torch.ones(3), requires_grad=mp.optimise_exponents)
        )
        self.register_parameter(
            'camera_intensities',
            nn.Parameter(torch.ones(3), requires_grad=mp.optimise_intensities)
        )

    def _init_cam_coeffs(self):
        """
        Initialise the camera coefficients with database values.
        """
        mp = self.parameters
        cc = self.cam_coeffs_db.clone().detach()

        self.register_parameter(
            'cam_intrinsics',
            nn.Parameter(cc[:, :4], requires_grad=mp.optimise_cam_coeffs and mp.optimise_cam_intrinsics)
        )
        self.register_parameter(
            'cam_translations',
            nn.Parameter(cc[:, 13:16], requires_grad=mp.optimise_cam_coeffs and mp.optimise_cam_translations)
        )
        self.register_parameter(
            'cam_distortions',
            nn.Parameter(cc[:, 16:21], requires_grad=mp.optimise_cam_coeffs and mp.optimise_cam_distortions)
        )
        self.register_parameter(
            'cam_shifts',
            nn.Parameter(cc[:, 21].unsqueeze(-1), requires_grad=mp.optimise_cam_coeffs and mp.optimise_cam_shifts)
        )

        # Extract Euler rotation angles from the rotation matrices
        R = cc[:, 4:13].reshape((3, 3, 3))
        rotation_preangles = torch.zeros(3, 3, 2)
        Rs = []

        for i in range(3):
            Ri = R[i]
            if Ri[2, 0] != 1 and Ri[2, 0] != -1:
                theta = -torch.arcsin(Ri[2, 0])
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                psi = torch.atan2(
                    Ri[2, 1] / cos_theta,
                    Ri[2, 2] / cos_theta
                )

                phi = torch.atan2(
                    Ri[1, 0] / cos_theta,
                    Ri[0, 0] / cos_theta,
                )
            else:
                phi = 0
                if Ri[2, 0] == -1:
                    theta = np.pi / 2
                    psi = phi + torch.atan2(Ri[0, 1], Ri[0, 2])
                else:
                    theta = -np.pi / 2
                    psi = -phi + torch.atan2(-Ri[0, 1], -Ri[0, 2])

                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

            cos_psi = torch.cos(psi)
            sin_psi = torch.sin(psi)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            rotation_preangles[i] = torch.tensor([
                [cos_phi, sin_phi],
                [cos_theta, sin_theta],
                [cos_psi, sin_psi],
            ])

            Ri2 = make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi)
            assert torch.allclose(Ri, Ri2, atol=1e-3)
            Rs.append(Ri2.flatten())

        # Optimise just the rotation angles, not the full rotation matrix.
        self.register_parameter(
            'cam_rotation_preangles',
            nn.Parameter(rotation_preangles, requires_grad=mp.optimise_cam_coeffs and mp.optimise_cam_rotations)
        )

    def _init_outputs(self):
        """
        Initialise the output buffers.
        """
        D = self.parameters.depth

        # Camera rotation matrices (flattened)
        self.register_buffer('cam_rotations', torch.zeros(3, 9))

        # Setup buffers for the outputs
        mt = self.get_state('masks_target')
        for d in range(D):
            self.register_buffer(f'points_2d_{d}', torch.zeros(2**d))
            self.register_buffer(f'masks_curve_{d}', torch.zeros_like(mt[d]))
            self.register_buffer(f'masks_target_residuals_{d}', torch.zeros_like(mt[d]))
            self.register_buffer(f'scores_{d}', torch.zeros(2**d))

        # Curve lengths
        self.register_buffer('curve_lengths', torch.zeros(D))

    def register_parameter(
            self,
            name: str,
            param: Optional[Union[List[nn.Parameter], nn.Parameter]]
    ) -> None:
        """
        Override to allow setting of a parameter with a list defining the parameter at each depth.
        """
        if type(param) == list:
            param_list = []
            for d, p in enumerate(param):
                self.register_parameter(f'{name}_{d}', p)
                param_list.append(self._parameters[f'{name}_{d}'])
            setattr(self, name, param_list)
        else:
            return super().register_parameter(name, param)

    def get_state(self, key: str) -> torch.Tensor:
        """
        Retrieve a buffer or parameter.
        """
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._buffers:
            return self._buffers[key]
        elif key == 'cam_coeffs':
            return self.get_cam_coeffs()
        elif key in ['masks_target', 'masks_target_residuals', 'masks_curve', 'points_2d', 'scores']:
            return [self._buffers[f'{key}_{d}'] for d in range(self.parameters.depth)]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise RuntimeError(f'Could not get state for {key}.')

    def set_state(self, key: str, data: torch.Tensor):
        """
        Set a buffer or parameter value.
        """
        if type(data) == list:
            params = self.get_state(key)
            assert type(params) == list and len(data) == len(params)
            for i in range(len(data)):
                params[i].data = data[i].clone()
        elif key in self._parameters:
            self._parameters[key].data = data.clone()
        elif key in self._buffers:
            self._buffers[key].data = data.clone()
        else:
            raise RuntimeError(f'Could not set state for {key}.')

    def get_cam_coeffs(self) -> torch.Tensor:
        """
        Build rotation matrices and collate all the camera coefficients together.
        """
        Rs = []
        rotation_preangles = self._parameters['cam_rotation_preangles']
        for i in range(3):
            pre = rotation_preangles[i]
            cos_phi, sin_phi = pre[0]
            cos_theta, sin_theta = pre[1]
            cos_psi, sin_psi = pre[2]
            Ri = make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi)
            Rs.append(Ri.flatten())
        Rs = torch.stack(Rs)
        self.set_state('cam_rotations', Rs)

        return torch.cat([
            self.cam_intrinsics,
            Rs,
            self.cam_translations,
            self.cam_distortions,
            self.cam_shifts,
        ], dim=1)

    def set_stats(self, stats: Dict[str, torch.Tensor]):
        """
        Set frame stats.
        """
        self.stats = {**self.stats, **stats}

    def freeze(self):
        """
        Freeze the frame - turns off requires_grad for all parameters.
        """
        self.is_frozen = True
        for key in PARAMETER_NAMES:
            p = self.get_state(key)
            if type(p) == list:
                for pi in p:
                    pi.requires_grad_(False)
            else:
                p.requires_grad_(False)
