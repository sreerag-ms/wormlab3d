from typing import Dict, Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model import Cameras, MFParameters, Frame
from wormlab3d.midlines3d.mf_methods import make_rotation_matrix, normalise, integrate_curvature, an_orthonormal

PARAMETER_NAMES = [
    'cam_intrinsics',
    'cam_rotation_preangles',
    'cam_translations',
    'cam_distortions',
    'cam_shifts',
    'X0',
    'T0',
    'M10',
    'length',
    'curvatures',
    'points',
    'sigmas',
    'exponents',
    'intensities',
    'camera_sigmas',
    'camera_exponents',
    'camera_intensities',
    'filters'
]

CURVATURE_PARAMETER_NAMES = [
    'X0',
    'T0',
    'M10',
    'length',
    'curvatures',
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
]

TRANSIENTS_NAMES = [
    'images',
    'cam_coeffs_db',
    'points_3d_base',
    'points_2d_base',
    'curvatures_smoothed',
    'sigmas_smoothed',
    'exponents_smoothed',
    'intensities_smoothed',
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


def _extract_euler_angles(R: torch.Tensor) -> torch.Tensor:
    """
    Extract Euler rotation angles from the rotation matrices
    """
    assert R.shape == (3, 3, 3)
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

    return rotation_preangles


class FrameState(nn.Module):
    def __init__(
            self,
            frame: Frame,
            parameters: MFParameters,
            prev_frame_state: 'FrameState' = None,
            master_frame_state: 'FrameState' = None,
            use_master_points: bool = True,
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
            if not use_master_points:
                self._init_points_parameters()

            # Set up points/curvatures buffers
            else:
                D = parameters.depth
                D_min = parameters.depth_min
                ks = ['points', ] if parameters.curvature_mode else CURVATURE_PARAMETER_NAMES
                for k in ks:
                    for d in range(D_min, D):
                        self.register_buffer(f'{k}_{d}', torch.zeros((2**d, 3)))
                for d in range(D_min, D):
                    self.register_buffer(f'curvatures_smoothed_{d}', torch.zeros((2**d, 2)))

            for k in PARAMETER_NAMES:
                is_buffer = (k == 'points' and parameters.curvature_mode) \
                            or (k in CURVATURE_PARAMETER_NAMES and not parameters.curvature_mode)
                if k in ['points', ] + CURVATURE_PARAMETER_NAMES and (is_buffer or not use_master_points):
                    with torch.no_grad():
                        self.set_state(k, master_frame_state.get_state(k))
                else:
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
                for k in ['masks_curve', 'points_2d', 'scores']:
                    self.set_state(k, prev_frame_state.get_state(k))
                self.stats = prev_frame_state.stats

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

        # Linearly interpolate target mask resolutions from 8x8 to crop_size x crop_size (200x200 default)
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        sizes = torch.linspace(8, self.frame.trial.crop_size, D).to(torch.int32)

        # Generate downsampled target masks
        for d in range(D_min, D):
            image_size = sizes[d]

            if d < D - 1:
                # Downsample the full resolution version to the reduced size
                masks_ds = F.interpolate(masks_fr, (image_size, image_size), mode='nearest')

                # Upsample to restore the target size
                masks_rs = F.interpolate(masks_ds, (self.frame.trial.crop_size, self.frame.trial.crop_size),
                                         mode='bilinear', align_corners=False)

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
        self._init_points_parameters()

        # Initialise the sigmas
        sigmas = []
        sigmas_init = max(mp.sigmas_min + 0.01, mp.sigmas_init)
        for d in range(mp.depth_min, mp.depth):
            sigmas_d = nn.Parameter(torch.tensor(sigmas_init), requires_grad=mp.optimise_sigmas)
            sigmas.append(sigmas_d)
        self.register_parameter('sigmas', sigmas)

        # Initialise the exponents all to 1
        exponents = []
        for d in range(mp.depth_min, mp.depth):
            exponents_d = nn.Parameter(torch.tensor(1.), requires_grad=mp.optimise_exponents)
            exponents.append(exponents_d)
        self.register_parameter('exponents', exponents)

        # Initialise the intensities all to 1
        intensities = []
        for d in range(mp.depth_min, mp.depth):
            intensities_d = nn.Parameter(torch.tensor(1.), requires_grad=mp.optimise_intensities)
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

        # Camera filters
        if mp.filter_size is not None:
            filters = torch.zeros((3, mp.filter_size, mp.filter_size), dtype=torch.float32)
            filters[:, int(mp.filter_size / 2), int(mp.filter_size / 2)] = 1.
            filters += torch.randn_like(filters) * 1e-4
            filters /= filters.norm(dim=(1, 2), keepdim=True)
            self.register_parameter(
                'filters',
                nn.Parameter(filters, requires_grad=True)
            )
        else:
            self.register_parameter(
                'filters',
                nn.Parameter(torch.zeros((3, 1, 1), dtype=torch.float32), requires_grad=False)
            )

    def _init_points_parameters(self):
        """
        Initialise the multiscale curve points and curvatures.
        """
        if self.parameters.curvature_mode:
            self._init_points_parameters_curvature_mode()
        else:
            self._init_points_parameters_points_mode()

    def _init_points_parameters_points_mode(self):
        """
        Initialise the multiscale curve points in points mode.
        """
        mp = self.parameters

        # Initialise the points perfectly spaced along a line of random length.
        points = []
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
            if d >= mp.depth_min:
                x = nn.Parameter(x, requires_grad=True)
                points.append(x)
                self.register_buffer(f'X0_{d}', torch.zeros(3))
                self.register_buffer(f'T0_{d}', torch.tensor([1., 0., 0.]))
                self.register_buffer(f'M10_{d}', torch.tensor([0., 1., 0.]))
                self.register_buffer(f'length_{d}', torch.tensor(1.))
                self.register_buffer(f'curvatures_{d}', torch.zeros(2**d, 2))

        self.register_parameter('points', points)

    def _init_points_parameters_curvature_mode(self):
        """
        Initialise the multiscale curve points and curvatures in curvatures mode.
        """
        mp = self.parameters

        # Pick a random midpoint near the centre
        X0 = torch.normal(mean=torch.zeros(3), std=0.1)
        X0s = [nn.Parameter(X0, requires_grad=True) for _ in range(mp.depth - mp.depth_min)]

        # Pick a random tangent direction
        T0 = torch.normal(mean=torch.zeros(3), std=1)
        T0 = normalise(T0)
        T0s = [nn.Parameter(T0, requires_grad=True) for _ in range(mp.depth - mp.depth_min)]

        # Pick a consistent M10 direction
        M10s = [
            nn.Parameter(an_orthonormal(T0s[i].unsqueeze(0))[0].detach(), requires_grad=True)
            for i in range(mp.depth - mp.depth_min)
        ]

        # Init lengths
        l = torch.tensor(mp.length_init)
        lengths = [nn.Parameter(l, requires_grad=True) for _ in range(mp.depth - mp.depth_min)]

        # Start in a straight line configuration
        curvatures = []
        for i, d in enumerate(range(mp.depth_min, mp.depth)):
            N = 2**d
            K = nn.Parameter(torch.zeros(N, 2), requires_grad=True)
            points_d, tangents_d, M1_d = integrate_curvature(
                X0s[i].unsqueeze(0),
                T0s[i].unsqueeze(0),
                lengths[i].unsqueeze(0),
                K.unsqueeze(0),
                M10=M10s[i].unsqueeze(0)
            )
            curvatures.append(K)
            self.register_buffer(f'points_{d}', points_d[0].detach())
            self.register_buffer(f'curvatures_smoothed_{d}', K.detach())

        self.register_parameter('X0', X0s)
        self.register_parameter('T0', T0s)
        self.register_parameter('M10', M10s)
        self.register_parameter('length', lengths)
        self.register_parameter('curvatures', curvatures)

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
        rotation_preangles = _extract_euler_angles(R)

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
        D_min = self.parameters.depth_min

        # Camera rotation matrices (flattened)
        self.register_buffer('cam_rotations', torch.zeros(3, 9))

        # Setup buffers for the outputs
        mt = self.get_state('masks_target')
        for i, d in enumerate(range(D_min, D)):
            self.register_buffer(f'points_2d_{d}', torch.zeros(2**d))
            self.register_buffer(f'masks_curve_{d}', torch.zeros_like(mt[i]))
            self.register_buffer(f'masks_target_residuals_{d}', torch.zeros_like(mt[i]))
            self.register_buffer(f'scores_{d}', torch.zeros(2**d))
            self.register_buffer(f'sigmas_smoothed_{d}', torch.zeros(2**d))
            self.register_buffer(f'exponents_smoothed_{d}', torch.zeros(2**d))
            self.register_buffer(f'intensities_smoothed_{d}', torch.zeros(2**d))

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
                d_str = f'{d + self.parameters.depth_min}'
                self.register_parameter(f'{name}_{d_str}', p)
                param_list.append(self._parameters[f'{name}_{d_str}'])
            setattr(self, name, param_list)
        else:
            return super().register_parameter(name, param)

    def get_state(self, key: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Retrieve a buffer or parameter.
        """
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._buffers:
            return self._buffers[key]
        elif key == 'cam_coeffs':
            return self.get_cam_coeffs()
        elif hasattr(self, key):
            return getattr(self, key)
        elif key in ['points', 'curvatures_smoothed', 'sigmas_smoothed', 'exponents_smoothed', 'intensities_smoothed',
                     'masks_target', 'masks_target_residuals', 'masks_curve', 'points_2d',
                     'scores'] + CURVATURE_PARAMETER_NAMES:
            return [self._buffers[f'{key}_{d}'] for d in range(self.parameters.depth_min, self.parameters.depth)]
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

    def copy_state(self, from_state: 'FrameState'):
        """
        Copy buffers, parameters and transient values over from another frame state.
        """
        with torch.no_grad():
            for k in BUFFER_NAMES + PARAMETER_NAMES + TRANSIENTS_NAMES:
                self.set_state(k, from_state.get_state(k))

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
