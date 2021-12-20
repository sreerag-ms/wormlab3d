from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model import Cameras, MFModelParameters
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_DELTA_ANGLES_BASIS, ENCODING_MODE_POINTS, ENCODING_MODE_MSC
from wormlab3d.midlines3d.args_finder import OptimiserArgs

PARAMETER_NAMES = [
    'cam_intrinsics',
    # 'cam_rotations',
    'cam_rotation_preangles',
    'cam_translations',
    'cam_distortions',
    'cam_shifts',
    # 'cloud_points',
    'curve_parameters',
    # 'curve_length',
    # 'blur_sigmas_cloud',
    'blur_sigmas_curve',
    'blur_intensities_curve',
    'blur_sigmas_cameras_sfs',
    'blur_intensities_cameras_sfs',
]

CAM_PARAMETER_NAMES = [
    'intrinsics',
    # 'rotations',
    'rotation_preangles',
    'translations',
    'distortions',
    'shifts',
]

BUFFER_NAMES = [
    'images',
    'masks_target',
    'cam_coeffs_db',
    'cam_rotations',
    'points_3d_base',
    'points_2d_base',
    # 'worm_length_db',
    # 'masks_cloud',
    'masks_curve',
    # 'curve_points',
    # 'cloud_points_scores',
    'curve_points_scores'
]


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


def _make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi):
    return torch.stack([
        torch.stack([
            cos_theta * cos_phi,
            sin_psi * sin_theta * cos_phi - cos_psi * sin_phi,
            cos_psi * sin_theta * cos_phi + sin_psi * sin_phi
        ]),
        torch.stack([
            cos_theta * sin_phi,
            sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
            cos_psi * sin_theta * sin_phi - sin_psi * cos_phi
        ]),
        torch.stack([
            -sin_theta,
            sin_psi * cos_theta,
            cos_psi * cos_theta
        ])
    ])  # , device=cos_phi.device)


class FrameState(nn.Module):
    def __init__(
            self,
            frame_num: int,
            images: torch.Tensor,
            masks_target: torch.Tensor,
            cameras: Cameras,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
            worm_length_db: torch.Tensor,
            model_params: MFModelParameters,
            optimiser_args: OptimiserArgs,
            prev_frame_state: 'FrameState' = None,
            master_frame_state: 'FrameState' = None,
    ):
        super().__init__()
        self.frame_num = frame_num
        self.register_buffer('images', images)
        self.register_buffer('cam_coeffs_db', _extract_camera_coefficients(cameras))
        self.register_buffer('points_3d_base', points_3d_base)
        self.register_buffer('points_2d_base', points_2d_base)
        self.register_buffer('worm_length_db', worm_length_db)

        if model_params.curve_mode == ENCODING_MODE_MSC:
            for d, mt in enumerate(masks_target):
                self.register_buffer(f'masks_target_{d}', mt)
        else:
            self.register_buffer('masks_target', masks_target)

        self.is_frozen = False
        self.model_params = model_params
        self.optimiser_args = optimiser_args
        self.prev_frame_state = prev_frame_state
        self.master_frame_state = master_frame_state

        if master_frame_state is not None:
            for k in PARAMETER_NAMES:
                self.register_parameter(k, master_frame_state.get_state(k))
        else:
            self._init_parameters()
            self._init_cam_coeffs()
        self._init_outputs()
        self.stats: Dict[str, torch.Tensor] = {}

        # Copy state over from previous state if available
        if prev_frame_state is not None:
            with torch.no_grad():
                for k in PARAMETER_NAMES:
                    self.set_state(k, prev_frame_state.get_state(k))

                for k in [
                    # 'masks_cloud',
                    'masks_curve',
                    # 'curve_points',
                    # 'cloud_points_scores',
                    'curve_points_scores'
                ]:
                    self.set_state(k, prev_frame_state.get_state(k))

                if self.model_params.curve_mode != ENCODING_MODE_MSC:
                    # Add noise to cloud points
                    if self.optimiser_args.cloud_points_perturbation > 0:
                        cp = self._parameters['cloud_points']
                        mean = torch.zeros(model_params.n_cloud_points, 3)
                        noise = torch.normal(mean=mean, std=self.optimiser_args.cloud_points_perturbation)
                        noise = noise.to(cp.device)
                        cp.data += noise

                    # Add noise to sigmas
                    if self.optimiser_args.cloud_sigmas_perturbation > 0:
                        cs = self._parameters['blur_sigmas_cloud']
                        # mean = torch.zeros(model_params.n_cloud_points)
                        # noise = torch.normal(mean=mean, std=self.optimiser_args.cloud_sigmas_perturbation)
                        # noise = torch.abs(noise).to(cp.device)
                        # cs.data += noise
                        # cs.data = cs.clamp(min=1e-3)
                        cs.data = torch.ones_like(cs) * 1e-2

                # Straighten-out curve
                if model_params.curve_mode == ENCODING_MODE_DELTA_ANGLES:
                    delta_angles = self._parameters['curve_parameters'][7:]
                    delta_angles.data = delta_angles * 0.5

    def get_state(self, key: str) -> torch.Tensor:
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._buffers:
            return self._buffers[key]
        elif key == 'cam_coeffs':
            return self.get_cam_coeffs()
        elif self.model_params.curve_mode == ENCODING_MODE_MSC and key in ['masks_target', 'masks_curve',
                                                                           'curve_points_scores']:
            return [self._buffers[f'{key}_{d}'] for d in range(self.model_params.ms_curve_depth)]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise RuntimeError(f'Could not get state for {key}.')

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        if type(param) == list:
            param_list = []
            for d, p in enumerate(param):
                self.register_parameter(f'{name}_{d}', p)
                param_list.append(self._parameters[f'{name}_{d}'])
            setattr(self, name, param_list)
        else:
            return super().register_parameter(name, param)

    def get_cam_coeffs(self):
        # Build rotation matrices
        Rs = []
        rotation_preangles = self._parameters['cam_rotation_preangles']
        for i in range(3):
            pre = rotation_preangles[i]
            cos_phi, sin_phi = pre[0]
            cos_theta, sin_theta = pre[1]
            cos_psi, sin_psi = pre[2]
            Ri = _make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi)
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

    def set_state(self, key: str, data: torch.Tensor):
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

    def set_stats(self, stats: Dict[str, torch.Tensor]):
        self.stats = {**self.stats, **stats}

    def freeze(self):
        self.is_frozen = True
        for key in PARAMETER_NAMES:
            self.get_state(key).requires_grad_(False)

    def _init_parameters(self):
        """
        Initialise the camera coefficients, the cloud points and the curve parameters.
        """
        mp = self.model_params
        oa = self.optimiser_args

        # Distribute initial cloud points randomly on surface of a sphere
        mean = torch.zeros(mp.n_cloud_points, 3)
        x = torch.normal(mean=mean, std=1)
        x = x / torch.norm(x, dim=-1, keepdim=True) * 0.01
        cloud_points = x
        self.register_parameter('cloud_points', nn.Parameter(cloud_points, requires_grad=oa.optimise_cloud))

        # Curve encoding varies depending on mode
        if mp.curve_mode == ENCODING_MODE_MSC:
            curve_parameters = []
            for d in range(mp.ms_curve_depth):
                if d == 0:
                    x = torch.normal(mean=torch.zeros(3), std=1 / (2**(d + 4)))
                    x = x.unsqueeze(0)
                elif d == 1:
                    x = torch.normal(mean=torch.cat([x, x]), std=1 / (2**(d + 4)))
                else:
                    r = (x[1:] - x[:-1]).mean(dim=0, keepdim=True)  # / 3
                    r = r[0] / 3
                    x = torch.stack([
                        torch.linspace(float(x[0, j] - r[j]), float(x[-1, j] + r[j]), 2**d)
                        for j in range(3)
                    ], axis=1)
                # print(x.shape)
                # print(x)
                x = nn.Parameter(x, requires_grad=oa.optimise_curve)
                curve_parameters.append(x)

            # x = torch.zeros(1, 3)
            # for d in range(mp.ms_curve_depth):
            #     mean = torch.zeros(2**d, 3)
            #     if d > 0:
            #         mean += torch.cat([x] * 2, dim=0)
            #     x = torch.normal(mean=mean, std=1 / (2**(d+4)))
            #     x = nn.Parameter(x, requires_grad=oa.optimise_curve)
            #     # curve_parameters.append(self._parameters[f'curve_parameters_{d}'])
            #     curve_parameters.append(x)
            self.register_parameter(f'curve_parameters', curve_parameters)
            # self.curve_parameters = curve_parameters

        else:
            curve_parameters = []
            if mp.curve_mode == ENCODING_MODE_POINTS:
                cc0 = torch.linspace(-np.sqrt(3) / 6, np.sqrt(3) / 6, mp.n_curve_points)
                curve_parameters.append(torch.stack([cc0] * 3, axis=1))
            elif mp.curve_mode == ENCODING_MODE_DELTA_VECTORS:
                offset0 = torch.zeros(3)
                curve_parameters.append(offset0)
                delta_vectors0 = torch.normal(mean=torch.zeros(3 * mp.n_curve_points), std=0.2)
                curve_parameters.append(delta_vectors0)
            else:
                offset0 = torch.zeros(3)
                curve_parameters.append(offset0)
                pre_angles0 = torch.rand(size=(4,)) * 2 - 1
                curve_parameters.append(pre_angles0)

                if mp.curve_mode == ENCODING_MODE_DELTA_ANGLES:
                    delta_angles0 = torch.normal(mean=torch.zeros(2 * (mp.n_curve_points - 1)), std=0.1)
                    curve_parameters.append(delta_angles0)
                elif mp.curve_mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
                    amps0 = torch.normal(mean=torch.zeros(2 * mp.n_curve_basis_fns), std=1)
                    phases0 = torch.normal(mean=torch.zeros(2 * mp.n_curve_basis_fns), std=1)
                    curve_parameters.append(amps0)
                    curve_parameters.append(phases0)
            curve_parameters = torch.cat(curve_parameters)
            curve_parameters = nn.Parameter(curve_parameters, requires_grad=oa.optimise_curve)
            self.register_parameter('curve_parameters', curve_parameters)

        # Curve length cloned from database value
        curve_length = self.worm_length_db.clone().detach()
        self.register_parameter('curve_length', nn.Parameter(curve_length, requires_grad=oa.optimise_curve_length))

        # Blur sigmas
        blur_sigmas_cloud = torch.ones(mp.n_cloud_points) * mp.blur_sigmas_cloud_init
        self.register_parameter('blur_sigmas_cloud',
                                nn.Parameter(blur_sigmas_cloud, requires_grad=oa.optimise_cloud_sigmas))

        if mp.curve_mode == ENCODING_MODE_MSC:
            blur_sigmas_curve = []
            blur_intensities_curve = []
            for d in range(mp.ms_curve_depth):
                sigmas_d = torch.ones(2**d) * mp.blur_sigmas_curve_init / (2**d)
                sigmas_d = nn.Parameter(sigmas_d, requires_grad=oa.optimise_curve_sigmas)
                self.register_parameter(f'blur_sigmas_curve_{d}', sigmas_d)
                blur_sigmas_curve.append(self._parameters[f'blur_sigmas_curve_{d}'])
                self.blur_sigmas_curve = blur_sigmas_curve

                intensities_d = torch.ones(2**d)  # / 2
                intensities_d = nn.Parameter(intensities_d, requires_grad=oa.optimise_curve_intensities)
                self.register_parameter(f'blur_intensities_curve_{d}', intensities_d)
                blur_intensities_curve.append(self._parameters[f'blur_intensities_curve_{d}'])
                self.blur_intensities_curve = blur_intensities_curve


            self.register_parameter('blur_sigmas_cameras_sfs', nn.Parameter(torch.ones(3), requires_grad=oa.optimise_curve_sigmas))
            self.register_parameter('blur_intensities_cameras_sfs', nn.Parameter(torch.ones(3), requires_grad=oa.optimise_curve_intensities))

        else:
            blur_sigmas_curve = torch.ones(mp.n_curve_points) * mp.blur_sigmas_curve_init
            blur_sigmas_curve = nn.Parameter(blur_sigmas_curve, requires_grad=oa.optimise_curve_sigmas)
            self.register_parameter('blur_sigmas_curve', blur_sigmas_curve)

    def _init_cam_coeffs(self):
        oa = self.optimiser_args

        # Initial camera coefficients are cloned from the database
        cam_coeffs = self.cam_coeffs_db.clone().detach()

        self.register_parameter(
            'cam_intrinsics',
            nn.Parameter(cam_coeffs[:, :4], requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_intrinsics)
        )
        self.register_parameter(
            'cam_translations',
            nn.Parameter(cam_coeffs[:, 13:16], requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_translations)
        )
        self.register_parameter(
            'cam_distortions',
            nn.Parameter(cam_coeffs[:, 16:21], requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_distortions)
        )
        self.register_parameter(
            'cam_shifts',
            nn.Parameter(cam_coeffs[:, 21].unsqueeze(-1),
                         requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_shifts)
        )

        # Extract Euler rotation angles from the rotation matrices
        R = cam_coeffs[:, 4:13].reshape((3, 3, 3))
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

            Ri2 = _make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi)
            assert torch.allclose(Ri, Ri2, atol=1e-3)
            Rs.append(Ri2.flatten())

        # Rs = torch.stack(Rs)

        self.register_parameter(
            'cam_rotation_preangles',
            nn.Parameter(rotation_preangles, requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_rotations)
        )
        # self.register_buffer('cam_rotations', Rs)

        # print(self.get_state('cam_rotations'))
        # self.register_parameter(
        #     'cam_rotations',
        #     nn.Parameter(cam_coeffs[:, 4:13], requires_grad=oa.optimise_cam_coeffs and oa.optimise_cam_rotations)
        # )
        #
        #
        # self.register_buffer('cam_coeffs', torch.cat([
        #     self.cam_intrinsics,
        #     self.cam_rotations,
        #     self.cam_translations,
        #     self.cam_distortions,
        #     self.cam_shifts,
        # ], dim=1))

    def _init_outputs(self):
        """
        Initialise the camera coefficients, the cloud points and the curve parameters.
        """
        mp = self.model_params

        # Camera rotation matrices (flattened)
        self.register_buffer('cam_rotations', torch.zeros(3, 9))

        # Setup buffers for the outputs
        if mp.curve_mode == ENCODING_MODE_MSC:
            # self.masks_curve = []
            # self.curve_points_scores = []
            mt = self.get_state('masks_target')
            for d in range(mp.ms_curve_depth):
                self.register_buffer(f'masks_curve_{d}', torch.zeros_like(mt[d]))
                self.register_buffer(f'curve_points_scores_{d}', torch.zeros(2**d))
                # self.masks_curve.append(self._buffers[f'masks_curve_{d}'])
                # self.curve_points_scores.append(self._buffers[f'curve_points_scores_{d}'])
        else:
            self.register_buffer('masks_cloud', torch.zeros_like(self.masks_target))
            self.register_buffer('masks_curve', torch.zeros_like(self.masks_target))
            self.register_buffer('curve_points', torch.zeros(mp.n_curve_points, 3))
            self.register_buffer('cloud_points_scores', torch.zeros(mp.n_cloud_points))
            self.register_buffer('curve_points_scores', torch.zeros(mp.n_curve_points))
