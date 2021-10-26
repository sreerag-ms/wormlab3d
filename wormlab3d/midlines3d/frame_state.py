from typing import Dict

import numpy as np
import torch
from torch import nn
from wormlab3d import CAMERA_IDXS
from wormlab3d import logger
from wormlab3d.data.model import Cameras, MFModelParameters
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_DELTA_ANGLES_BASIS, ENCODING_MODE_POINTS
from wormlab3d.midlines3d.args_finder import OptimiserArgs

PARAMETER_NAMES = [
    'cam_coeffs',
    'cloud_points',
    'curve_parameters',
    'curve_length',
    'blur_sigmas_cloud',
    'blur_sigmas_curve',
]

BUFFER_NAMES = [
    'images',
    'masks_target',
    'cam_coeffs_db',
    'points_3d_base',
    'points_2d_base',
    'worm_length_db',
    'masks_cloud',
    'masks_curve',
    'curve_points',
    'cloud_points_scores',
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
            optimiser_args: OptimiserArgs
    ):
        super().__init__()
        self.frame_num = frame_num
        self.register_buffer('images', images)
        self.register_buffer('masks_target', masks_target)
        self.register_buffer('cam_coeffs_db', _extract_camera_coefficients(cameras))
        self.register_buffer('points_3d_base', points_3d_base)
        self.register_buffer('points_2d_base', points_2d_base)
        self.register_buffer('worm_length_db', worm_length_db)

        self.model_params = model_params
        self.optimiser_args = optimiser_args
        self._init_parameters()
        self._init_outputs()
        self.stats: Dict[str, torch.Tensor] = {}

    def get_state(self, key: str) -> torch.Tensor:
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._buffers:
            return self._buffers[key]
        else:
            raise RuntimeError(f'Could not get state for {key}.')

    def set_state(self, key: str, data: torch.Tensor):
        if key in self._parameters:
            self._parameters[key].data = data
        elif key in self._buffers:
            self._buffers[key].data = data
        else:
            raise RuntimeError(f'Could not set state for {key}.')

    def set_stats(self, stats: Dict[str, torch.Tensor]):
        self.stats = {**self.stats, **stats}

    def freeze(self):
        for key in PARAMETER_NAMES:
            self.get_state(key).requires_grad(False)

    def _init_parameters(self):
        """
        Initialise the camera coefficients, the cloud points and the curve parameters.
        """
        logger.debug(f'Initialising frame state parameters.')
        mp = self.model_params

        # Initial camera coefficients are cloned from the database
        cam_coeffs = self.cam_coeffs_db.clone().detach()

        # Distribute initial cloud points randomly on surface of a sphere
        mean = torch.zeros(mp.n_cloud_points, 3)
        x = torch.normal(mean=mean, std=1)
        x = x / torch.norm(x, dim=-1, keepdim=True) * 0.4
        cloud_points = x

        # Curve encoding varies depending on mode
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

        # Curve length cloned from database value
        curve_length = self.worm_length_db.clone().detach()

        # Blur sigmas
        blur_sigmas_cloud = torch.ones(mp.n_cloud_points) * mp.blur_sigmas_cloud_init
        blur_sigmas_curve = torch.ones(mp.n_curve_points) * mp.blur_sigmas_curve_init

        # Store them in the parameter holder
        oa = self.optimiser_args
        self.register_parameter('cam_coeffs', nn.Parameter(cam_coeffs, requires_grad=oa.optimise_cam_coeffs))
        self.register_parameter('cloud_points', nn.Parameter(cloud_points, requires_grad=oa.optimise_cloud))
        self.register_parameter('curve_parameters', nn.Parameter(curve_parameters, requires_grad=oa.optimise_curve))
        self.register_parameter('curve_length', nn.Parameter(curve_length, requires_grad=oa.optimise_curve_length))
        self.register_parameter('blur_sigmas_cloud',
                                nn.Parameter(blur_sigmas_cloud, requires_grad=oa.optimise_cloud_sigmas))
        self.register_parameter('blur_sigmas_curve',
                                nn.Parameter(blur_sigmas_curve, requires_grad=oa.optimise_curve_sigmas))

    def _init_outputs(self):
        """
        Initialise the camera coefficients, the cloud points and the curve parameters.
        """
        logger.debug(f'Initialising frame state outputs.')
        mp = self.model_params

        # Setup buffers for the outputs
        self.register_buffer('masks_cloud', torch.zeros_like(self.masks_target))
        self.register_buffer('masks_curve', torch.zeros_like(self.masks_target))
        self.register_buffer('curve_points', torch.zeros(mp.n_curve_points, 3))
        self.register_buffer('cloud_points_scores', torch.zeros(mp.n_cloud_points))
        self.register_buffer('curve_points_scores', torch.zeros(mp.n_curve_points))
