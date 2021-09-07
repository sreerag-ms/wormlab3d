import os
import time
from typing import Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from mongoengine import DoesNotExist
from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from wormlab3d import CAMERA_IDXS, PREPARED_IMAGE_SIZE
from wormlab3d import logger, LOGS_PATH
from wormlab3d.data.model import SegmentationMasks, Trial, Frame, Cameras, MFCheckpoint, MFModelParameters
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_DELTA_ANGLES_BASIS, MAX_DECAY_FACTOR, ENCODING_MODE_POINTS
from wormlab3d.midlines3d.args_finder import ModelArgs, OptimiserArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.args_finder.optimiser_args import LOSS_CURVE_TARGET_MASKS
from wormlab3d.midlines3d.dynamic_cameras import DynamicCameras
from wormlab3d.nn.args.optimiser_args import LOSS_MSE, LOSS_LOGDIFF, LOSS_KL
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')
cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'

PARAMETERS = [
    'cam_coeffs',
    'cloud_points',
    'curve_parameters',
    'curve_length',
    'blur_sigmas_cloud',
    'blur_sigmas_curve',
]

PRINT_KEYS = [
    'cloud_points_scores/mean',
    'n_relocated',
    'loss/3d',
    'loss/render',
    'curve_points_scores/mean',
    'loss/preangles',
    'loss/worm_length',
    'worm_length'
]


def _avg_pool_2d(grad, oob_grad_val=0., mode='constant'):
    # Average pooling with overlap and boundary values
    padded_grad = F.pad(grad, (1, 1, 1, 1), mode=mode, value=oob_grad_val)
    ag = F.avg_pool2d(input=padded_grad, kernel_size=3, stride=2, padding=0)
    return ag


def render_points(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]
    device = points.device

    # Shift to [-1,+1]
    points = (points - 100) / 100

    # Reshape
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)

    # Build x and y grids
    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=device)
    yv, xv = torch.meshgrid([grid, grid])

    # 1 x 1 x H x W x 2
    m = torch.cat([xv[..., None], yv[..., None]], dim=-1)[None, None]
    p2 = points[:, :, None, None, :]

    # Make (un-normalised) gaussian blobs centred at the coordinates
    mmp2 = m - p2
    dst = mmp2[..., 0]**2 + mmp2[..., 1]**2
    blobs = torch.exp(-(dst / (2 * blur_sigmas**2)[:, :, None, None]))

    # Mask is the sum of the blobs
    masks = blobs.sum(dim=1)

    # Normalise
    masks = masks.clamp(max=1.)
    sum_ = masks.sum(dim=(1, 2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks_normed = masks / sum_

    # Reshape
    masks_normed = masks_normed.reshape(bs, 3, *PREPARED_IMAGE_SIZE)
    blobs = blobs.reshape(bs, 3, n_worm_points, *PREPARED_IMAGE_SIZE)

    return masks_normed, blobs


def render_curve(points, blur_sigmas):
    n_worm_points = points.shape[2]
    bs = points.shape[0]

    # Shift to [-1,+1]
    points = (points - 100) / 100

    # Reshape
    points = points.reshape(bs * 3, n_worm_points, 2)
    points = points.clamp(min=-1, max=1)
    a = points[:, :-1]
    b = points[:, 1:]

    def sumprod(x, y, keepdim=True):
        return torch.sum(x * y, dim=-1, keepdim=keepdim)

    grid = torch.linspace(-1.0, 1.0, PREPARED_IMAGE_SIZE[0], dtype=torch.float32, device=a.device)

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
    sig = (blur_sigmas[:, 1:] + blur_sigmas[:, :-1]) / 2
    lines = torch.exp(-d / (sig**2)[:, :, None, None])
    # lines = torch.exp(-(dst / (2 * blur_sigmas**2)[:, :, None, None]))

    # Sum the lines together to make the render
    masks = lines.sum(dim=1)
    masks = masks.clamp(max=1)

    # Normalise and reshape
    sum_ = masks.sum(dim=(1, 2), keepdim=True)
    sum_ = sum_.clamp(min=1e-8)
    masks = (masks / sum_).reshape(bs, 3, *PREPARED_IMAGE_SIZE)

    return masks


class ParameterHolder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise RuntimeError('The forward method of this model is not enabled!')

    def get(self, key: str) -> torch.Tensor:
        return self._parameters[key]

    def set(self, key: str, value: torch.Tensor, requires_grad: bool = False):
        if key in self._parameters:
            self._parameters[key].data = value
        else:
            self.register_parameter(key, nn.Parameter(value, requires_grad=requires_grad))


class ProjectRenderScoreModel(nn.Module):
    def __init__(
            self,
            points_3d_base: torch.Tensor,
            points_2d_base: torch.Tensor,
    ):
        super().__init__()
        self.cams = DynamicCameras()
        self.register_buffer('points_2d_base', points_2d_base)
        self.register_buffer('points_3d_base', points_3d_base)

        # todo Grow the worm out slowly
        self.register_buffer('decay_factor', torch.tensor(MAX_DECAY_FACTOR, dtype=torch.float32))

    def forward(self):
        raise RuntimeError('The forward method of this model is not enabled!')

    def forward_cloud(
            self,
            cam_coeffs: torch.Tensor,
            cloud_points: torch.Tensor,
            masks_target: torch.Tensor,
            blur_sigmas_cloud: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Render the point cloud
        cloud_points_2d = self._project_to_2d(cam_coeffs, cloud_points)
        masks_cloud, blobs = render_points(cloud_points_2d, blur_sigmas_cloud)

        # Normalise blobs
        sum_ = blobs.sum(dim=(2, 3), keepdim=True)
        sum_ = sum_.clamp(min=1e-8)
        blobs_normed = blobs / sum_

        # Calculate points scores
        cloud_points_scores = (blobs_normed * masks_target.unsqueeze(2)).sum(dim=(3, 4)).mean(dim=1)

        return masks_cloud, cloud_points_scores

    def forward_curve(
            self,
            cam_coeffs: torch.Tensor,
            curve_points: torch.Tensor,
            blur_sigmas_curve: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Render the curve
        curve_points_2d = self._project_to_2d(cam_coeffs, curve_points)
        masks_curves = render_curve(curve_points_2d, blur_sigmas_curve)

        # todo: Get the point scores for the curve points
        curve_points_scores = torch.zeros_like(blur_sigmas_curve)

        return masks_curves, curve_points_scores

    def _project_to_2d(
            self,
            cam_coeffs: torch.Tensor,
            points_3d: torch.Tensor
    ):
        bs = cam_coeffs.shape[0]
        device = cam_coeffs.device

        # Add the 3d centre point offset to centre on the camera
        points_3d = points_3d + self.points_3d_base

        # Project 3D points to 2D
        points_2d = self.cams.forward(cam_coeffs, points_3d)

        # Re-centre according to 2D base points plus a (100,100) to put it in the centre of the cropped image
        image_centre_pt = torch.ones((bs, 1, 1, 2), dtype=torch.float32, device=device) * PREPARED_IMAGE_SIZE[0] / 2
        points_2d = points_2d - self.points_2d_base.unsqueeze(1) + image_centre_pt

        return points_2d


class Midline3DFinder:
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            source_args: SourceArgs,
            model_args: ModelArgs,
            optimiser_args: OptimiserArgs,
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.source_args = source_args
        self.model_args = model_args
        self.optimiser_args = optimiser_args

        # Store tensor parameters in a module
        self.parameters = ParameterHolder()

        # Load objects from the database
        self._init_sources()

        # Initialise the model
        self.model, self.model_params = self._init_model()

        # Initialise initial conditions and trainable parameters
        self._init_parameters()

        # Runtime params
        self.device = self._init_devices()

        # Optimisers
        self.optimiser_cc, self.optimiser_curve = self._init_optimisers()

        # Checkpoints
        self.checkpoint = self._init_checkpoint()

        # Plotting
        self.plot_3d_azim = -60

    @property
    def logs_path(self) -> str:
        return self.get_logs_path(self.checkpoint)

    @staticmethod
    def get_logs_path(checkpoint: MFCheckpoint) -> str:
        return LOGS_PATH + \
               f'/trial_{checkpoint.masks.frame.trial.id}' \
               f'/frame_{checkpoint.masks.frame.frame_num:06d}_{checkpoint.masks.id}' \
               f'/{checkpoint.model_params.created:%Y%m%d_%H:%M}_{checkpoint.model_params.id}'

    @property
    def step(self):
        return self.checkpoint.step_cc + self.checkpoint.step_curve

    def __getattr__(self, key):
        """
        Allow parameters to be accessed as member variables.
        """
        return self.parameters.get(key)

    def _init_sources(self):
        """
        Load the masks instance and related objects from the database,
        """
        logger.info(f'Initialising sources.')
        self.masks: SegmentationMasks = SegmentationMasks.objects.get(id=self.source_args.masks_id)
        self.trial: Trial = self.masks.trial
        self.images = self.masks.get_images()
        self.frame: Frame = self.masks.frame
        self.cameras: Cameras = self.frame.get_cameras()
        self.cam_coeffs_db = self._extract_camera_coefficients()

        # Initialise the target, setting anything above threshold to 1 and normalising
        masks_target = torch.from_numpy(self.masks.X)
        masks_target[masks_target > self.source_args.masks_target_ceil_threshold] = 1
        masks_target = masks_target / masks_target.sum(axis=(1, 2), keepdim=True)
        self.parameters.set('masks_target', masks_target)

        # Base points
        self.parameters.set('points_3d_base', torch.tensor(self.frame.centre_3d.point_3d))
        self.parameters.set('points_2d_base', torch.tensor(self.frame.centre_3d.reprojected_points_2d))

        # Worm length (needed?)
        self.worm_length_db = torch.tensor(self.trial.experiment.worm_length)
        logger.debug(f'Worm length (db) = {self.worm_length_db}')

    def _extract_camera_coefficients(self) -> torch.Tensor:
        """
        Load the camera coefficients from the database object.
        """
        fx = np.array([self.cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
        fy = np.array([self.cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
        cx = np.array([self.cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
        cy = np.array([self.cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
        R = np.array([self.cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
        t = np.array([self.cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
        d = np.array([self.cameras.distortion[c] for c in CAMERA_IDXS])
        s = np.array([[0, ]] * 3)
        cam_coeffs = np.concatenate([
            fx.reshape(3, 1), fy.reshape(3, 1), cx.reshape(3, 1), cy.reshape(3, 1), R.reshape(3, 9), t, d, s
        ], axis=1).astype(np.float32)
        return torch.from_numpy(cam_coeffs)

    def _init_model(self) -> Tuple[ProjectRenderScoreModel, MFModelParameters]:
        """
        Build the model.
        """
        logger.info(f'Initialising model.')

        model_params = None
        params = self.model_args.get_db_params()

        # Try to load an existing model
        if self.model_args.load:
            # If we have a model id then load this from the database
            if self.model_args.model_id is not None:
                model_params = MFModelParameters.objects.get(id=self.model_args.model_id)
            else:
                # Otherwise, try to find one matching the same parameters
                model_params_matching = MFModelParameters.objects(**params)
                if model_params_matching.count() > 0:
                    model_params = model_params_matching[0]
                    logger.info(f'Found {len(model_params_matching)} suitable models in database, using most recent.')
                else:
                    logger.info(f'No suitable models found in database.')
            if model_params is not None:
                logger.info(f'Loaded model (id={model_params.id}, created={model_params.created}).')

        # Not loaded model, so create one
        if model_params is None:
            model_params = MFModelParameters(**params)
            model_params.save()
            logger.info(f'Saved model parameters to database (id={model_params.id})')

        # Instantiate the model
        model = ProjectRenderScoreModel(
            points_3d_base=self.points_3d_base,
            points_2d_base=self.points_2d_base,
        )

        return model, model_params

    def _init_parameters(self):
        """
        Initialise the camera coefficients, the cloud points and the curve parameters.
        """
        logger.info(f'Initialising parameters.')
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

        # Curve length
        curve_length = self.worm_length_db.clone().detach()

        # Blur sigmas
        blur_sigmas_cloud = torch.ones(mp.n_cloud_points) * mp.blur_sigmas_cloud_init
        blur_sigmas_curve = torch.ones(mp.n_curve_points) * mp.blur_sigmas_curve_init

        # Add batch dims
        self.masks_target = self.masks_target.unsqueeze(0)
        self.points_3d_base = self.points_3d_base.unsqueeze(0)
        self.points_2d_base = self.points_2d_base.unsqueeze(0)
        cam_coeffs = cam_coeffs.unsqueeze(0)
        cloud_points = cloud_points.unsqueeze(0)
        curve_parameters = curve_parameters.unsqueeze(0)
        curve_length = curve_length.unsqueeze(0)
        blur_sigmas_cloud = blur_sigmas_cloud.unsqueeze(0)
        blur_sigmas_curve = blur_sigmas_curve.unsqueeze(0)

        # Store them in the parameter holder
        oa = self.optimiser_args
        self.parameters.set('cam_coeffs', cam_coeffs, oa.optimise_cam_coeffs)
        self.parameters.set('cloud_points', cloud_points, oa.optimise_cloud)
        self.parameters.set('curve_parameters', curve_parameters, oa.optimise_curve)
        self.parameters.set('curve_length', curve_length, oa.optimise_curve_length)
        self.parameters.set('blur_sigmas_cloud', blur_sigmas_cloud, oa.optimise_cloud_sigmas)
        self.parameters.set('blur_sigmas_curve', blur_sigmas_curve, oa.optimise_curve_sigmas)

        # Setup blank tensors for the outputs
        self.masks_cloud = torch.zeros_like(self.masks_target)
        self.masks_curve = torch.zeros_like(self.masks_target)
        self.curve_points = torch.zeros(1, mp.n_curve_points, 3)
        self.cloud_points_scores = torch.zeros(1, mp.n_cloud_points)
        self.curve_points_scores = torch.zeros(1, mp.n_curve_points)

    def _init_optimisers(self) -> Tuple[Optimizer, Optimizer]:
        """
        Set up the joint cameras and cloud optimiser and the curve optimiser.
        """
        logger.info('Initialising optimisers.')
        oa = self.optimiser_args

        cls_cc: Optimizer = getattr(torch.optim, oa.algorithm_cc)
        optimiser_cc = cls_cc(
            [
                {'params': (self.cam_coeffs,), 'lr': oa.lr_cam_coeffs},
                {'params': (self.cloud_points,), 'lr': oa.lr_cloud_points},
                {'params': (self.blur_sigmas_cloud,), 'lr': oa.lr_cloud_sigmas},
            ],
            amsgrad=True,
            weight_decay=0
        )

        cls_curve: Optimizer = getattr(torch.optim, oa.algorithm_cc)
        optimiser_curve = cls_curve(
            params=[
                {'params': (self.curve_parameters,), 'lr': oa.lr_curve_points},
                {'params': (self.curve_length,), 'lr': oa.lr_curve_points},
                {'params': (self.blur_sigmas_curve,), 'lr': oa.lr_curve_sigmas},
            ],
            weight_decay=0
        )

        return optimiser_cc, optimiser_curve

    def _init_devices(self):
        """
        Find available devices and try to use what we want.
        """
        if self.runtime_args.gpu_only:
            cpu_or_gpu = 'gpu'
        elif self.runtime_args.cpu_only:
            cpu_or_gpu = 'cpu'
        else:
            cpu_or_gpu = None

        if cpu_or_gpu == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info('Using GPU')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)

            # Move target masks to gpu
            self.masks_target = self.masks_target.to(device)
        else:
            if cpu_or_gpu == 'gpu':
                raise RuntimeError('GPU requested but not available. Aborting.')
            logger.info('Using CPU')

        return device

    def _init_checkpoint(self) -> MFCheckpoint:
        """
        The current checkpoint instance contains the most up to date instance of the model.
        This is not persisted to the database until we actually want to checkpoint it, so should
        be thought of more as a checkpoint-buffer.
        """

        # Load previous checkpoint
        prev_checkpoint: MFCheckpoint = None
        if self.runtime_args.resume:
            try:
                if self.runtime_args.resume_from in ['latest', 'best_cc', 'best_curve']:
                    if self.runtime_args.resume_from == 'latest':
                        order_by = '-created'
                    elif self.runtime_args.resume_from == 'best_cc':
                        order_by = '+loss_cc'
                    else:
                        order_by = '+loss_curve'
                    prev_checkpoints = MFCheckpoint.objects(
                        masks=self.masks,
                        model_params=self.model_params
                    ).order_by(order_by)
                    if prev_checkpoints.count() > 0:
                        logger.info(
                            f'Found {prev_checkpoints.count()} previous checkpoints. '
                            f'Using {self.runtime_args.resume_from}.'
                        )
                        prev_checkpoint = prev_checkpoints[0]
                    else:
                        logger.error(
                            f'Found no checkpoints for masks={self.masks.id} and model={self.model_params.id}.'
                        )
                        raise DoesNotExist()
                else:
                    prev_checkpoint = MFCheckpoint.objects.get(
                        id=self.runtime_args.resume_from
                    )
                logger.info(f'Loaded checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}')
                logger.info(f'Loss cc = {prev_checkpoint.loss_cc:.6f}')
                logger.info(f'Loss curve = {prev_checkpoint.loss_curve:.6f}')
                if len(prev_checkpoint.metrics_cc) > 0:
                    logger.info('CC metrics:')
                    for key, val in prev_checkpoint.metrics_cc.items():
                        logger.info(f'\t{key}: {val:.4E}')
                if len(prev_checkpoint.metrics_curve) > 0:
                    logger.info('Curve metrics:')
                    for key, val in prev_checkpoint.metrics_curve.items():
                        logger.info(f'\t{key}: {val:.4E}')
            except DoesNotExist:
                raise RuntimeError(f'Could not load checkpoint={self.runtime_args.resume_from}')

        # Either clone the previous checkpoint to use as the starting point
        if prev_checkpoint is not None:
            checkpoint = prev_checkpoint.clone()

            # Update the model references to the ones now in use
            if checkpoint.model_params.id != self.model_params.id:
                logger.warning('Model parameters have changed! This may cause problems!')
                checkpoint.model_params = self.model_params

            # Args are stored against the checkpoint, so just override them
            checkpoint.runtime_args = to_dict(self.runtime_args)
            checkpoint.source_args = to_dict(self.source_args)
            checkpoint.optimiser_args = to_dict(self.optimiser_args)

            # Load the parameter states
            path = f'{self.get_logs_path(prev_checkpoint)}/checkpoints/{prev_checkpoint.id}.chkpt'
            state = torch.load(path, map_location=self.device)
            for p in PARAMETERS:
                self.parameters.set(p, state[p])
            if self.optimiser_args.algorithm_cc != prev_checkpoint.optimiser_args['algorithm_cc']:
                self.optimiser_cc.step()
            if self.optimiser_args.algorithm_curve != prev_checkpoint.optimiser_args['algorithm_curve']:
                self.optimiser_curve.step()
            self.optimiser_cc.load_state_dict(state['optimiser_cc_state_dict'])
            self.optimiser_curve.load_state_dict(state['optimiser_curve_state_dict'])
            logger.info(f'Loaded state from "{path}"')

        # ..or start a new checkpoint
        else:
            checkpoint = MFCheckpoint(
                masks=self.masks,
                model_params=self.model_params,
                runtime_args=to_dict(self.runtime_args),
                source_args=to_dict(self.source_args),
                optimiser_args=to_dict(self.optimiser_args),
            )

        return checkpoint

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.logs_path + '/events/' + START_TIMESTAMP, flush_secs=5)

    def _configure_paths(self):
        """
        Initialise the logs and plot directories
        """
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path + '/checkpoints', exist_ok=True)
        os.makedirs(self.logs_path + '/events', exist_ok=True)
        os.makedirs(self.logs_path + '/plots', exist_ok=True)

    def save_checkpoint(self):
        """
        Save the checkpoint information to the database and the parameters to file.
        """
        logger.info('Saving checkpoint...')
        self.checkpoint.save()
        path = f'{self.logs_path}/checkpoints/{self.checkpoint.id}.chkpt'
        params = {
            p: self.parameters.get(p)
            for p in PARAMETERS
        }
        torch.save({
            **params,
            'optimiser_cc_state_dict': self.optimiser_cc.state_dict(),
            'optimiser_curve_state_dict': self.optimiser_curve.state_dict(),
        }, path)

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint
        self.checkpoint = self.checkpoint.clone()

    def train(self, n_steps_cc: int, n_steps_curve: int):
        """
        Train the model.
        """
        self._configure_paths()
        self._init_tb_logger()

        # todo: fix this
        self.model.decay_factor = torch.tensor(0., device=self.device)

        # Initial plots
        self._make_plots(pre_step=True, show_curve=self.checkpoint.step_curve > 0)

        # Train cc
        if n_steps_cc > 0 and (self.optimiser_args.optimise_cloud or self.optimiser_args.optimise_cam_coeffs):
            self._train_cc(n_steps_cc)

        # Train curve
        if n_steps_curve > 0 and self.optimiser_args.optimise_curve:
            self._train_curve(n_steps_curve)

    def _train_cc(self, n_steps: int):
        """
        Train the camera coefficients and cloud points.
        """
        logger.info('======== Training the camera coefficients and cloud points ========')
        start_step = self.checkpoint.step_cc + 1
        final_step = start_step + n_steps

        # Train the cam coeffs and cloud points
        for step in range(start_step, final_step):
            loss, stats = self._train_step_cc()

            # Update step and checkpoint loss
            self.checkpoint.step_cc += 1
            self.checkpoint.loss_cc = loss.item()
            self.checkpoint.metrics_cc = stats

            # Log
            self._log_progress(
                cc_or_curve='cc',
                step=step,
                final_step=final_step,
                loss=loss,
                stats=stats
            )

            # Make plots
            self._make_plots(
                show_cloud=True,
                show_curve=False,
                plot_sigmas=self.optimiser_args.optimise_cloud_sigmas,
                plot_scores=True
            )

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and self.step % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

    def _train_step_cc(self) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Train the cam coeffs and cloud points for a single step.
        """

        # Run the parameters through the model to get the outputs
        masks_cloud, cloud_points_scores = self.model.forward_cloud(
            cam_coeffs=self.cam_coeffs,
            cloud_points=self.cloud_points,
            masks_target=self.masks_target,
            blur_sigmas_cloud=self.blur_sigmas_cloud,
        )

        # Calculate gradients and take optimisation step
        loss, stats = self._calculate_renders_losses(
            render=masks_cloud,
            target=self.masks_target,
            metric=self.optimiser_args.loss_cc,
            multiscale=self.optimiser_args.loss_cc_multiscale,
        )
        self.optimiser_cc.zero_grad()
        loss.backward()
        self.optimiser_cc.step()

        # Do some point relocations
        n_relocated = self._relocate_cloud_points(cloud_points_scores)

        stats = {**stats, **{
            'cloud_points_scores/mean': cloud_points_scores.mean(),
            'cloud_points_scores/var': cloud_points_scores.var(),
            'blur_sigmas_cloud/mean': self.blur_sigmas_cloud.mean(),
            'blur_sigmas_cloud/var': self.blur_sigmas_cloud.var(),
            'n_relocated': n_relocated
        }}

        self.masks_cloud = masks_cloud
        self.cloud_points_scores = cloud_points_scores

        return loss, stats

    def _relocate_cloud_points(self, cloud_points_scores: torch.Tensor) -> int:
        """
        Every so often, any points with low scores should clone to somewhere nearby a point with a better score.
        """
        if not (self.optimiser_args.relocate_every_n_steps > 0
                and (self.checkpoint.step_cc + 1) % self.optimiser_args.relocate_every_n_steps == 0):
            return 0

        # Default the maximum number of points to relocate to 1% of the total points if not defined
        max_turnover = self.optimiser_args.relocate_max_points
        if max_turnover is None:
            max_turnover = int(self.model_params.n_cloud_points * 0.01)

        # Iterate over batch
        bs = cloud_points_scores.shape[0]
        for i in range(bs):
            # Check which points scored below threshold
            n_scored_too_low = (cloud_points_scores[i] <= self.optimiser_args.relocate_score_threshold).sum()
            n_to_relocate = min(max_turnover, n_scored_too_low)

            if n_to_relocate > 0:
                scored_idxs = torch.argsort(cloud_points_scores[i], descending=True)
                src_idxs = scored_idxs[-n_to_relocate:]
                dest_idxs = scored_idxs[:n_to_relocate]

                # Randomise destinations
                random_idxs = torch.randperm(n_to_relocate)
                dest_idxs = dest_idxs[random_idxs]

                with torch.no_grad():
                    # Relocate points
                    self.cloud_points[i][src_idxs] = torch.normal(
                        mean=self.cloud_points[i][dest_idxs],
                        std=self.blur_sigmas_cloud[i][dest_idxs][:, None].expand_as(self.cloud_points[i][dest_idxs])
                    )

        return n_to_relocate

    def _train_curve(self, n_steps: int):
        """
        Train the curve parameters.
        """
        logger.info('======== Training the curve parameters ========')
        start_step = self.checkpoint.step_curve + 1
        final_step = start_step + n_steps

        # Calculate the cloud masks (this doesn't change now)
        with torch.no_grad():
            self.masks_cloud, self.cloud_points_scores = self.model.forward_cloud(
                cam_coeffs=self.cam_coeffs,
                cloud_points=self.cloud_points,
                masks_target=self.masks_target,
                blur_sigmas_cloud=self.blur_sigmas_cloud,
            )

            # Calculate the initial curve points
            self.curve_points = self._get_curve_coordinates()

            # Determine target
            if self.optimiser_args.loss_curve_target == LOSS_CURVE_TARGET_MASKS:
                self.curve_target = self.masks_target
            elif self.optimiser_args.loss_curve_3d:
                T = self.optimiser_args.loss_3d_cloud_threshold
                if T > 0:
                    self.curve_target = self.cloud_points[self.cloud_points_scores > T]
                else:
                    self.curve_target = self.cloud_points
            else:
                self.curve_target = self.masks_cloud

        # Train the curve parameters
        for step in range(start_step, final_step):
            loss, stats = self._train_step_curve()

            # Update step and checkpoint loss
            self.checkpoint.step_curve += 1
            self.checkpoint.loss_curve = loss.item()
            self.checkpoint.metrics_curve = stats

            # Log
            self._log_progress(
                cc_or_curve='curve',
                step=step,
                final_step=final_step,
                loss=loss,
                stats=stats
            )

            # Make plots
            self._make_plots(show_cloud=True, show_curve=True)

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and self.step % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

            # # Update decay factor
            # if step > n_warmup_steps:
            #     self.model.decay_factor = torch.tensor(max(0, 1 - step / (n_adjustment_steps / 2)) * MAX_DECAY_FACTOR)
            #
            # # Set max curvature
            # if step > (n_warmup_steps + n_straight_steps):
            #     x = (step - n_warmup_steps - n_straight_steps) / (n_steps - n_warmup_steps - n_straight_steps)
            #     max_revolutions = torch.tensor(min(1, max(0, x))) * self.max_revolutions_absolute
            #     self.max_revolutions = max_revolutions

    def _train_step_curve(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train the curve parameters for a single step.
        """

        # Run the parameters through the model to get the outputs
        masks_curve, curve_points_scores = self.model.forward_curve(
            cam_coeffs=self.cam_coeffs,
            curve_points=self.curve_points,
            blur_sigmas_curve=self.blur_sigmas_curve,
        )

        loss_render = torch.tensor(0., device=self.device)
        loss_3d = torch.tensor(0., device=self.device)
        stats_render = {}
        stats_3d = {}

        if self.optimiser_args.loss_curve_3d:
            loss_3d, stats_3d = self._calculate_3d_losses()
        else:
            loss_render, stats_render = self._calculate_renders_losses(
                render=masks_curve,
                target=self.curve_target,
                metric=self.optimiser_args.loss_curve,
                multiscale=self.optimiser_args.loss_curve_multiscale,
            )

        loss_curve, stats_curve = self._calculate_curve_losses()

        loss = loss_3d + loss_render + loss_curve

        self.optimiser_curve.zero_grad()
        loss.backward()
        self.optimiser_curve.step()

        self.masks_curve = masks_curve
        self.curve_points = self._get_curve_coordinates()
        self.curve_points_scores = curve_points_scores

        stats = {**stats_3d, **stats_render, **stats_curve, **{
            'loss/3d': loss_3d.item(),
            'loss/render': loss_render.item(),
            'loss/curve': loss_curve.item(),
            'worm_length': self.curve_length.mean(),
            'decay_factor': self.model.decay_factor,
            'curve_points_scores/mean': curve_points_scores.mean(),
            'curve_points_scores/var': curve_points_scores.var(),
            'blur_sigmas_curve/mean': self.blur_sigmas_curve.mean(),
            'blur_sigmas_curve/var': self.blur_sigmas_curve.var(),
            # 'max_revolutions': self.max_revolutions,
        }}

        return loss, stats

    def _calculate_3d_losses(self):
        """
        Calculate the how well the curve fits the control points.
        """

        # Calculate pairwise distances between the cloud points and the curve points
        dists = torch.cdist(self.curve_points, self.curve_target)
        min_points = 3  # todo: investigate

        # Only consider distances within a segment-length of each point
        cc = self.curve_points
        segment_lengths = torch.norm(cc[:, 1:] - cc[:, :-1], dim=-1)
        max_distance = segment_lengths.mean() * 2  # todo: investigate

        # Sort the distances relative to the each
        dists_curve_control, _ = dists.sort(dim=2)
        dists_control_curve, _ = dists.sort(dim=1)

        # Set any distances greater than the max to 0
        dists_curve_control[dists_curve_control > max_distance] = 0
        dists_control_curve[dists_control_curve > max_distance] = 0

        # Only consider the nearest 5% of the filtered cloud points to each curve point within distance cutoff
        n_cloud_points = max(min_points, int(self.curve_target.shape[1] * 0.05))
        dists_curve_control_filtered = dists_curve_control[:, :, :n_cloud_points]

        # Only consider the nearest 5% curve points to each cloud point
        n_curve_points = max(min_points, int(self.model_params.n_curve_points * 0.05))
        dists_control_curve_filtered = dists_control_curve[:, :n_curve_points]

        # Loss pulls in both directions
        loss_curve_control = torch.mean(dists_curve_control_filtered**2)
        loss_control_curve = torch.mean(dists_control_curve_filtered**2)
        loss = loss_curve_control + loss_control_curve

        # Add centre-point pull  # todo: is this necessary / does it help?
        loss_cp = torch.mean((self.curve_points.mean(dim=0) - self.curve_target.mean(dim=0))**2)
        loss = loss + loss_cp

        stats = {
            'loss/3d_curve_control': loss_curve_control,
            'loss/3d_control_curve': loss_control_curve,
            'loss/loss_cp': loss_cp,
            '3d_dists/curve_control/mean': dists.mean(),
            '3d_dist/curve_control/var': dists.var(),
        }

        return loss, stats

    def _calculate_curve_losses(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Losses for curve length and curvatures.
        """
        loss = 0
        stats = {}

        if self.model_params.curve_mode == ENCODING_MODE_DELTA_ANGLES:
            pre_angles = self.curve_parameters[:, 3:7]
            theta0_reg_loss = torch.mean((1 - torch.norm(pre_angles[:, :2], dim=1))**2)
            phi0_reg_loss = torch.mean((1 - torch.norm(pre_angles[:, 2:4], dim=1))**2)
            preangles_loss = theta0_reg_loss + phi0_reg_loss
            loss += preangles_loss
            stats['loss/preangles'] = preangles_loss

            # # Add worm-length loss - bigger worms are preferred
            # wl_loss = -torch.log(1 + self.curve_length.mean())
            # loss += wl_loss
            # stats['loss/worm_length'] = wl_loss

            return loss, stats

        # # Other encoding modes are fine by construction
        if self.curve_mode != ENCODING_MODE_POINTS:
            return loss, stats

        N = self.model_params.n_curve_points
        l = self.curve_length / (N - 1)
        cc = self.curve_points
        assert not is_bad(cc)

        # Calculate arc length
        segment_lengths = torch.norm(cc[:, 1:] - cc[:, :-1], dim=-1)
        arc_length = segment_lengths.sum()

        # Segment length loss
        # scale_sl = 50
        # sl_loss = scale_sl * ((segment_lengths - l)**2).sum()
        # sl_loss = torch.exp(scale_sl * ((segment_lengths - l).abs())).mean()
        sl_loss = torch.sum((segment_lengths - self.curve_length / N)**2)
        # sl_loss = segment_lengths.var()

        # Arc length loss
        al_loss = (arc_length - self.curve_length)**2

        # Calculate the maximum curvature (delta angle)
        max_delta_angle = self.model_params.max_revolutions * 2 * np.pi / N

        # Curvature loss
        a = cc[:, -2] - cc[:, 1:-1]
        b = cc[:, 2:] - cc[:, 1:-1]
        scale_al = 50
        min_dist_1hop = l * np.sqrt(2 * (1 - np.cos(np.pi - max_delta_angle)))
        curvature_loss = torch.exp(scale_al * (min_dist_1hop - torch.norm(a - b, dim=-1))).sum()
        # curvature_loss = (dists_loss/n_worm_points).sum()
        #
        # curvature_loss = ((angles.abs() - max_delta_angle)**2).sum()
        # curvature_loss = ((angles - max_delta_angle)**2).sum()
        # curvature_loss = ((angles / max_delta_angle)**4).sum() / 50000
        # if max_delta_angle > 0:
        #     curvature_loss = torch.clamp((angles / max_delta_angle)**4, max=100).mean()
        # else:
        #     curvature_loss = torch.clamp(angles**2, max=100).mean() * 10000
        # curvature_loss = angles[angles>max_delta_angle].sum() / 50000
        # curve_loss = sl_loss

        loss = sl_loss + curvature_loss
        # curve_loss = angles_loss + al_loss
        # curve_loss = al_loss
        # loss += sl_loss  # + angles_loss

        stats = {
            'arc_length': arc_length.item(),
            'seg_length/var': segment_lengths.var().item(),
            'loss/segment_length': sl_loss.item(),
            'loss/arc_length': al_loss.item(),
            'loss/curvature': curvature_loss.item(),
        }

        return loss, stats

    @staticmethod
    def _calculate_renders_losses(
            render: torch.Tensor,
            target: torch.Tensor,
            metric: str,
            multiscale: bool = False,
            calculate_all: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the losses between the given masks and the targets.
        """

        def loss_(m, x, y):
            if m == LOSS_MSE:
                l = F.mse_loss(x, y)
            elif m == LOSS_KL:
                l = F.kl_div(x, y)
            elif m == LOSS_LOGDIFF:
                l = torch.sum((torch.log(1 + x) - torch.log(1 + y))**2)
            return l

        stats = {}

        if multiscale:
            # Multiscale loss
            loss = 0
            masks_rep = render.clone()
            target_rep = target.clone()
            k = 1
            while masks_rep.shape[-1] > 1:
                loss += loss_(metric, masks_rep, target_rep)
                stats[f'loss/{metric}_{masks_rep.shape[-1]}'] = loss.item()

                # Downsample using 3x3 average pooling with stride of 2
                masks_rep = _avg_pool_2d(masks_rep, oob_grad_val=0)
                target_rep = _avg_pool_2d(target_rep, oob_grad_val=0)
                k += 1
        else:
            loss = loss_(metric, render, target)

        if calculate_all:
            for metric in [LOSS_MSE, LOSS_KL, LOSS_LOGDIFF]:
                stats[f'loss/{metric}'] = loss_(metric, render, target).item()

        return loss, stats

    def _get_curve_coordinates(self) -> torch.Tensor:
        """
        Decode the curve parameters into the 3D coordinates.
        """
        if self.model_params.curve_mode == ENCODING_MODE_POINTS:
            # Parameters are the curve coordinates
            return self.curve_parameters

        bs = self.curve_parameters.shape[0]

        # First 3 parameters are the offset
        offset = self.curve_parameters[:, :3]
        parameters = self.curve_parameters[:, 3:]

        N = self.model_params.n_curve_points
        if self.model_params.curve_mode == ENCODING_MODE_DELTA_VECTORS:
            # Remaining parameters are the delta vectors (de0s)
            delta_vectors = parameters.reshape((bs, N, 3))

            # Scale the ds's so that neighbouring points will be equidistant
            e0s = F.normalize(delta_vectors, dim=2)

        else:
            # Initial angles are unconstrained
            # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/11

            # theta: inclination - angle wrt z-axis: (0, pi)
            # phi: azimuth - rotation angle from x-y: (-pi, pi)
            pre_angles = parameters[:, :4]
            # pre_angles = F.hardtanh(pre_angles, min_val=-1, max_val=1)
            theta0 = (torch.atan2(pre_angles[:, 0], pre_angles[:, 1]) + np.pi) / 2
            phi0 = torch.atan2(pre_angles[:, 2], pre_angles[:, 3])
            parameters = parameters[:, 4:]

            # Determine maximum delta-angle
            max_delta_angle = self.model_params.max_revolutions * 2 * np.pi / N

            # Remaining parameters are the delta angles
            delta_angles = torch.tanh(parameters.reshape((bs, 2, -1))) * max_delta_angle

            # # Apply decay to delta angles so they go to 0 (ie, straight lines)
            # if self.decay_factor is not None:
            #     decay = torch.exp(
            #         -torch.arange(N, device=self.device) / N * self.decay_factor).repeat(bs)
            #     delta_angles = delta_angles * decay[1:].reshape((1, 1, N - 1))

            # Sum the initial angles with the delta angles to give the progression
            delta_thetas = torch.cat([theta0.unsqueeze(1), delta_angles[:, 0]], dim=-1)
            delta_phis = torch.cat([phi0.unsqueeze(1), delta_angles[:, 1]], dim=-1)
            thetas = torch.cumsum(delta_thetas, dim=-1)
            phis = torch.cumsum(delta_phis, dim=-1)

            # Convert to cartesian coordinates to find the e0 unit vectors
            e0s = torch.stack([
                torch.cos(phis) * torch.sin(thetas),
                torch.sin(phis) * torch.sin(thetas),
                torch.cos(thetas),
            ], dim=-1)

        # Scale the e0s (which have unit length) so the arc length is fixed
        tau = e0s * self.curve_length / N

        # Start at the offset and add the tangent vectors to form the curve
        curve_coordinates = offset + torch.cumsum(tau, dim=1)

        return curve_coordinates

    def _log_progress(self, cc_or_curve: str, step: int, final_step: int, loss: float, stats: dict):
        """
        Log the loss and stats to the tensorboard logger and command line.
        """
        log_msg = f'[{step}/{final_step - 1}]\tLoss: {loss:.7f}'
        checkpoint_step = getattr(self.checkpoint, f'step_{cc_or_curve}')
        self.tb_logger.add_scalar(f'{cc_or_curve}/loss', loss, checkpoint_step)
        for key, val in stats.items():
            self.tb_logger.add_scalar(f'{cc_or_curve}/{key}', val, checkpoint_step)
            if key in PRINT_KEYS:
                log_msg += f'\t{key}: {val:.5f}'
        logger.info(log_msg)

    def _make_plots(
            self,
            pre_step: bool = False,
            final_step: bool = False,
            show_cloud: bool = True,
            show_curve: bool = True,
            plot_sigmas: bool = False,
            plot_scores: bool = False,
    ):
        """
        Generate some plots.
        """
        # Select the idxs to plot
        bs = 1  # self.optimiser_args.batch_size
        n_examples = min(self.runtime_args.plot_n_examples, bs)
        idxs = np.random.choice(bs, n_examples, replace=False)

        # Make initial plots for all batch elements
        if pre_step:
            for idx in range(bs):
                self._plot_3d(idx, show_cloud, show_curve)
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and self.step % self.runtime_args.plot_every_n_steps == 0
        ):
            logger.info('Plotting.')
            for idx in idxs:
                self._plot_3d(idx, show_cloud, show_curve,
                              self.optimiser_args.loss_3d_cloud_threshold if show_curve else 0)
                self._plot_2d(idx, show_cloud, show_curve)
                if plot_sigmas:
                    self._plot_sigmas(idx)
                if plot_scores:
                    self._plot_scores(idx)

    def _plot_3d(
            self,
            idx: int,
            show_cloud: bool = True,
            show_curve: bool = True,
            cloud_point_threshold: float = 0,
    ):
        """
        Make a 3D scatter plot showing either or both of the cloud points and the curve points.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Rotate the perspective on every plot
        self.plot_3d_azim += 15
        ax.view_init(azim=self.plot_3d_azim)

        # Cloud points
        if show_cloud:
            cloud_points = self.cloud_points[idx]
            scores = self.cloud_points_scores[idx]
            if cloud_point_threshold > 0:
                above_threshold = scores > cloud_point_threshold
                scores = scores[above_threshold]
                cloud_points = cloud_points[above_threshold]
            x, y, z = (to_numpy(cloud_points[:, j]) for j in range(3))
            s1 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='autumn_r', s=20, alpha=0.4)
            fig.colorbar(s1)

        # Curve points
        if show_curve:
            curve_points = self.curve_points[idx]
            scores = self.curve_points_scores[idx]
            x, y, z = (to_numpy(curve_points[:, j]) for j in range(3))
            # s2 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='YlGnBu', s=50, marker='x', alpha=0.9)
            s2 = ax.scatter(x, y, z, color='black', s=75, marker='x', alpha=0.9)
            fig.colorbar(s2)

        ax.set_title(f'step_cc: {self.checkpoint.step_cc}. step_curve: {self.checkpoint.step_curve}.')
        fig.tight_layout()
        self._save_plot(fig, '3d')

    def _plot_2d(
            self,
            idx: int,
            show_cloud: bool = True,
            show_curve: bool = True
    ):
        """
        Plot the 2D mask renderings; target, cloud and/or curve.
        """
        X_target = to_numpy(self.masks_target[idx])
        X_cloud = to_numpy(self.masks_cloud[idx])
        X_curve = to_numpy(self.masks_curve[idx])

        n_rows = 1 + int(show_cloud) + int(show_curve)
        fig, axes = plt.subplots(n_rows, figsize=(8, 1 + 2 * n_rows))
        fig.suptitle(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame.frame_num}. '
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )

        # Stitch images and masks together
        image_triplet = np.concatenate(self.images, axis=1)
        X_target_triplet = np.concatenate(X_target, axis=1) / X_target.max()

        ax = axes[0]
        ax.set_title('Target')
        ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
        alphas = X_target_triplet.copy()
        alphas[alphas < 0.1] = 0
        alphas[alphas > 0.2] = 1
        ax.imshow(X_target_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
        ax.axis('off')
        row_idx = 1

        if show_cloud:
            X_cloud_triplet = np.concatenate(X_cloud, axis=1) / X_cloud.max()
            ax = axes[row_idx]
            ax.set_title('Cloud')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = X_cloud_triplet.copy()
            # alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X_cloud_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            ax.axis('off')
            row_idx += 1

        if show_curve:
            X_curve_triplet = np.concatenate(X_curve, axis=1) / X_curve.max()
            ax = axes[row_idx]
            ax.set_title('Curve')
            ax.imshow(image_triplet, cmap='gray', vmin=0, vmax=1)
            alphas = X_curve_triplet.copy()
            alphas[alphas < 0.1] = 0
            alphas[alphas > 0.2] = 1
            ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)
            ax.axis('off')

        fig.tight_layout()
        self._save_plot(fig, '2d')

    def _plot_sigmas(self, idx: int):
        """
        Plot the blur sigmas used for rendering.
        """
        sigmas_cloud = to_numpy(self.blur_sigmas_cloud[idx])
        sigmas_curve = to_numpy(self.blur_sigmas_curve[idx])

        n_rows = 2
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 8))
        fig.suptitle(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame.frame_num}. '
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )

        ax = axes[0, 0]
        ax.plot(sigmas_cloud)
        ax.set_title('blur_sigmas_cloud')

        ax = axes[0, 1]
        ax.plot(np.sort(sigmas_cloud))
        ax.set_title('sort(blur_sigmas_cloud)')

        ax = axes[1, 0]
        ax.plot(sigmas_curve)
        ax.set_title('blur_sigmas_curve')

        ax = axes[1, 1]
        ax.plot(np.sort(sigmas_curve))
        ax.set_title('sort(blur_sigmas_curve)')

        fig.tight_layout()
        self._save_plot(fig, 'sigmas')

    def _plot_scores(self, idx: int):
        """
        Plot the scores.
        """
        point_scores = to_numpy(self.cloud_points_scores[idx])
        n_rows = 2
        n_cols = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 8))
        fig.suptitle(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame.frame_num}. '
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )

        ax = axes[0]
        ax.plot(point_scores)
        ax.set_title('point_scores')

        ax = axes[1]
        ax.plot(np.sort(point_scores))
        ax.set_title('sort(point_scores)')

        fig.tight_layout()
        self._save_plot(fig, 'scores')

    def _save_plot(self, fig: Figure, plot_type: str):
        """
        Log the figure to the tensorboard logger and optionally save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path + f'/plots/{plot_type}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir + f'/{self.step:06d}.svg'
            plt.savefig(path, bbox_inches='tight')

        self.tb_logger.add_figure(plot_type, fig, self.step)
        self.tb_logger.flush()

        plt.close(fig)
