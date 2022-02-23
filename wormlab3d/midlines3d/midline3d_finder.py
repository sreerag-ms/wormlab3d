import os
from pathlib import Path
from typing import Tuple, Union, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mongoengine import DoesNotExist
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import gaussian_blur

from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGE_SIZE, START_TIMESTAMP
from wormlab3d.data.model import Trial, Cameras, MFCheckpoint, MFParameters, Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.args_finder import ParameterArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, CAM_PARAMETER_NAMES
from wormlab3d.midlines3d.mf_methods import generate_residual_targets, calculate_renders_losses, \
    calculate_neighbours_losses, calculate_parents_losses, calculate_aunts_losses, calculate_scores_losses, \
    calculate_sigmas_losses, calculate_intensities_losses, calculate_smoothness_losses, calculate_temporal_losses
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.nn.detector import ConvergenceDetector
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict, hash_data

cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'
img_extension = 'png'

PRINT_KEYS = [
    'loss/global',
    'loss/masks',
    'loss/neighbours',
    'loss/parents',
    'loss/aunts',
    'loss/temporal',
]

# torch.autograd.set_detect_anomaly(True)


class Midline3DFinder:
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            source_args: SourceArgs,
            parameter_args: ParameterArgs,
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.source_args = source_args
        self.parameter_args = parameter_args

        # Initialise the model and parameters
        self.model = self._init_model()
        self.parameters: MFParameters = self._init_parameters()

        # Initialise convergence detector
        self.convergence_detector = self._init_convergence_detector()

        # Check the devices
        self.device = self._init_devices()

        # Reconstruction
        self.reconstruction = self._init_reconstruction()

        # Load the trial and initialise trainable parameters
        self._init_trial()

        # Optimiser
        self.optimiser = self._init_optimiser()

        # Checkpoint
        self.checkpoint = self._init_checkpoint()

        # Plotting
        self.plot_3d_azim = -60

        # Loop vars
        self.frame_num = 0
        self.active_idx = 0

    @property
    def logs_path(self) -> Path:
        return self.get_logs_path(self.checkpoint)

    @staticmethod
    def get_logs_path(checkpoint: MFCheckpoint) -> Path:
        identifiers = {
            'parameters': str(checkpoint.parameters.id),
            **to_dict(checkpoint.source_args),
        }
        arg_hash = hash_data(identifiers)
        return LOGS_PATH / f'trial_{checkpoint.trial.id:03d}' / arg_hash

    @property
    def step(self):
        return self.checkpoint.step

    def __getattr__(self, key):
        """
        Allow batched parameters to be accessed as member variables.
        """
        if key in PARAMETER_NAMES + BUFFER_NAMES:
            return [fs.get_state(key) for fs in self.frame_batch]

    def _init_model(self) -> ProjectRenderScoreModel:
        """
        Build the model.
        """
        logger.info(f'Initialising model.')
        model = ProjectRenderScoreModel(
            image_size=PREPARED_IMAGE_SIZE[0]
        )
        model = torch.jit.script(model)
        return model

    def _init_parameters(self) -> MFParameters:
        """
        Build the model.
        """
        logger.info(f'Initialising parameters.')
        parameters = None
        params = self.parameter_args.get_db_params()

        # Try to load an existing model
        if self.parameter_args.load:
            # If we have a model id then load this from the database
            if self.parameter_args.params_id is not None:
                parameters = MFParameters.objects.get(id=self.parameter_args.params_id)
            else:
                # Otherwise, try to find one matching the same parameters
                params_matching = MFParameters.objects(**params)
                if params_matching.count() > 0:
                    parameters = params_matching[0]
                    logger.info(
                        f'Found {len(params_matching)} suitable parameter records in database, using most recent.')
                else:
                    logger.info(f'No suitable parameter records found in database.')
            if parameters is not None:
                logger.info(f'Loaded parameters (id={parameters.id}, created={parameters.created}).')

        # Not loaded model, so create one
        if parameters is None:
            parameters = MFParameters(**params)
            parameters.save()
            logger.info(f'Saved parameters to database (id={parameters.id})')

        return parameters

    def _init_convergence_detector(self) -> ConvergenceDetector:
        """
        Initialise the convergence detector.
        """
        logger.info(f'Initialising convergence detector.')
        cd = ConvergenceDetector(
            shape=(1 + self.parameters.depth,),
            tau_fast=self.parameters.convergence_tau_fast,
            tau_slow=self.parameters.convergence_tau_slow,
            threshold=self.parameters.convergence_threshold,
            patience=self.parameters.convergence_patience
        )
        cd = torch.jit.script(cd)
        return cd

    def _init_reconstruction(self) -> Reconstruction:
        """
        Load or create the reconstruction record.
        """
        logger.info('Initialising reconstruction.')
        params = {
            'trial': int(self.source_args.trial_id),
            'source': M3D_SOURCE_MF,
            'mf_parameters': self.parameters
        }
        start_frame = self.source_args.start_frame

        # Look for existing reconstruction
        reconstruction = None
        try:
            reconstruction = Reconstruction.objects.get(**params)
            if reconstruction.start_frame < start_frame:
                reconstruction.start_frame = start_frame
                reconstruction.save()
            logger.info(f'Loaded reconstruction (id={reconstruction.id}, created={reconstruction.created}).')
        except DoesNotExist:
            logger.info(f'No reconstruction record found in database.')

        if reconstruction is None:
            reconstruction = Reconstruction(
                **params,
                start_frame=start_frame,
                end_frame=start_frame
            )
            reconstruction.save()
            logger.info(f'Saved reconstruction record to database (id={reconstruction.id})')

        return reconstruction

    def _init_trial(self):
        """
        Load the trial.
        """
        logger.info('Initialising trial state.')
        self.trial: Trial = Trial.objects.get(id=self.source_args.trial_id)

        # Look for existing reconstruction
        reconstruction = None
        try:
            reconstruction = Reconstruction.objects.get(
                trial=self.trial,
                source=M3D_SOURCE_MF,
                mf_parameters=self.parameters,
            )
        except DoesNotExist:
            pass

        if reconstruction is None:
            reconstruction = Reconstruction(
                trial=self.trial,
                source=M3D_SOURCE_MF,
                mf_parameters=self.parameters,
                start_frame=self.source_args.start_frame,
                end_frame=self.source_args.start_frame
            )
            reconstruction.save()
            logger.info('Created reconstruction')

        # Prepare trial state
        self.trial_state = TrialState(
            reconstruction=self.reconstruction,
            start_frame=self.source_args.start_frame,
            end_frame=self.source_args.end_frame,
            read_only=False
        )

        # Master state
        self.master_frame_state = self._init_frame_state(self.trial_state.frame_nums[0])

        # Prepare batch state
        self.frame_batch: List[FrameState] = []
        mfs = self.master_frame_state if self.parameters.use_master else None
        for i in range(self.parameters.window_size):
            fs = self._init_frame_state(self.trial_state.frame_nums[i], master_frame_state=mfs)
            self.frame_batch.append(fs)

        # Previous frame state
        self.prev_frame_state: FrameState = None

    def _init_frame_state(
            self,
            frame_num: int,
            prev_frame_state: FrameState = None,
            master_frame_state: FrameState = None
    ) -> FrameState:
        """
        Load the frame.
        """
        frame = self.trial.get_frame(frame_num)
        logger.info(f'Initialising frame state for frame #{frame_num} (id={frame.id}).')

        # Load images
        if frame.images is None or len(frame.images) != 3:
            raise RuntimeError('Frame does not have a triplet of prepared images. Aborting!')
        images = torch.from_numpy(np.stack(frame.images))

        # Threshold images to make the target masks
        masks_fr = torch.zeros_like(images)
        masks_fr[images >= self.parameters.masks_threshold] = 1
        masks_fr[images < self.parameters.masks_threshold] = 0
        masks_fr = masks_fr.unsqueeze(0)

        # Linearly interpolate target mask resolutions from 8x8 to 200x200
        sizes = torch.linspace(8, PREPARED_IMAGE_SIZE[0], self.parameters.depth).to(torch.int32)

        # Generate downsampled target masks
        masks = []
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
            masks.append(masks_rs)

        # Load cameras
        cameras: Cameras = frame.get_cameras()

        # Use the fixed (interpolated and smoothed) centre point if available
        if frame.centre_3d_fixed is not None:
            p3d = frame.centre_3d_fixed
        else:
            p3d = frame.centre_3d

        # Initialise frame state
        frame_state = FrameState(
            frame_num=frame_num,
            images=images,
            masks_target=masks,
            cameras=cameras,
            points_3d_base=torch.tensor(p3d.point_3d),
            points_2d_base=torch.tensor(p3d.reprojected_points_2d),
            parameters=self.parameters,
            prev_frame_state=prev_frame_state,
            master_frame_state=master_frame_state
        )
        frame_state.to(self.device)

        return frame_state

    def _init_optimiser(self) -> Optimizer:
        """
        Set up the joint cameras and cloud optimiser and the curve optimiser.
        """
        logger.info('Initialising optimiser.')
        p = self.parameters

        if p.use_master:
            params = {
                k: self.master_frame_state.get_state(k)
                for k in PARAMETER_NAMES
            }
        else:
            params = {
                k: [fs.get_state(k) for fs in self.frame_batch]
                for k in PARAMETER_NAMES
            }

        optimiser_cls: Optimizer = getattr(torch.optim, p.algorithm)
        cam_params = [params[f'cam_{k}'] for k in CAM_PARAMETER_NAMES]

        params = [
            {'params': cam_params, 'lr': p.lr_cam_coeffs},
            {'params': params['points'], 'lr': p.lr_points},
            {'params': params['sigmas'], 'lr': p.lr_sigmas},
            {'params': params['intensities'], 'lr': p.lr_intensities},
            {'params': params['camera_sigmas'], 'lr': p.lr_sigmas},
            {'params': params['camera_intensities'], 'lr': p.lr_intensities},
        ]

        optimiser = optimiser_cls(
            params=params,
            weight_decay=0
        )

        return optimiser

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
            device = torch.device(f'cuda:{self.runtime_args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info('Using GPU.')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
        else:
            if cpu_or_gpu == 'gpu':
                raise RuntimeError('GPU requested but not available. Aborting.')
            logger.info('Using CPU.')

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
                if self.runtime_args.resume_from in ['latest', 'best']:
                    if self.runtime_args.resume_from == 'latest':
                        order_by = '-created'
                    else:
                        order_by = '+loss'

                    prev_checkpoints = MFCheckpoint.objects(
                        trial=self.trial,
                        parameters=self.parameters
                    ).order_by(order_by)

                    if prev_checkpoints.count() > 0:
                        logger.info(
                            f'Found {prev_checkpoints.count()} previous checkpoints. '
                            f'Using {self.runtime_args.resume_from}.'
                        )
                        prev_checkpoint = prev_checkpoints[0]
                    else:
                        logger.error(
                            f'Found no checkpoints for trial={self.trial.id} and model={self.parameters.id}.'
                        )
                        raise DoesNotExist()
                else:
                    prev_checkpoint = MFCheckpoint.objects.get(
                        id=self.runtime_args.resume_from
                    )
                logger.info(f'Loaded checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}.')
                if self.source_args.trial_id is not None:
                    logger.info(f'Frame number = {prev_checkpoint.frame_num}')
                logger.info(f'Loss = {prev_checkpoint.loss:.6f}')
                if len(prev_checkpoint.metrics) > 0:
                    logger.info('Metrics:')
                    for key, val in prev_checkpoint.metrics.items():
                        logger.info(f'\t{key}: {val:.4E}')
            except DoesNotExist:
                raise RuntimeError(f'Could not load checkpoint={self.runtime_args.resume_from}')

        # Either clone the previous checkpoint to use as the starting point
        if prev_checkpoint is not None:
            checkpoint = prev_checkpoint.clone()

            # Update the model references to the ones now in use
            if checkpoint.parameters.id != self.parameters.id:
                logger.warning('Parameters have changed! This may cause problems!')
                checkpoint.parameters = self.parameters

            # Args are stored against the checkpoint, so just override them
            checkpoint.runtime_args = to_dict(self.runtime_args)
            checkpoint.source_args = to_dict(self.source_args)
            checkpoint.parameter_args = to_dict(self.parameter_args)

        # ..or start a new checkpoint
        else:
            checkpoint = MFCheckpoint(
                trial=self.trial,
                parameters=self.parameters,
                frame_num=self.source_args.start_frame,
                runtime_args=to_dict(self.runtime_args),
                source_args=to_dict(self.source_args),
            )

        return checkpoint

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.logs_path / 'events' / START_TIMESTAMP, max_queue=100, flush_secs=30)

    def _configure_paths(self):
        """
        Initialise the logs and plot directories
        """
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path / 'events', exist_ok=True)
        os.makedirs(self.logs_path / 'plots', exist_ok=True)

    def save_checkpoint(self):
        """
        Save the checkpoint information to the database and the parameters to file.
        """
        logger.info('Saving checkpoint...')
        self.checkpoint.save()

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint
        self.checkpoint = self.checkpoint.clone()

        # Update checkpoint in TrialState
        self.trial_state.checkpoint = self.checkpoint

    def process_trial(self):
        """
        Process the trial.
        """
        p = self.parameters
        self._configure_paths()
        self._init_tb_logger()

        # Initial plots
        self._make_plots(pre_step=True)

        # Train
        w2 = int((p.window_size - 1) / 2)
        first_frame = self.checkpoint.frame_num
        n_frames = len(self.trial_state) - first_frame + 1

        for i, frame_num in enumerate(range(first_frame, self.trial_state.frame_nums[-1])):
            logger.info(f'======== Training frame #{frame_num} ({i}/{n_frames}) ========')
            active_idx = min(i, w2)
            self.frame_num = frame_num
            self.active_idx = active_idx

            # Reset convergence detection
            self.convergence_detector.reset_counters()

            # Reset frame step counter and train the batch
            self.checkpoint.step_frame = 0
            self.checkpoint.frame_num = frame_num
            self.train(frame_num == first_frame)

            # Save the state
            self.trial_state.update_frame_state(frame_num, self.master_frame_state)
            self.trial_state.save()

            # Update the reconstruction
            self.reconstruction.end_frame = max(frame_num + 1, self.reconstruction.end_frame)
            self.reconstruction.save()

            # Make plots
            if self.runtime_args.plot_every_n_frames > -1 \
                    and (i + 1) % self.runtime_args.plot_every_n_frames == 0:
                self._make_plots(final_step=True)

            # Roll window
            next_frame = self._init_frame_state(frame_num + w2 + 1, self.master_frame_state)
            if p.use_master:
                self.master_frame_state.frame_num = next_frame.frame_num
                with torch.no_grad():
                    for k in BUFFER_NAMES + PARAMETER_NAMES:
                        self.master_frame_state.set_state(k, next_frame.get_state(k))

            for j in range(p.window_size):
                curr_frame = self.frame_batch[j]
                # if j < min(i+1, w2):
                #     curr_frame.freeze()

                if i >= w2:
                    if j + 1 < p.window_size:
                        next_frame = self.frame_batch[j + 1]
                    elif i + w2 < len(self.trial_state):
                        next_frame = self._init_frame_state(frame_num + w2 + 1, curr_frame)
                    else:
                        continue

                    with torch.no_grad():
                        for k in BUFFER_NAMES + PARAMETER_NAMES:
                            curr_frame.set_state(k, next_frame.get_state(k))
                    curr_frame.frame_num = next_frame.frame_num

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_frames > 0 \
                    and frame_num % self.runtime_args.checkpoint_every_n_frames == 0:
                self.save_checkpoint()

    def train(self, first_frame: bool = False):
        """
        Train the camera coefficients and multiscale curve.
        """
        logger.info('----- Training the camera coefficients and multiscale curve -----')
        p = self.parameters
        self.optimiser = self._init_optimiser()
        max_steps = p.n_steps_init if first_frame else p.n_steps_max
        start_step = self.checkpoint.step_frame + 1
        final_step = start_step + max_steps

        # Get previous frame state
        if self.checkpoint.frame_num > self.reconstruction.start_frame:
            self.prev_frame_state = self._init_frame_state(self.checkpoint.frame_num - 1)
            self.prev_frame_state.freeze()
        else:
            self.prev_frame_state = None

        # Train the cam coeffs and multiscale curve
        for step in range(start_step, final_step):
            loss, loss_global, losses_depths, stats = self._train_step()

            # Update steps and checkpoint stats
            self.checkpoint.step += 1
            self.checkpoint.step_frame += 1
            self.checkpoint.loss = loss.item()
            self.checkpoint.metrics = stats

            # Log
            self._log_progress(
                step=step,
                final_step=final_step,
                loss=loss,
                stats=stats
            )

            # Make plots
            self._make_plots()

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and self.step % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

            # Update convergence detector
            losses = torch.tensor(
                [loss_global, *losses_depths],
                device=self.device
            )
            self.convergence_detector.forward(losses)

            # When all of the losses have converged, break
            if not first_frame and self.convergence_detector.converged.all():
                break

        self.tb_logger.add_scalar('train_steps', self.checkpoint.step_frame, self.checkpoint.frame_num)

    def _train_step(self) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Train the cam coeffs and multiscale curve for a single step.
        """
        p = self.parameters
        D = p.depth

        # Collect parameters
        if p.use_master:
            cam_coeffs = self.master_frame_state.get_state('cam_coeffs').unsqueeze(0)
            cam_rotation_preangles = self.master_frame_state.get_state('cam_rotation_preangles').unsqueeze(0)
            points_3d_base = self.master_frame_state.get_state('points_3d_base').unsqueeze(0)
            points_2d_base = self.master_frame_state.get_state('points_2d_base').unsqueeze(0)
            camera_sigmas = self.master_frame_state.get_state('camera_sigmas').unsqueeze(0)
            camera_intensities = self.master_frame_state.get_state('camera_intensities').unsqueeze(0)
            points = [self.master_frame_state.get_state('points')[d].unsqueeze(0) for d in range(D)]
            masks_target = [self.master_frame_state.get_state('masks_target')[d].unsqueeze(0) for d in range(D)]
            sigmas = [self.master_frame_state.get_state('sigmas')[d].unsqueeze(0) for d in range(D)]
            intensities = [self.master_frame_state.get_state('intensities')[d].unsqueeze(0) for d in range(D)]
        else:
            cam_coeffs = torch.stack([f.get_state('cam_coeffs') for f in self.frame_batch])
            cam_rotation_preangles = torch.stack([f.get_state('cam_rotation_preangles') for f in self.frame_batch])
            points_3d_base = torch.stack([f.get_state('points_3d_base') for f in self.frame_batch])
            points_2d_base = torch.stack([f.get_state('points_2d_base') for f in self.frame_batch])
            camera_sigmas = torch.stack([f.get_state('camera_sigmas') for f in self.frame_batch])
            camera_intensities = torch.stack([f.get_state('camera_intensities') for f in self.frame_batch])
            points = [torch.stack([f.get_state('points')[d] for f in self.frame_batch]) for d in range(D)]
            masks_target = [torch.stack([f.get_state('masks_target')[d] for f in self.frame_batch]) for d in range(D)]
            sigmas = [torch.stack([f.get_state('sigmas')[d] for f in self.frame_batch]) for d in range(D)]
            intensities = [torch.stack([f.get_state('intensities')[d] for f in self.frame_batch]) for d in range(D)]

        # Generate the outputs
        masks, points_2d, scores, points_smoothed, sigmas_smoothed, intensities_smoothed = self.model.forward(
            cam_coeffs=cam_coeffs,
            points_3d_base=points_3d_base,
            points_2d_base=points_2d_base,
            points=points,
            masks_target=masks_target,
            sigmas=sigmas,
            intensities=intensities,
            camera_sigmas=camera_sigmas,
            camera_intensities=camera_intensities,
        )

        # Generate targets with added residuals
        masks_target_residuals = generate_residual_targets(masks_target, masks)

        # Calculate the losses
        loss, loss_global, losses_depths, stats = self._calculate_losses(
            cam_rotation_preangles=cam_rotation_preangles,
            points=points,
            masks_target=masks_target_residuals,
            sigmas=sigmas,
            intensities=intensities,
            camera_sigmas=camera_sigmas,
            camera_intensities=camera_intensities,
            masks=masks,
            scores=scores,
            points_smoothed=points_smoothed,
            sigmas_smoothed=sigmas_smoothed,
            intensities_smoothed=intensities_smoothed,
        )

        # Take optimisation step
        if is_bad(loss):
            logger.warning('Bad loss, skipping parameter update.')
        else:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            # Clamp the sigmas and intensities
            with torch.no_grad():
                sigs = [*self.master_frame_state.get_state('sigmas'), ]
                sigs.extend(*[f.get_state('sigmas') for f in self.frame_batch])
                for sigs_i in sigs:
                    sigs_i.data = sigs_i.clamp(min=5e-3)

                ints = [*self.master_frame_state.get_state('intensities'), ]
                ints.extend(*[f.get_state('intensities') for f in self.frame_batch])
                for ints_i in ints:
                    ints_i.data = ints_i.clamp(min=1e-2, max=1.5)

        # Update master state
        self.master_frame_state.set_state('masks_curve', [masks[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('masks_target_residuals',
                                          [masks_target_residuals[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('points_2d', [points_2d[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('scores', [scores[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_stats(stats)

        # Update batch state
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('masks_curve', [masks[d][i] for d in range(D)])
            fs.set_state('masks_target_residuals', [masks_target_residuals[d][i] for d in range(D)])
            fs.set_state('points_2d', [points_2d[d][i] for d in range(D)])
            fs.set_state('scores', [scores[d][i] for d in range(D)])
            fs.set_stats(stats)

        return loss, loss_global, losses_depths, stats

    def _calculate_losses(
            self,
            cam_rotation_preangles: torch.Tensor,
            points: torch.Tensor,
            masks_target: torch.Tensor,
            sigmas: torch.Tensor,
            intensities: torch.Tensor,
            camera_sigmas: torch.Tensor,
            camera_intensities: torch.Tensor,
            masks: torch.Tensor,
            scores: torch.Tensor,
            points_smoothed: torch.Tensor,
            sigmas_smoothed: torch.Tensor,
            intensities_smoothed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Dict[str, float]]:
        """
        Calculate the losses.
        """
        p = self.parameters
        stats = {}
        loss_global = 0.
        losses_depths = []

        def _log_parameter_stats(d_, key_, var_):
            stats[f'{key_}/{d_}/mean'] = var_.mean()
            if d_ > 1:
                stats[f'{key_}/{d_}/var'] = var_.var()

        # Previous points used for temporal losses
        if self.prev_frame_state is not None:
            points_prev = self.prev_frame_state.points
        else:
            points_prev = None

        # Losses calculated at each depth
        losses = {
            # 'masks': self._calculate_mask_losses(masks_target, masks),
            'masks': calculate_renders_losses(masks, masks_target, p.loss_masks_metric, p.loss_masks_multiscale),
            'neighbours': calculate_neighbours_losses(points),
            'parents': calculate_parents_losses(points),
            'aunts': calculate_aunts_losses(points),
            'scores': calculate_scores_losses(scores),
            'sigmas': calculate_sigmas_losses(sigmas, sigmas_smoothed),
            'intensities': calculate_intensities_losses(intensities, intensities_smoothed),
            'smoothness': calculate_smoothness_losses(points, points_smoothed),
            'temporal': calculate_temporal_losses(points, points_prev),
        }

        # Log the total loss for each type
        for k, losses_k in losses.items():
            loss_k = sum(losses_k)
            stats[f'loss/{k}'] = loss_k.item()

        # Sum the losses at each depth
        for d in range(p.depth):
            loss_d = 0.
            for k, losses_k in losses.items():
                w = getattr(p, f'loss_{k}')
                if w > 0:
                    loss_d += w * losses_k[d]
            stats[f'loss/depth/{d}'] = loss_d.item()
            losses_depths.append(loss_d)

            # Log actual losses at each depth
            if self.runtime_args.log_level > 0:
                for k in losses.keys():
                    stats[f'loss_d/{k}/{d}'] = losses[k][d].item()

            # Log additional stats
            if self.runtime_args.log_level > 1:
                if d > 0:
                    # Track distance to neighbours
                    dist_neighbours = torch.norm(points[d][:, 1:] - points[d][:, -1], dim=-1)
                    _log_parameter_stats(d, 'dists/neighbours', dist_neighbours)

                    # Track distance to parent
                    curve_points_parent = torch.repeat_interleave(points[d - 1], repeats=2, dim=1)
                    dist_parent = torch.norm(points[d] - curve_points_parent, dim=-1)
                    _log_parameter_stats(d, 'dists/parent', dist_parent)

                # Scores, sigmas, intensities
                _log_parameter_stats(d, 'scores', scores[d])
                _log_parameter_stats(d, 'sigmas', sigmas[d])
                _log_parameter_stats(d, 'intensities', intensities[d])

        # Sigma cameras sfs should average 1
        camera_sigmas_loss = torch.sum((camera_sigmas - 1)**2)
        loss_global += camera_sigmas_loss
        stats['loss/camera_sigmas'] = camera_sigmas_loss.item()

        # Intensities cameras sfs should average 1
        camera_intensities_loss = torch.sum((camera_intensities - 1)**2)
        loss_global += camera_intensities_loss
        stats['loss/camera_intensities'] = camera_intensities_loss.item()

        # Camera rotation preangles loss
        preangles_loss = torch.mean((1 - torch.norm(cam_rotation_preangles, dim=3))**2)
        loss_global += preangles_loss
        stats['loss/cam_preangles'] = preangles_loss

        # Log the combined global losses
        stats['loss/global'] = loss_global.item()

        # Log additional stats if required
        if self.runtime_args.log_level > 1:
            stats['scores/total'] = torch.cat(scores, dim=1).sum()

            # Log the scale factors
            for i in range(3):
                stats[f'camera_sigmas/{i}'] = camera_sigmas[:, i].mean()
                stats[f'camera_intensities/{i}'] = camera_intensities[:, i].mean()

        # Sum the global losses with the losses generated at each depth
        loss = sum(losses_depths) + loss_global
        stats['loss/total'] = loss.item()

        return loss, loss_global, losses_depths, stats

    def _log_progress(self, step: int, final_step: int, loss: float, stats: dict):
        """
        Log the loss and stats to the tensorboard logger and command line.
        """
        log_msg = f'[{step}/{final_step - 1}]\tLoss: {loss:.5E}'
        step = self.checkpoint.step
        for key, val in stats.items():
            self.tb_logger.add_scalar(key, val, step)
            if key in PRINT_KEYS:
                log_msg += f'\t{key}: {val:.3E}'
        logger.info(log_msg)

        # Log camera coefficients
        if self.runtime_args.log_level > 0:
            # Extract parameters
            intrinsics = self.master_frame_state.get_state('cam_intrinsics')
            translation = self.master_frame_state.get_state('cam_translations')
            distortion = self.master_frame_state.get_state('cam_distortions')
            shifts = self.master_frame_state.get_state('cam_shifts')

            # Rotation angles
            rotation_preangles = self.master_frame_state.get_state('cam_rotation_preangles')
            rotation_angles = torch.atan2(rotation_preangles[:, :, 0], rotation_preangles[:, :, 1])

            # Log
            for i in range(3):
                self.tb_logger.add_scalar(f'cam_coeffs/fx/{i}', intrinsics[i, 0].item(), step)
                self.tb_logger.add_scalar(f'cam_coeffs/fy/{i}', intrinsics[i, 1].item(), step)
                self.tb_logger.add_scalar(f'cam_coeffs/cx/{i}', intrinsics[i, 2].item(), step)
                self.tb_logger.add_scalar(f'cam_coeffs/cy/{i}', intrinsics[i, 3].item(), step)
                for j in range(3):
                    self.tb_logger.add_scalar(f'cam_coeffs/t[{j}]/{i}', translation[i, j].item(), step)
                for j in range(5):
                    self.tb_logger.add_scalar(f'cam_coeffs/d[{j}]/{i}', distortion[i, j].item(), step)
                self.tb_logger.add_scalar(f'cam_coeffs/shifts/{i}', shifts[i].item(), step)

                for j, angle in enumerate(['phi', 'theta', 'psi']):
                    self.tb_logger.add_scalar(
                        f'cam_rotation_angles/{i}/{angle}',
                        rotation_angles[i, j].item(),
                        step
                    )

        # Log the convergence detection parameters
        if self.runtime_args.log_level > 1:
            detector = self.convergence_detector
            if detector is not None:
                state_vars = ['mu_fast', 'mu_slow', 'convergence_count', 'converged']
                for d in range(self.parameters.depth + 1):
                    d_str = 'global' if d == 0 else d - 1
                    for k in state_vars:
                        self.tb_logger.add_scalar(f'detector/{d_str}/{k}', getattr(detector, k)[d].item(), step)

    def _make_plots(
            self,
            pre_step: bool = False,
            final_step: bool = False,
    ):
        """
        Generate some plots.
        """

        # Make initial plots for all batch elements
        if pre_step:
            self._plot_3d()
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and self.step % self.runtime_args.plot_every_n_steps == 0
        ):
            logger.info('Plotting.')
            self._plot_3d()
            self._plot_2d()
            self._plot_point_stats()

    def _plot_3d(self):
        """
        Make a multiscale curve 3D scatter plot.
        """
        cmap_vertices = 'autumn_r'
        cmap_curve = 'jet'
        D = self.parameters.depth
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        # Rotate the perspective on every plot
        self.plot_3d_azim += 1

        fig = plt.figure(figsize=(n_cols * 4, 1 + n_rows * 4))
        fig.suptitle(self._plot_title())
        gs = GridSpec(n_rows, n_cols)
        d = 0

        points = self.master_frame_state.get_state('points')
        sigmas = self.master_frame_state.get_state('sigmas')
        scores = self.master_frame_state.get_state('scores')

        for i in range(n_rows):
            for j in range(n_cols):
                if d >= D:
                    break
                ax = fig.add_subplot(gs[i, j], projection='3d')
                ax.view_init(azim=self.plot_3d_azim)
                ax.set_title(f'd={d}')
                # cla(ax)

                # Scatter vertices
                vertices = to_numpy(points[d])
                scores_d = to_numpy(scores[d])
                sigmas_d = np.clip(
                    2000 * to_numpy(sigmas[d]),
                    a_min=10,
                    a_max=1000,
                )
                x, y, z = (vertices[:, j] for j in range(3))
                ax.scatter(x, y, z, c=scores_d, cmap=cmap_vertices, s=sigmas_d, alpha=0.4)

                # # Draw lines connecting points
                colours = np.linspace(0, 1, len(vertices))
                v2 = vertices[:, None, :]
                segments = np.concatenate([v2[:-1], v2[1:]], axis=1)
                lc = Line3DCollection(segments, array=colours, cmap=cmap_curve, zorder=-2, alpha=0.2)
                ax.add_collection(lc)

                # Find axes limits
                if d > 0:
                    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
                    ax.set_box_aspect(np.ptp(limits, axis=1))

                d += 1

        fig.tight_layout()
        self._save_plot(fig, '3D')

    def _plot_2d(self):
        """
        Plot the 2D mask renderings of the mutiscale curves.
        """
        D = self.parameters.depth
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        fig = plt.figure(figsize=(n_cols * 3, 1 + n_rows * 3))
        fig.suptitle(self._plot_title())
        gs = GridSpec(n_rows, n_cols)
        d = 0

        masks_target = self.master_frame_state.get_state('masks_target_residuals')
        points_2d = self.master_frame_state.get_state('points_2d')
        masks_curve = self.master_frame_state.get_state('masks_curve')

        images = to_numpy(self.master_frame_state.get_state('images'))
        image_triplet = np.concatenate(images, axis=1)
        image_grid = np.concatenate([np.ones_like(image_triplet), image_triplet, np.ones_like(image_triplet)], axis=0)
        M, N = tuple(image_grid.shape)

        scatter_sizes = (np.linspace(1, 0, D)**2 * 100) + 0.3

        for i in range(n_rows):
            for j in range(n_cols):
                if d >= D:
                    break
                ax = fig.add_subplot(gs[i, j])
                ax.set_title(f'd={d}')
                ax.axis('off')

                X_target = to_numpy(masks_target[d])
                X_curve = to_numpy(masks_curve[d])

                # Stitch images and masks together
                ax.imshow(image_grid, cmap='gray', vmin=0, vmax=1)
                ax.set_xlim((0, N))
                ax.set_ylim((M, 0))

                # Target overlay
                X_target_triplet = np.concatenate(X_target, axis=1)
                ax.imshow(X_target_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal',
                          extent=(0, N - 1, int(M / 3), 0))

                # Curve overlay
                X_curve_triplet = np.concatenate(X_curve, axis=1) / max(1e-5, X_curve.max())
                alphas = X_curve_triplet.copy() * 0.5
                ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                          extent=(0, N - 1, 2 * int(M / 3), int(M / 3)))

                # Scatter the midline points
                p2d = to_numpy(points_2d[d]).transpose(1, 0, 2)
                for k in range(3):
                    p = p2d[k] + (0, 200)
                    if k == 1:
                        p += (200, 0)
                    elif k == 2:
                        p += (400, 0)
                    ax.scatter(p[:, 0], p[:, 1], cmap='jet', c=np.linspace(0, 1, len(p)), s=scatter_sizes[d], alpha=0.6)

                # Errors
                errors_triplet = X_curve_triplet - X_target_triplet
                errors_triplet = errors_triplet / np.abs(errors_triplet).max()
                ax.imshow(errors_triplet, vmin=-1, vmax=1, cmap='PRGn', aspect='equal',
                          extent=(0, N - 1, M - 1, 2 * int(M / 3)))

                d += 1

        fig.tight_layout()
        self._save_plot(fig, '2D')

    def _plot_point_stats(self):
        """
        Plot the point scores, sigmas and intensities for the curves.
        """
        ra = self.runtime_args
        D = self.parameters.depth
        cmap = plt.get_cmap('jet')
        scores = self.master_frame_state.get_state('scores')
        sigmas = self.master_frame_state.get_state('sigmas')
        intensities = self.master_frame_state.get_state('intensities')

        n_rows = int(ra.plot_scores) + int(ra.plot_sigmas) + int(ra.plot_intensities)
        if n_rows == 0:
            return

        fig, axes = plt.subplots(n_rows, figsize=(8, 8), squeeze=False)
        axes = axes[:, 0]

        i = 0
        if ra.plot_scores:
            ax_scores = axes[i]
            ax_scores.set_title('Scores')
            i += 1
        if ra.plot_sigmas:
            ax_sigmas = axes[i]
            camera_sigmas = self.master_frame_state.get_state('camera_sigmas')
            sfs = ', '.join([f'{sf:.3f}' for sf in camera_sigmas])
            ax_sigmas.set_title(f'sigmas\nCameras: {sfs}.')
            i += 1
        if ra.plot_intensities:
            ax_intensities = axes[i]
            camera_intensities = self.master_frame_state.get_state('camera_intensities')
            sfs = ', '.join([f'{sf:.3f}' for sf in camera_intensities])
            ax_intensities.set_title(f'Intensities\nCameras: {sfs}.')

        fig.suptitle(self._plot_title())

        colours = [cmap(d) for d in np.linspace(0, 1, D)]
        positions = [
            np.linspace(0, 1, 2**d + 2)[1:-1]
            for d in range(D)
        ]

        for d in range(D):
            plot_args = {'label': f'd={d}', 'color': colours[d], 'alpha': 0.5}
            scatter_args = {'color': colours[d], 'alpha': 0.8, 's': 10}

            # Scores
            if ra.plot_scores:
                scores_d = to_numpy(scores[d])
                ax_scores.plot(positions[d], scores_d, **plot_args)
                ax_scores.scatter(x=positions[d], y=scores_d, **scatter_args)

            # Sigmas
            if ra.plot_sigmas:
                sigmas_d = to_numpy(sigmas[d])
                ax_sigmas.plot(positions[d], sigmas_d, **plot_args)
                ax_sigmas.scatter(x=positions[d], y=sigmas_d, **scatter_args)

            # Intensities
            if ra.plot_intensities:
                intensities_d = to_numpy(intensities[d])
                ax_intensities.plot(positions[d], intensities_d, **plot_args)
                ax_intensities.scatter(x=positions[d], y=intensities_d, **scatter_args)

        fig.tight_layout()
        self._save_plot(fig, 'point_stats')

    def _plot_title(self) -> str:
        return f'Trial: {self.trial.id}. {self.trial.date:%Y%m%d}. ' \
               f'Frame: {self.master_frame_state.frame_num}/{self.trial_state.frame_nums[-1]}.\n' \
               f'Global step: {self.checkpoint.step}. Frame step: {self.checkpoint.step_frame}.'

    def _save_plot(self, fig: Figure, plot_type: str):
        """
        Either log the figure to the tensorboard logger or save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path / 'plots' / plot_type
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir / f'{self.frame_num:05d}_{self.step:06d}.{img_extension}'
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(plot_type, fig, self.step)
            self.tb_logger.flush()

        plt.close(fig)
