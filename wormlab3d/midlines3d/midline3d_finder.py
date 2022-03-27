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

from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGE_SIZE, START_TIMESTAMP
from wormlab3d.data.model import Trial, MFCheckpoint, MFParameters, Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.args_finder import ParameterArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, CAM_PARAMETER_NAMES, \
    TRANSIENTS_NAMES, CURVATURE_PARAMETER_NAMES
from wormlab3d.midlines3d.mf_methods import generate_residual_targets, calculate_renders_losses, \
    calculate_scores_losses, calculate_smoothness_losses, calculate_neighbours_losses, calculate_parents_losses, \
    calculate_aunts_losses, calculate_curvature_losses, calculate_temporal_losses, calculate_parents_losses_curvatures, \
    calculate_smoothness_losses_curvatures, calculate_curvature_losses_curvatures, calculate_temporal_losses_curvatures, \
    calculate_temporal_losses_curvature_deltas, calculate_curvature_losses_curvature_deltas
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.nn.detector import ConvergenceDetector
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict, hash_data

cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'
img_extension = 'png'

PRINT_KEYS = [
    'loss/masks',
    # 'loss/parents',
    'loss/curvature',
    'loss/smoothness',
    'loss/temporal',
    # 'loss/global',
    'loss/scores'
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

        # Initialise the parameters and model
        self.parameters: MFParameters = self._init_parameters()
        self.model = self._init_model()

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
        if key in PARAMETER_NAMES + BUFFER_NAMES + TRANSIENTS_NAMES:
            return [fs.get_state(key) for fs in self.frame_batch]

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

    def _init_model(self) -> ProjectRenderScoreModel:
        """
        Build the model.
        """
        logger.info(f'Initialising model.')
        model = ProjectRenderScoreModel(
            image_size=PREPARED_IMAGE_SIZE[0],
            render_mode=self.parameters.render_mode,
            curvature_mode=self.parameters.curvature_mode,
            curvature_deltas=self.parameters.curvature_deltas,
            length_min=self.parameters.length_min,
            length_max=self.parameters.length_max,
            curvature_max=self.parameters.curvature_max,
            dX0_limit=self.parameters.dX0_limit,
            dl_limit=self.parameters.dl_limit,
            dk_limit=self.parameters.dk_limit,
            dpsi_limit=self.parameters.dpsi_limit,
        )
        model = torch.jit.script(model)
        return model

    def _init_convergence_detector(self) -> ConvergenceDetector:
        """
        Initialise the convergence detector.
        """
        logger.info(f'Initialising convergence detector.')
        p = self.parameters
        cd = ConvergenceDetector(
            shape=(1 + p.depth - p.depth_min,),
            tau_fast=p.convergence_tau_fast,
            tau_slow=p.convergence_tau_slow,
            threshold=p.convergence_threshold,
            patience=p.convergence_patience
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
        self.master_frame_state = self.trial_state.init_frame_state(
            self.trial_state.frame_nums[0],
            trainable=True,
            load=self.runtime_args.resume,
            device=self.device
        )

        # Prepare batch state
        self.frame_batch: List[FrameState] = []
        mfs = self.master_frame_state if self.parameters.use_master else None
        for i in range(self.parameters.window_size):
            if i == 0:
                frame_num = mfs.frame_num
            else:
                # Try to find a frame with sufficiently different images
                diff_frame = self.trial.find_next_frame_with_different_images(
                    self.frame_batch[-1].frame_num,
                    threshold=self.parameters.window_image_diff_threshold
                )
                frame_num = diff_frame.frame_num

            fs = self.trial_state.init_frame_state(
                frame_num,
                trainable=True,
                load=self.runtime_args.resume,
                master_frame_state=mfs,
                use_master_points=i == 0,
                device=self.device
            )

            # Zero-out the initial curvature-deltas
            if i > 0 and self.parameters.curvature_mode and self.parameters.curvature_deltas:
                with torch.no_grad():
                    for v in fs.get_state('X0'):
                        v.data.zero_()
                    for v in fs.get_state('T0'):
                        v.data.zero_()
                    for v in fs.get_state('length'):
                        v.data.zero_()
                    for v in fs.get_state('curvatures'):
                        v.data.zero_()

            self.frame_batch.append(fs)

        # Last optimised frame state
        self.last_frame_state: FrameState = None

    def _init_optimiser(self) -> Optimizer:
        """
        Set up the joint cameras and cloud optimiser and the curve optimiser.
        """
        logger.info('Initialising optimiser.')
        p = self.parameters
        points_keys = CURVATURE_PARAMETER_NAMES if p.curvature_mode else ['points', ]

        if p.use_master:
            params = {
                k: self.master_frame_state.get_state(k)
                for k in PARAMETER_NAMES
            }
            if self.parameters.window_size > 1:
                for points_key in points_keys:
                    for i in range(self.parameters.window_size):
                        params[points_key].extend(self.frame_batch[i].get_state(points_key))
        else:
            params = {
                k: [fs.get_state(k) for fs in self.frame_batch]
                for k in PARAMETER_NAMES
            }

        # Merge the curvature parameters into a single parameter group
        if p.curvature_mode:
            del params['points']
            point_params = []
            for ck in CURVATURE_PARAMETER_NAMES:
                point_params.extend(params[ck])
            params['points'] = point_params

        optimiser_cls: Optimizer = getattr(torch.optim, p.algorithm)
        cam_params = [params[f'cam_{k}'] for k in CAM_PARAMETER_NAMES]

        params = [
            {'params': cam_params, 'lr': p.lr_cam_coeffs},
            {'params': params['points'], 'lr': p.lr_points},
            {'params': params['sigmas'], 'lr': p.lr_sigmas},
            {'params': params['exponents'], 'lr': p.lr_exponents},
            {'params': params['intensities'], 'lr': p.lr_intensities},
            {'params': params['camera_sigmas'], 'lr': p.lr_sigmas},
            {'params': params['camera_exponents'], 'lr': p.lr_exponents},
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
        mfs = self.master_frame_state
        self._configure_paths()
        self._init_tb_logger()
        logger.info(f'Logs path: {self.logs_path}.')

        # Initial plots
        self._make_plots(pre_step=True)

        # Train
        # w2 = int((p.window_size - 1) / 2)
        frame_skip = 1 if p.frame_skip is None else p.frame_skip
        first_frame = self.checkpoint.frame_num
        frame_nums = list(range(first_frame, self.trial_state.frame_nums[-1]))[::frame_skip]
        n_frames = len(frame_nums)

        for i, frame_num in enumerate(frame_nums):
            logger.info(f'======== Training frame #{frame_num} ({i + 1}/{n_frames}) ========')
            # active_idx = min(i, w2)
            self.frame_num = frame_num
            self.active_idx = 0  # active_idx

            # Reset convergence detection
            self.convergence_detector.reset_counters()

            # Reset frame step counter and train the batch
            self.checkpoint.step_frame = 0
            self.checkpoint.frame_num = frame_num
            self.train(frame_num == first_frame)

            # Save the state
            self.trial_state.update_frame_state(frame_num, mfs)
            self.trial_state.save()

            # Update the reconstruction
            self.reconstruction.end_frame = max(frame_num + 1, self.reconstruction.end_frame)
            self.reconstruction.save()

            # Interpolate skipped frame parameters
            if p.frame_skip and self.last_frame_state is not None:
                logger.info('Interpolating skipped frames.')
                for k in BUFFER_NAMES + PARAMETER_NAMES:
                    var = self.trial_state.get(k, start_frame=self.last_frame_state.frame_num, end_frame=frame_num + 1)
                    v = torch.from_numpy(np.stack([var[0], var[-1]]))
                    v = v.to(self.device)
                    v = v.reshape([1, 2, -1])
                    v = v.permute(0, 2, 1)
                    v = F.interpolate(v, size=p.frame_skip + 1, mode='linear', align_corners=True)
                    v = v.squeeze().T.reshape(var.shape)
                    v = v.cpu().numpy()
                    var[1:-1] = v[1:-1]
                self.trial_state.save()

            # Make plots
            if self.runtime_args.plot_every_n_frames > -1:
                if self.last_frame_state is None \
                        and (frame_num - first_frame + 1) % self.runtime_args.plot_every_n_frames == 0:
                    self._make_plots(final_step=True)
                else:
                    for j in range(p.frame_skip):
                        plot_frame_num = self.last_frame_state.frame_num + j + 1
                        if (plot_frame_num - first_frame + 1) % self.runtime_args.plot_every_n_frames == 0:
                            if plot_frame_num == mfs.frame_num:
                                fs = mfs
                                skipped = False
                            else:
                                fs = self.trial_state.init_frame_state(frame_num=plot_frame_num)
                                skipped = True
                            self._make_plots(final_step=True, frame_state=fs, skipped=skipped)

            # Freeze just-optimised frame
            self.last_frame_state = self.trial_state.init_frame_state(
                frame_num=frame_num,
                trainable=False,
                load=False,
                prev_frame_state=mfs,
                device=self.device
            )

            # Roll window
            next_frame_num = frame_num + frame_skip  # + w2
            next_frame = self.trial_state.init_frame_state(
                frame_num=next_frame_num,
                trainable=True,
                load=False,
                prev_frame_state=mfs,
                device=self.device
            )
            if p.use_master:
                mfs.frame_num = next_frame.frame_num
                mfs.copy_state(next_frame)

                # Update first frame in batch to be the same as the master frame
                self.frame_batch[0].copy_state(mfs)
                self.frame_batch[0].frame_num = mfs.frame_num

                for j in range(1, p.window_size):
                    prev_frame = self.frame_batch[j - 1]
                    curr_frame = self.frame_batch[j]

                    # Try to find a frame with sufficiently different images
                    diff_frame = self.trial.find_next_frame_with_different_images(
                        prev_frame.frame_num,
                        threshold=self.parameters.window_image_diff_threshold
                    )
                    next_frame = self.trial_state.init_frame_state(
                        frame_num=diff_frame.frame_num,
                        trainable=True,
                        load=False,
                        prev_frame_state=curr_frame,
                        device=self.device
                    )

                    curr_frame.copy_state(next_frame)
                    curr_frame.frame_num = next_frame.frame_num

            # for j in range(p.window_size):
            #     curr_frame = self.frame_batch[j]
            #     # if j < min(i+1, w2):
            #     #     curr_frame.freeze()
            #     if i >= w2:
            #         if j + 1 < p.window_size:
            #             next_frame = self.frame_batch[j + 1]
            #         elif i + w2 < len(self.trial_state):
            #             next_frame = self.trial_state.init_frame_state(
            #                 frame_num=next_frame_num,
            #                 trainable=True,
            #                 load=False,
            #                 prev_frame_state=curr_frame,
            #                 device=self.device
            #             )
            #         else:
            #             continue
            #
            #         with torch.no_grad():
            #             for k in BUFFER_NAMES + PARAMETER_NAMES + TRANSIENTS_NAMES:
            #                 curr_frame.set_state(k, next_frame.get_state(k))
            #         curr_frame.frame_num = next_frame.frame_num

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
            self._make_plots(first_frame=first_frame)

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
        D = p.depth - p.depth_min

        # Collect parameters
        cam_coeffs = torch.stack([f.get_state('cam_coeffs') for f in self.frame_batch])
        cam_rotation_preangles = torch.stack([f.get_state('cam_rotation_preangles') for f in self.frame_batch])
        points_3d_base = torch.stack([f.get_state('points_3d_base') for f in self.frame_batch])
        points_2d_base = torch.stack([f.get_state('points_2d_base') for f in self.frame_batch])
        camera_sigmas = torch.stack([f.get_state('camera_sigmas') for f in self.frame_batch])
        camera_exponents = torch.stack([f.get_state('camera_exponents') for f in self.frame_batch])
        camera_intensities = torch.stack([f.get_state('camera_intensities') for f in self.frame_batch])
        X0 = [torch.stack([f.get_state('X0')[d] for f in self.frame_batch]) for d in range(D)]
        T0 = [torch.stack([f.get_state('T0')[d] for f in self.frame_batch]) for d in range(D)]
        length = [torch.stack([f.get_state('length')[d] for f in self.frame_batch]) for d in range(D)]
        curvatures = [torch.stack([f.get_state('curvatures')[d] for f in self.frame_batch]) for d in range(D)]
        points = [torch.stack([f.get_state('points')[d] for f in self.frame_batch]) for d in range(D)]
        masks_target = [torch.stack([f.get_state('masks_target')[d] for f in self.frame_batch]) for d in range(D)]
        sigmas = [torch.stack([f.get_state('sigmas')[d] for f in self.frame_batch]) for d in range(D)]
        exponents = [torch.stack([f.get_state('exponents')[d] for f in self.frame_batch]) for d in range(D)]
        intensities = [torch.stack([f.get_state('intensities')[d] for f in self.frame_batch]) for d in range(D)]

        # Generate the outputs
        masks, detection_masks, points_2d, scores, curvatures_smoothed, points_smoothed, sigmas_smoothed, exponents_smoothed, intensities_smoothed = self.model.forward(
            cam_coeffs=cam_coeffs,
            points_3d_base=points_3d_base,
            points_2d_base=points_2d_base,
            X0=X0,
            T0=T0,
            length=length,
            curvatures=curvatures,
            points=points,
            masks_target=masks_target,
            sigmas=sigmas,
            exponents=exponents,
            intensities=intensities,
            camera_sigmas=camera_sigmas,
            camera_exponents=camera_exponents,
            camera_intensities=camera_intensities,
            length_warmup=self.checkpoint.step < p.length_warmup_steps,
        )

        # Generate targets with added residuals
        masks_target_residuals = generate_residual_targets(masks_target, masks, detection_masks)

        # Calculate the losses
        loss, loss_global, losses_depths, stats = self._calculate_losses(
            cam_rotation_preangles=cam_rotation_preangles,
            X0=X0,
            T0=T0,
            length=length,
            curvatures=curvatures,
            points=points,
            masks_target=masks_target_residuals,
            sigmas=sigmas,
            exponents=exponents,
            intensities=intensities,
            camera_sigmas=camera_sigmas,
            camera_exponents=camera_exponents,
            camera_intensities=camera_intensities,
            masks=masks,
            scores=scores,
            curvatures_smoothed=curvatures_smoothed,
            points_smoothed=points_smoothed,
        )

        # Take optimisation step
        if is_bad(loss):
            logger.warning('Bad loss, skipping parameter update.')
        else:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self._clamp_parameters(points_smoothed)

        # Update master state
        self.master_frame_state.set_state('masks_curve', [masks[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('masks_target_residuals',
                                          [masks_target_residuals[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('points_2d', [points_2d[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('scores', [scores[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('sigmas_smoothed', [sigmas_smoothed[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('exponents_smoothed',
                                          [exponents_smoothed[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_state('intensities_smoothed',
                                          [intensities_smoothed[d][self.active_idx] for d in range(D)])
        self.master_frame_state.set_stats(stats)
        if p.curvature_mode:
            self.master_frame_state.set_state('points', [points_smoothed[d][self.active_idx] for d in range(D)])
            self.master_frame_state.set_state('curvatures_smoothed',
                                              [curvatures_smoothed[d][self.active_idx] for d in range(D)])
        else:
            self.master_frame_state.set_state('curvatures', [curvatures_smoothed[d][self.active_idx] for d in range(D)])

        # Update batch state
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('masks_curve', [masks[d][i] for d in range(D)])
            fs.set_state('masks_target_residuals', [masks_target_residuals[d][i] for d in range(D)])
            fs.set_state('points_2d', [points_2d[d][i] for d in range(D)])
            fs.set_state('scores', [scores[d][i] for d in range(D)])
            fs.set_state('sigmas_smoothed', [sigmas_smoothed[d][i] for d in range(D)])
            fs.set_state('exponents_smoothed', [exponents_smoothed[d][i] for d in range(D)])
            fs.set_state('intensities_smoothed', [intensities_smoothed[d][i] for d in range(D)])
            fs.set_stats(stats)
            if p.curvature_mode:
                fs.set_state('points', [points_smoothed[d][i] for d in range(D)])
                fs.set_state('curvatures_smoothed', [curvatures_smoothed[d][i] for d in range(D)])
            else:
                fs.set_state('curvatures', [curvatures_smoothed[d][i] for d in range(D)])

        return loss, loss_global, losses_depths, stats

    def _calculate_losses(
            self,
            cam_rotation_preangles: torch.Tensor,
            X0: List[torch.Tensor],
            T0: List[torch.Tensor],
            length: List[torch.Tensor],
            curvatures: List[torch.Tensor],
            points: List[torch.Tensor],
            masks_target: List[torch.Tensor],
            sigmas: List[torch.Tensor],
            exponents: List[torch.Tensor],
            intensities: List[torch.Tensor],
            camera_sigmas: torch.Tensor,
            camera_exponents: torch.Tensor,
            camera_intensities: torch.Tensor,
            masks: List[torch.Tensor],
            scores: List[torch.Tensor],
            curvatures_smoothed: List[torch.Tensor],
            points_smoothed: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Dict[str, float]]:
        """
        Calculate the losses.
        """
        p = self.parameters
        stats = {}
        loss_global = 0.
        losses_depths = []

        def _log_parameter_stats(d_, key_, var_):
            if var_.nelement() == 1:
                stats[f'{key_}/{d_}'] = var_.item()
            else:
                stats[f'{key_}/{d_}/mean'] = var_.mean()
                if d_ > 1:
                    stats[f'{key_}/{d_}/var'] = var_.var()

        # Previous points used for temporal losses
        if self.last_frame_state is not None:
            X0_prev = self.last_frame_state.get_state('X0')
            T0_prev = self.last_frame_state.get_state('T0')
            length_prev = self.last_frame_state.get_state('length')
            curvatures_prev = self.last_frame_state.get_state('curvatures')
            points_prev = self.last_frame_state.get_state('points')
        else:
            X0_prev = None
            T0_prev = None
            length_prev = None
            curvatures_prev = None
            points_prev = None

        # Losses calculated at each depth
        losses = {
            'masks': calculate_renders_losses(masks, masks_target, p.loss_masks_metric, p.loss_masks_multiscale),
            'scores': calculate_scores_losses(scores),
        }

        if p.curvature_mode:
            losses = {**losses, **{
                'parents': calculate_parents_losses_curvatures(X0, T0, length, curvatures, curvatures_smoothed),
                'smoothness': calculate_smoothness_losses_curvatures(curvatures, curvatures_smoothed),
            }}
            if p.curvature_deltas:
                losses['temporal'] = calculate_temporal_losses_curvature_deltas(
                    X0, T0, length, curvatures,
                    X0_prev, T0_prev, length_prev, curvatures_prev
                )
                losses['curvature'] = calculate_curvature_losses_curvature_deltas(curvatures)
            else:
                losses['temporal'] = calculate_temporal_losses_curvatures(
                    X0, T0, length, curvatures,
                    X0_prev, T0_prev, length_prev, curvatures_prev
                )
                losses['curvature'] = calculate_curvature_losses_curvatures(curvatures)
        else:
            losses = {**losses, **{
                'neighbours': calculate_neighbours_losses(points),
                'parents': calculate_parents_losses(points),
                'aunts': calculate_aunts_losses(points),
                'smoothness': calculate_smoothness_losses(points, points_smoothed),
                'curvature': calculate_curvature_losses(points_smoothed, curvatures_smoothed),
                'temporal': calculate_temporal_losses(points, points_prev),
            }}

        # Log the total loss for each type
        for k, losses_k in losses.items():
            loss_k = sum(losses_k)
            stats[f'loss/{k}'] = loss_k.item()

        # Sum the losses at each depth
        for i, d in enumerate(range(p.depth_min, p.depth)):
            loss_d = 0.
            for k, losses_k in losses.items():
                w = getattr(p, f'loss_{k}')
                if w > 0:
                    loss_d += w * losses_k[i]
            stats[f'loss/depth/{d}'] = loss_d.item()
            losses_depths.append(loss_d)

            # Log actual losses at each depth
            if self.runtime_args.log_level > 0:
                for k in losses.keys():
                    stats[f'loss_d/{k}/{d}'] = losses[k][i].item()

            # Log additional stats
            if self.runtime_args.log_level > 1:
                if d > 0:
                    # Track distance to neighbours
                    dist_neighbours = torch.norm(points[i][:, 1:] - points[i][:, -1], dim=-1)
                    _log_parameter_stats(d, 'dists/neighbours', dist_neighbours)

                    # Track distance to parent
                    if i > 0:
                        curve_points_parent = torch.repeat_interleave(points[i - 1], repeats=2, dim=1)
                        dist_parent = torch.norm(points[i] - curve_points_parent, dim=-1)
                        _log_parameter_stats(d, 'dists/parent', dist_parent)

                # Lengths, scores, sigmas, exponents and intensities
                _log_parameter_stats(d, 'lengths', length[i])
                _log_parameter_stats(d, 'scores', scores[i])
                _log_parameter_stats(d, 'sigmas', sigmas[i])
                _log_parameter_stats(d, 'exponents', exponents[i])
                _log_parameter_stats(d, 'intensities', intensities[i])

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
                stats[f'camera_sigmas/{i}'] = camera_sigmas[:, i].var()
                stats[f'camera_exponents/{i}'] = camera_exponents[:, i].var()
                stats[f'camera_intensities/{i}'] = camera_intensities[:, i].var()

        # Sum the global losses with the losses generated at each depth
        loss = sum(losses_depths) + loss_global
        stats['loss/total'] = loss.item()

        return loss, loss_global, losses_depths, stats

    def _clamp_parameters(self, points_smoothed: torch.Tensor):
        """
        Ensure all the parameters fall within the constraints.
        """
        p = self.parameters
        D = p.depth - p.depth_min

        with torch.no_grad():
            if p.curvature_mode:
                for i, fs in enumerate(self.frame_batch):
                    if p.curvature_deltas and i != 0:
                        # Clamping only needed for the master frame in deltas-mode.
                        break
                    length = fs.get_state('length')
                    curvatures = fs.get_state('curvatures')
                    for d in range(D):
                        length_d = length[d]
                        curvatures_d = curvatures[d]

                        # Ensure that the worm does not get too long/short.
                        if self.checkpoint.step > p.length_warmup_steps:
                            l = length_d.clamp(
                                min=p.length_min,
                                max=p.length_max
                            )
                        else:
                            # In length warmup phase, linearly grow the length from length_init to length_min
                            l = torch.tensor(
                                p.length_init
                                + (p.length_min - p.length_init) * self.checkpoint.step / p.length_warmup_steps,
                                device=self.device
                            )
                        length[d].data = l

                        # Ensure curvature doesn't get too large.
                        K = curvatures_d
                        k = torch.norm(K, dim=-1)
                        k_max = p.curvature_max * 2 * torch.pi / (K.shape[0] + 2 - 1)
                        K = torch.where(
                            (k > k_max)[:, None],
                            K * (k_max / (k + 1e-6))[:, None],
                            K
                        )
                        curvatures[d].data = K
            else:
                # Adjust points to be a quarter of the mean segment-length between parent points.
                for i, fs in enumerate(self.frame_batch):
                    points = fs.get_state('points')
                    for d in range(2, D):
                        points_d = points[d]
                        parents = points_smoothed[d - 1][i]
                        x = torch.mean(torch.norm(parents[1:] - parents[:-1], dim=-1)) / 4
                        parents_repeated = torch.repeat_interleave(parents, repeats=2, dim=0)
                        direction = points_d - parents_repeated
                        points_anchored = parents_repeated \
                                          + x * (direction / torch.norm(direction, dim=-1, keepdim=True))
                        points[d].data = torch.cat([
                            points_d[0][None, :],
                            points_anchored[1:-1],
                            points_d[-1][None, :]
                        ], dim=0)

            # Clamp the sigmas, exponents and intensities
            render_parameters_limits = {
                'sigmas': {
                    'min': 3e-2
                },
                'exponents': {
                    'min': 0.5,
                    'max': 10
                },
                'intensities': {
                    'min': 0.1,
                    'max': 10
                }
            }

            for sei in ['sigmas', 'exponents', 'intensities']:
                params = [*self.master_frame_state.get_state(sei), ]
                for v in params:
                    v.data = v.clamp(**render_parameters_limits[sei])

                # Camera scaling factors should average 1
                v = self.master_frame_state.get_state(f'camera_{sei}')
                v.data = v / v.mean()

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
                for d in range(self.parameters.depth_min, self.parameters.depth + 1):
                    d_str = 'global' if d == self.parameters.depth_min else d - 1
                    for k in state_vars:
                        self.tb_logger.add_scalar(f'detector/{d_str}/{k}', getattr(detector, k)[d].item(), step)

    def _make_plots(
            self,
            pre_step: bool = False,
            final_step: bool = False,
            frame_state: FrameState = None,
            skipped: bool = False,
            first_frame: bool = False
    ):
        """
        Generate some plots.
        """
        if frame_state is None:
            frame_state = self.master_frame_state

        # Make initial plots for all batch elements
        if pre_step:
            self._plot_3d(frame_state, skipped)
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and self.step % self.runtime_args.plot_every_n_steps == 0
        ) or (
                first_frame
                and self.runtime_args.plot_every_n_init_steps > -1
                and self.step % self.runtime_args.plot_every_n_init_steps == 0
        ):
            logger.info(f'Plotting frame #{frame_state.frame_num}.')
            self._plot_3d(frame_state, skipped)
            self._plot_2d(frame_state, skipped)
            if self.parameters.window_size > 1:
                self._plot_2d_batch()
            if self.parameters.curvature_mode:
                self._plot_curvatures()
            self._plot_point_stats(frame_state, skipped)

    def _plot_3d(self, frame_state: FrameState, skipped: bool = False):
        """
        Make a multiscale curve 3D scatter plot.
        """
        cmap_vertices = 'autumn_r'
        cmap_curve = 'jet'
        D = self.parameters.depth - self.parameters.depth_min
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        # Rotate the perspective on every plot
        self.plot_3d_azim += 1

        fig = plt.figure(figsize=(n_cols * 4, 1 + n_rows * 4))
        fig.suptitle(self._plot_title(frame_state, skipped))
        gs = GridSpec(n_rows, n_cols)
        d = 0

        points = frame_state.get_state('points')
        sigmas = frame_state.get_state('sigmas_smoothed')
        scores = frame_state.get_state('scores')
        length = frame_state.get_state('length')

        for i in range(n_rows):
            for j in range(n_cols):
                if d >= D:
                    break
                ax = fig.add_subplot(gs[i, j], projection='3d')
                ax.view_init(azim=self.plot_3d_azim)
                title = f'd={d + self.parameters.depth_min}'
                if self.parameters.curvature_mode:
                    title += f', l={length[d]:.2f}'
                ax.set_title(title)
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
        self._save_plot(fig, '3D', frame_state)

    def _plot_2d(self, frame_state: FrameState, skipped: bool = False):
        """
        Plot the 2D mask renderings of the mutiscale curves.
        """
        D = self.parameters.depth
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        fig = plt.figure(figsize=(n_cols * 3, 1 + n_rows * 3))
        fig.suptitle(self._plot_title(frame_state, skipped))
        gs = GridSpec(n_rows, n_cols)
        d = self.parameters.depth_min
        di = 0

        masks_target = frame_state.get_state('masks_target_residuals')
        points_2d = frame_state.get_state('points_2d')
        masks_curve = frame_state.get_state('masks_curve')

        images = to_numpy(frame_state.get_state('images'))
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

                X_target = to_numpy(masks_target[di])
                X_curve = to_numpy(masks_curve[di])

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
                p2d = to_numpy(points_2d[di]).transpose(1, 0, 2)
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
                di += 1

        fig.tight_layout()
        self._save_plot(fig, '2D', frame_state)

    def _plot_2d_batch(self):
        """
        Plot a batch of 2D mask renderings of the mutiscale curves.
        """
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        n_rows = D - D_min
        n_cols = self.parameters.window_size

        fig = plt.figure(figsize=(n_cols * 2, 1 + n_rows * 2))
        fig.suptitle(self._plot_title(self.master_frame_state))
        gs = GridSpec(
            n_rows,
            n_cols,
            wspace=0.01,
            hspace=0.02,
            width_ratios=[1] * n_cols,
            top=0.92,
            bottom=0.05,
            left=0.05,
            right=0.95
        )

        for i, frame_state in enumerate(self.frame_batch):
            masks_target = frame_state.get_state('masks_target_residuals')
            points_2d = frame_state.get_state('points_2d')
            masks_curve = frame_state.get_state('masks_curve')

            images = to_numpy(frame_state.get_state('images'))
            image_triplet = np.concatenate(images, axis=1)
            image_grid = np.concatenate([np.ones_like(image_triplet), image_triplet, np.ones_like(image_triplet)],
                                        axis=0)
            M, N = tuple(image_grid.shape)

            scatter_sizes = (np.linspace(1, 0, D)**2 * 100) + 0.3

            for j, d in enumerate(range(D_min, D)):
                ax = fig.add_subplot(gs[j, i])
                if j == 0:
                    ax.set_title(f'frame_num={frame_state.frame_num}')
                if i == 0:
                    ax.text(-0.1, 0.45, f'd={d}', transform=ax.transAxes, rotation='vertical')
                ax.axis('off')

                X_target = to_numpy(masks_target[j])
                X_curve = to_numpy(masks_curve[j])

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
                p2d = to_numpy(points_2d[j]).transpose(1, 0, 2)
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

        self._save_plot(fig, '2D_batch', self.master_frame_state)

    def _plot_point_stats(self, frame_state: FrameState, skipped: bool = False):
        """
        Plot the point scores, sigmas, exponents and intensities for the curves.
        """
        ra = self.runtime_args
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        cmap = plt.get_cmap('jet')
        scores = frame_state.get_state('scores')
        sigmas = frame_state.get_state('sigmas_smoothed')
        exponents = frame_state.get_state('exponents_smoothed')
        intensities = frame_state.get_state('intensities_smoothed')

        n_rows = int(ra.plot_scores) + int(ra.plot_sigmas) + int(ra.plot_exponents) + int(ra.plot_intensities)
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
            camera_sigmas = frame_state.get_state('camera_sigmas')
            sfs = ', '.join([f'{sf:.3f}' for sf in camera_sigmas])
            ax_sigmas.set_title(f'sigmas\nCameras: {sfs}.')
            i += 1
        if ra.plot_exponents:
            ax_exponents = axes[i]
            camera_exponents = frame_state.get_state('camera_exponents')
            sfs = ', '.join([f'{sf:.3f}' for sf in camera_exponents])
            ax_exponents.set_title(f'exponents\nCameras: {sfs}.')
            i += 1
        if ra.plot_intensities:
            ax_intensities = axes[i]
            camera_intensities = frame_state.get_state('camera_intensities')
            sfs = ', '.join([f'{sf:.3f}' for sf in camera_intensities])
            ax_intensities.set_title(f'Intensities\nCameras: {sfs}.')

        fig.suptitle(self._plot_title(frame_state, skipped))

        colours = [cmap(d) for d in np.linspace(0, 1, D)]
        positions = [
            np.linspace(0, 1, 2**d + 2)[1:-1]
            for d in range(D)
        ]

        for i, d in enumerate(range(D_min, D)):
            plot_args = {'label': f'd={d}', 'color': colours[d], 'alpha': 0.5}
            scatter_args = {'color': colours[d], 'alpha': 0.8, 's': 10}

            # Scores
            if ra.plot_scores:
                scores_d = to_numpy(scores[i])
                ax_scores.plot(positions[d], scores_d, **plot_args)
                ax_scores.scatter(x=positions[d], y=scores_d, **scatter_args)

            # Sigmas
            if ra.plot_sigmas:
                sigmas_d = to_numpy(sigmas[i])
                ax_sigmas.plot(positions[d], sigmas_d, **plot_args)
                ax_sigmas.scatter(x=positions[d], y=sigmas_d, **scatter_args)

            # Exponents
            if ra.plot_exponents:
                exponents_d = to_numpy(exponents[i])
                ax_exponents.plot(positions[d], exponents_d, **plot_args)
                ax_exponents.scatter(x=positions[d], y=exponents_d, **scatter_args)

            # Intensities
            if ra.plot_intensities:
                intensities_d = to_numpy(intensities[i])
                ax_intensities.plot(positions[d], intensities_d, **plot_args)
                ax_intensities.scatter(x=positions[d], y=intensities_d, **scatter_args)

        fig.tight_layout()
        self._save_plot(fig, 'point_stats', frame_state)

    def _plot_curvatures(self):
        """
        Plot the curvatures.
        """
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        cmap = plt.get_cmap('jet')

        fig, axes = plt.subplots(6, D - D_min, figsize=((D - D_min) * 6 + 2, 10), squeeze=False)
        fig.suptitle(self._plot_title(self.master_frame_state))

        colours = [cmap(i) for i in np.linspace(0, 1, len(self.frame_batch))]
        positions = [
            np.linspace(0, 1, 2**d + 2)[1:-1]
            for d in range(D)
        ]
        k_axes = {d: axes[0, i] for i, d in enumerate(range(D_min, D))}
        psi_axes = {d: axes[1, i] for i, d in enumerate(range(D_min, D))}
        m1_axes = {d: axes[2, i] for i, d in enumerate(range(D_min, D))}
        m2_axes = {d: axes[3, i] for i, d in enumerate(range(D_min, D))}
        dk_axes = {d: axes[4, i] for i, d in enumerate(range(D_min, D))}
        dpsi_axes = {d: axes[5, i] for i, d in enumerate(range(D_min, D))}

        for i, frame_state in enumerate(self.frame_batch):
            curvatures = frame_state.get_state('curvatures_smoothed')
            plot_args = {
                'label': f'frame={frame_state.frame_num}',
                'color': colours[i],
                'alpha': 0.5
            }
            scatter_args = {'color': colours[i], 'alpha': 0.8, 's': 10}

            for j, d in enumerate(range(D_min, D)):
                K = to_numpy(curvatures[j]) * (2**d - 1)

                # Curvature magnitude
                k = np.linalg.norm(K, axis=-1)
                k_ax = k_axes[d]
                if i == 0:
                    k_ax.set_title(f'd={d}')
                    k_ax.set_ylabel('$|\kappa|=|m_1+m_2|$')
                k_ax.plot(positions[d], k, **plot_args)
                k_ax.scatter(x=positions[d], y=k, **scatter_args)
                k_ax.legend()

                # Curvature angles
                psi = np.arctan2(K[..., 0], K[..., 1])
                psi_ax = psi_axes[d]
                if i == 0:
                    psi_ax.set_title(f'd={d}')
                    psi_ax.set_ylabel('$\psi=\measuredangle\kappa$')
                psi_ax.plot(positions[d], psi, **plot_args)
                psi_ax.scatter(x=positions[d], y=psi, **scatter_args)
                psi_ax.legend()

                # m1
                m1 = K[:, 0]
                m1_ax = m1_axes[d]
                if i == 0:
                    m1_ax.set_ylabel('$m_1$')
                m1_ax.plot(positions[d], m1, **plot_args)
                m1_ax.scatter(x=positions[d], y=m1, **scatter_args)

                # m2
                m2 = K[:, 1]
                m2_ax = m2_axes[d]
                if i == 0:
                    m2_ax.set_ylabel('$m_1$')
                m2_ax.plot(positions[d], m2, **plot_args)
                m2_ax.scatter(x=positions[d], y=m2, **scatter_args)

                # deltas
                if i > 0:
                    curvatures_prev = self.frame_batch[i - 1].get_state('curvatures_smoothed')
                    Kp = to_numpy(curvatures_prev[j]) * (2**d - 1)

                    # delta curvature magnitudes
                    k_prev = np.linalg.norm(Kp, axis=-1)
                    dk = np.abs(k - k_prev)
                    dk_ax = dk_axes[d]
                    if i == 1:
                        dk_ax.set_ylabel('$\delta\kappa$')
                    dk_ax.plot(positions[d], dk, **plot_args)
                    dk_ax.scatter(x=positions[d], y=dk, **scatter_args)

                    # delta curvature angles
                    psi_prev = np.arctan2(Kp[..., 0], Kp[..., 1])
                    dpsi = np.abs(psi - psi_prev)
                    dpsi_ax = dpsi_axes[d]
                    if i == 1:
                        dpsi_ax.set_ylabel('$\delta\psi$')
                    dpsi_ax.plot(positions[d], dpsi, **plot_args)
                    dpsi_ax.scatter(x=positions[d], y=dpsi, **scatter_args)

        fig.tight_layout()
        self._save_plot(fig, 'curvatures', frame_state)

    def _plot_title(self, frame_state: FrameState, skipped: bool = False) -> str:
        title = f'Trial: {self.trial.id}. {self.trial.date:%Y%m%d}. ' \
                f'Frame: {frame_state.frame_num}/{self.trial_state.frame_nums[-1]}.\n' \
                f'Global step: {self.checkpoint.step}. '
        if not skipped:
            title += f'Frame step: {self.checkpoint.step_frame}.'
        else:
            title += f'(Interpolated).'

        return title

    def _save_plot(self, fig: Figure, plot_type: str, frame_state: FrameState):
        """
        Either log the figure to the tensorboard logger or save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path / 'plots' / plot_type
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir / f'{frame_state.frame_num:05d}_{self.step:06d}.{img_extension}'
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(plot_type, fig, self.step)
            self.tb_logger.flush()

        plt.close(fig)
