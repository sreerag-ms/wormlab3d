import os
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mongoengine import DoesNotExist
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.backends import cudnn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from simple_worm.plot3d import MidpointNormalize
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Trial, MFCheckpoint, MFParameters, Reconstruction
from wormlab3d.data.model.mf_parameters import CURVATURE_INTEGRATION_MIDPOINT, CURVATURE_INTEGRATION_HT
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.args_finder import ParameterArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, CAM_PARAMETER_NAMES, \
    TRANSIENTS_NAMES, CURVATURE_PARAMETER_NAMES
from wormlab3d.midlines3d.mf_methods import generate_residual_targets, calculate_renders_losses, \
    calculate_scores_losses, calculate_smoothness_losses, calculate_neighbours_losses, calculate_parents_losses, \
    calculate_aunts_losses, calculate_curvature_losses, calculate_temporal_losses, calculate_parents_losses_curvatures, \
    calculate_smoothness_losses_curvatures, calculate_curvature_losses_curvatures, calculate_temporal_losses_curvatures, \
    calculate_temporal_losses_curvature_deltas, calculate_curvature_losses_curvature_deltas, \
    calculate_intersection_losses_curvatures, calculate_alignment_losses_curvatures, \
    integrate_curvature, normalise, orthogonalise, calculate_consistency_losses_curvatures_ht
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.nn.LBFGS import FullBatchLBFGS
from wormlab3d.nn.args.optimiser_args import OPTIMISER_LBFGS_NEW
from wormlab3d.nn.detector import ConvergenceDetector
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict

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
    'loss/scores',
    'loss/intersections',
    'loss/alignment',
    'loss/consistency',
    'shifts',
    'lr'
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

        # Set random seed
        self._set_seed()

        # Initialise the parameters
        self.parameters: MFParameters = self._init_parameters()

        # Initialise convergence detector
        self.convergence_detector = self._init_convergence_detector()

        # Check the devices
        self.device = self._init_devices()

        # Reconstruction
        self.reconstruction = self._init_reconstruction()

        # Load the trial and initialise trainable parameters
        self._init_trial()

        # Initialise the model
        self.model = self._init_model()

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
        return LOGS_PATH / f'trial_{checkpoint.trial.id:03d}' / f'{checkpoint.parameters.created:%Y%m%d_%H:%M}_{checkpoint.parameters.id}'

    @property
    def step(self):
        return self.checkpoint.step

    def __getattr__(self, key):
        """
        Allow batched parameters to be accessed as member variables.
        """
        if key in PARAMETER_NAMES + BUFFER_NAMES + TRANSIENTS_NAMES:
            return [fs.get_state(key) for fs in self.frame_batch]

    def _set_seed(self):
        """
        Set the random seed everywhere.
        """
        seed = self.runtime_args.seed
        logger.info(f'Setting random seed = {seed}.')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Can uncomment these for full deterministic output:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def _init_parameters(self) -> MFParameters:
        """
        Build the model.
        """
        logger.info(f'Initialising parameters.')
        parameters = None
        pa = self.parameter_args
        params = pa.get_db_params()
        if self.runtime_args.finetune_mode:
            assert not pa.use_master, 'Cannot use_master in finetune mode!'
            assert pa.length_warmup_steps is None, 'Cannot use length_warmup_steps in finetune mode!'
            assert pa.length_regrow_steps is None, 'Cannot use length_regrow_steps in finetune mode!'
            assert pa.curvature_relaxation_factor is None, 'Cannot use curvature_relaxation_factor in finetune mode!'
            assert pa.frame_skip is None or pa.frame_skip < pa.window_size, 'frame_skip must be less than window_size in finetune mode!'

        # Try to load an existing model
        if pa.load:
            # If we have a model id then load this from the database
            if pa.params_id is not None:
                parameters = MFParameters.objects.get(id=pa.params_id)
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
            # Fetch a reconstruction by id, check that the trial and source match and update the parameters if required.
            if self.runtime_args.resume_from != 'latest':
                reconstruction = Reconstruction.objects.get(id=self.runtime_args.resume_from)
                if reconstruction.trial.id != params['trial']:
                    raise RuntimeError('Cannot resume from a reconstruction for a different trial!')
                if reconstruction.source != params['source']:
                    raise RuntimeError('Cannot resume from a different midline source!')
                if reconstruction.mf_parameters.id != params['mf_parameters'].id:
                    logger.warning('Parameters have changed! This may cause problems!')
                    reconstruction.mf_parameters = self.parameters
            else:
                reconstruction = Reconstruction.objects.get(**params)
            if self.runtime_args.copy_state is not None:
                raise RuntimeError('Can only copy state to a new reconstruction!')
            if reconstruction.start_frame > start_frame:
                reconstruction.start_frame = start_frame
            reconstruction.save()
            logger.info(f'Loaded reconstruction (id={reconstruction.id}, created={reconstruction.created}).')
        except DoesNotExist:
            err = 'No reconstruction record found in database.'
            if self.runtime_args.resume and \
                    (self.runtime_args.copy_state is None or self.runtime_args.resume_from != 'latest'):
                raise RuntimeError(err)
            logger.info(err)

        if reconstruction is None:
            if self.runtime_args.copy_state is not None:
                copied_from = Reconstruction.objects.get(id=self.runtime_args.copy_state)
                params['start_frame'] = copied_from.start_frame
                params['end_frame'] = copied_from.end_frame
                params['copied_from'] = copied_from
            else:
                params['start_frame'] = start_frame
                params['end_frame'] = start_frame
                params['copied_from'] = None
            reconstruction = Reconstruction(**params)
            reconstruction.save()
            logger.info(f'Saved reconstruction record to database (id={reconstruction.id})')

        return reconstruction

    def _init_trial(self):
        """
        Load the trial.
        """
        logger.info('Initialising trial state.')
        sa = self.source_args
        ra = self.runtime_args
        self.trial: Trial = self.reconstruction.trial

        # Copy state across from previous reconstruction
        if self.runtime_args.copy_state is not None:
            copy_state = TrialState(reconstruction=self.reconstruction.copied_from)
        else:
            copy_state = None

        if sa.end_frame == -1:
            start_frame = sa.start_frame
            end_frame = sa.end_frame
        else:
            start_frame = min(sa.start_frame, sa.end_frame)
            end_frame = max(sa.start_frame, sa.end_frame)

        # Prepare trial state
        self.trial_state = TrialState(
            reconstruction=self.reconstruction,
            start_frame=start_frame,
            end_frame=end_frame,
            read_only=False,
            load_only=ra.resume if copy_state is None else False,
            copy_state=copy_state
        )

        # Get frame numbers in the order they will be tackled
        frame_nums = self.trial_state.frame_nums
        if sa.direction == -1:
            frame_nums = frame_nums[::-1]

        # In fix mode load the start and end frames as targets
        if ra.fix_mode:
            assert sa.end_frame != -1, 'End frame must be defined in fix-mode.'
            self.fix_target_start = self.trial_state.init_frame_state(
                sa.start_frame,
                trainable=False,
                load=True,
                device=self.device
            )
            self.fix_target_end = self.trial_state.init_frame_state(
                sa.end_frame,
                trainable=False,
                load=True,
                device=self.device
            )

            # Start optimisation from 1-in from the start frame
            start_frame = frame_nums[1]
        else:
            start_frame = frame_nums[0]

        # Master state
        self.master_frame_state = self.trial_state.init_frame_state(
            start_frame,
            trainable=True,
            load=ra.resume,
            device=self.device
        )

        # Prepare batch state
        self.frame_batch: List[FrameState] = []
        mfs = self.master_frame_state if self.parameters.use_master else None
        for i in range(self.parameters.window_size):
            if i == 0:
                frame_num = start_frame
            elif ra.finetune_mode or self.parameters.window_image_diff_threshold <= 0:
                # Use contiguous batches in finetune mode
                frame_num += 1
            else:
                # Try to find a frame with sufficiently different images
                diff_frame = self.trial.find_next_frame_with_different_images(
                    self.frame_batch[-1].frame_num,
                    threshold=self.parameters.window_image_diff_threshold,
                    direction=sa.direction
                )
                frame_num = diff_frame.frame_num

            fs = self.trial_state.init_frame_state(
                frame_num,
                trainable=True,
                load=ra.resume,
                master_frame_state=mfs,
                use_master_points=i == 0,
                device=self.device
            )

            # Set the curves to the same
            if i > 0 and not ra.resume:
                for k in CURVATURE_PARAMETER_NAMES:
                    fs.set_state(k, self.frame_batch[0].get_state(k))

            # Zero-out the initial curvature-deltas
            if i > 0 and self.parameters.curvature_mode and self.parameters.curvature_deltas:
                with torch.no_grad():
                    for k in CURVATURE_PARAMETER_NAMES:
                        for v in fs.get_state(k):
                            v.data.zero_()

            self.frame_batch.append(fs)

        # Last optimised frame state
        self.last_frame_state: FrameState = None

        # Shrunken lengths
        self.shrunken_lengths = torch.stack(self.master_frame_state.get_state('length')).detach()

    def _init_model(self) -> ProjectRenderScoreModel:
        """
        Build the model.
        """
        logger.info(f'Initialising model.')
        p = self.parameters
        model = ProjectRenderScoreModel(
            image_size=self.trial.crop_size,
            render_mode=p.render_mode,
            second_render_prob=p.second_render_prob,
            filter_size=p.filter_size,
            sigmas_min=p.sigmas_min,
            sigmas_max=p.sigmas_max,
            intensities_min=p.intensities_min,
            curvature_mode=p.curvature_mode,
            curvature_deltas=p.curvature_deltas,
            curvature_smoothing=p.curvature_smoothing,
            curvature_integration=p.curvature_integration,
            curvature_integration_algorithm=p.curvature_integration_algorithm,
            length_min=p.length_min,
            length_max=p.length_max,
            curvature_max=p.curvature_max,
            dX0_limit=p.dX0_limit,
            dl_limit=p.dl_limit,
            dk_limit=p.dk_limit,
            dpsi_limit=p.dpsi_limit,
            clamp_X0=p.clamp_X0
        )
        model = torch.jit.script(model)
        model = model.to(self.device)
        return model

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
        cam_params = [params[f'cam_{k}'] for k in CAM_PARAMETER_NAMES]

        # Merge the curvature parameters into a single parameter group
        lbfgs_params = []
        lbfgs_keys = ['X0', 'T0', 'M10', 'X0ht', 'T0ht', 'M10ht', 'curvatures']
        if p.curvature_mode:
            del params['points']
            point_params = []
            for ck in CURVATURE_PARAMETER_NAMES:
                if ck in lbfgs_keys and p.algorithm == OPTIMISER_LBFGS_NEW:
                    def add_params(p_):
                        if type(p_) == list:
                            for p2 in p_:
                                add_params(p2)
                        elif type(p_) == Parameter:
                            lbfgs_params.append(p_)

                    add_params(params[ck])
                elif ck == 'curvatures':
                    continue
                else:
                    point_params.extend(params[ck])
            params['points'] = point_params

        opt_params = [
            {'params': cam_params, 'lr': p.lr_cam_coeffs},
            {'params': params['points'], 'lr': p.lr_points},
            {'params': params['sigmas'], 'lr': p.lr_sigmas},
            {'params': params['exponents'], 'lr': p.lr_exponents},
            {'params': params['intensities'], 'lr': p.lr_intensities},
            {'params': params['camera_sigmas'], 'lr': p.lr_sigmas},
            {'params': params['camera_exponents'], 'lr': p.lr_exponents},
            {'params': params['camera_intensities'], 'lr': p.lr_intensities},
            {'params': params['filters'], 'lr': p.lr_filters},
        ]

        if p.curvature_mode and p.algorithm != OPTIMISER_LBFGS_NEW:
            opt_params.append({
                'params': params['curvatures'], 'lr': p.lr_curvatures
            })

        # Merge the other parameters into flat lists
        for i, pg in enumerate(opt_params):
            if type(pg['params'][0]) == list:
                opt_params[i]['params'] = [pp for ppl in opt_params[i]['params'] for pp in ppl]

        if p.algorithm == OPTIMISER_LBFGS_NEW:
            assert p.curvature_mode, 'Can only use LBFGS algorithm in curvature mode.'

            # Use the LBFGS optimiser for the curvatures only
            optimiser = FullBatchLBFGS(
                lbfgs_params,
                lr=0.1,
                history_size=100,
                line_search='Wolfe',
                debug=False,
            )

            # Use AdamW to optimise all of the other parameters
            self.optimiser_gd = torch.optim.AdamW(
                params=opt_params,
                weight_decay=0
            )
        else:
            optimiser_cls: Optimizer = getattr(torch.optim, p.algorithm)
            optimiser = optimiser_cls(
                params=opt_params,
                weight_decay=0
            )

        return optimiser

    def _init_lr_scheduler(self) -> Optional[ReduceLROnPlateau]:
        """
        Set up the learning rate scheduler.
        """
        logger.info('Initialising lr scheduler.')
        p = self.parameters
        if p.lr_decay is None or p.lr_decay <= 0:
            return None
        if p.algorithm == OPTIMISER_LBFGS_NEW:
            optim = self.optimiser_gd
        else:
            optim = self.optimiser
        scheduler = ReduceLROnPlateau(
            optim,
            mode='min',
            factor=p.lr_decay,
            patience=p.lr_patience,
            min_lr=p.lr_min
        )

        return scheduler

    def _init_checkpoint(self) -> MFCheckpoint:
        """
        The checkpoint contains all the arguments and keeps track of progress.
        """

        # Load previous checkpoint
        prev_checkpoints = MFCheckpoint.objects(
            trial=self.trial,
            reconstruction=self.reconstruction,
        ).order_by('-created')

        if prev_checkpoints.count() > 0:
            prev_checkpoint = prev_checkpoints[0]
            logger.info(f'Found {prev_checkpoints.count()} previous checkpoints.')
            logger.info(f'Loaded previous checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}.')
            logger.info(f'Saved at frame = {prev_checkpoint.frame_num}.')
            logger.info(f'Loss = {prev_checkpoint.loss:.6f}')
            if len(prev_checkpoint.metrics) > 0:
                logger.info('Metrics:')
                for key, val in prev_checkpoint.metrics.items():
                    logger.info(f'\t{key}: {val:.4E}')

            # Clone the previous checkpoint to use as the starting point
            checkpoint = prev_checkpoint.clone()
            checkpoint.step = 0
            checkpoint.parameters = self.parameters
            checkpoint.frame_num = self.master_frame_state.frame_num
            checkpoint.runtime_args = to_dict(self.runtime_args)
            checkpoint.source_args = to_dict(self.source_args)
        else:
            logger.info(f'Found no checkpoints for reconstruction={self.reconstruction.id}. Creating new.')
            checkpoint = MFCheckpoint(
                trial=self.trial,
                reconstruction=self.reconstruction,
                parameters=self.parameters,
                frame_num=self.master_frame_state.frame_num,
                runtime_args=to_dict(self.runtime_args),
                source_args=to_dict(self.source_args),
            )
        checkpoint.save()

        # Set checkpoint reference in trial state
        self.trial_state.checkpoint = checkpoint

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

    def process_trial(self):
        """
        Process the trial.
        """
        p = self.parameters
        ra = self.runtime_args
        direction = self.source_args.direction
        mfs = self.master_frame_state
        self._configure_paths()
        self._init_tb_logger()
        logger.info(f'Logs path: {self.logs_path}.')

        # Initial plots
        self._make_plots(pre_step=True)

        # Get frame numbers to be optimised
        frame_nums = self.trial_state.frame_nums
        if self.source_args.direction == -1:
            frame_nums = frame_nums[::-1]
        if ra.fix_mode:
            frame_nums = frame_nums[1:-1]
        frame_skip = 1 if p.frame_skip is None else p.frame_skip
        first_frame = self.checkpoint.frame_num
        if not ra.finetune_mode and p.window_image_diff_threshold > 0 and p.use_master:
            frame_nums = frame_nums[::frame_skip]
        n_frames = len(frame_nums)
        w2 = int(p.window_size / 2)
        to_skip = 0

        # Train
        for i, frame_num in enumerate(frame_nums):
            if to_skip > 0:
                to_skip -= 1
                continue
            logger.info(f'======== Training frame #{frame_num} ({i + 1}/{n_frames}) ========')
            self.frame_num = frame_num
            self.active_idx = min(i, w2)

            # Reset convergence detection
            self.convergence_detector.reset_counters()

            # Reset frame step counter and train the batch
            self.checkpoint.step_frame = 0
            self.checkpoint.frame_num = frame_num
            self.train(frame_num == first_frame)

            # Save the state
            if p.use_master:
                self.trial_state.update_frame_state(frame_num, mfs)
            else:
                for f in self.frame_batch:
                    self.trial_state.update_frame_state(f.frame_num, f)
            self.trial_state.save()

            # Update the reconstruction
            if direction == 1:
                self.reconstruction.end_frame = max(frame_num + 1, self.reconstruction.end_frame)
            else:
                self.reconstruction.start_frame = min(frame_num, self.reconstruction.start_frame)
            self.reconstruction.save()

            # Interpolate skipped frame parameters
            if p.frame_skip and self.last_frame_state is not None and not ra.finetune_mode \
                    and (p.window_image_diff_threshold > 0 or p.use_master):
                logger.info('Interpolating skipped frames.')
                if direction == 1:
                    start_frame = self.last_frame_state.frame_num
                    end_frame = frame_num + 1
                else:
                    start_frame = frame_num
                    end_frame = self.last_frame_state.frame_num + 1

                for k in BUFFER_NAMES + PARAMETER_NAMES:
                    var = self.trial_state.get(k, start_frame=start_frame, end_frame=end_frame)
                    v = torch.from_numpy(np.stack([var[0], var[-1]]))
                    v = v.to(self.device)
                    v = v.reshape([1, 2, -1])
                    v = v.permute(0, 2, 1)
                    v = F.interpolate(v, size=p.frame_skip + 1, mode='linear', align_corners=True)
                    v = v.squeeze().T.reshape(var.shape)
                    v = v.cpu().numpy()
                    var[1:-1] = v[1:-1]

                # Interpolate the stats
                last_stats = self.last_frame_state.stats
                curr_stats = mfs.stats
                for k, v in curr_stats.items():
                    interpolated_stats = np.linspace(float(last_stats[k]), float(curr_stats[k]), p.frame_skip + 2)
                    if direction == -1:
                        interpolated_stats = interpolated_stats[::-1]
                    for j, ifn in enumerate(range(start_frame + 1, end_frame - 1)):
                        self.trial_state.stats[k][ifn] = float(interpolated_stats[1 + j])
                self.trial_state.save()

            # Make plots
            if ra.plot_every_n_frames > -1:
                if self.last_frame_state is None \
                        and (frame_num - first_frame + 1) % ra.plot_every_n_frames == 0:
                    self._make_plots(final_step=True)
                else:
                    skip = p.frame_skip if p.frame_skip is not None else 1
                    for j in range(skip):
                        if self.last_frame_state is not None:
                            plot_frame_num = self.last_frame_state.frame_num + direction * (j + 1)
                        else:
                            plot_frame_num = frame_num - 1 + direction * (j + 1)
                        if (plot_frame_num - first_frame + 1) % ra.plot_every_n_frames == 0:
                            if plot_frame_num == mfs.frame_num:
                                fs = mfs
                                skipped = False
                            else:
                                fs = self.trial_state.init_frame_state(frame_num=plot_frame_num)
                                skipped = True
                            self._make_plots(final_step=True, frame_state=fs, skipped=skipped)

            if ra.finetune_mode or (p.window_image_diff_threshold == 0 and not p.use_master):
                # In finetune mode just shift the batch along and append the next frames
                f0 = self.frame_batch[0].frame_num + frame_skip
                f1 = min(frame_nums[-1] - 1, self.frame_batch[-1].frame_num + frame_skip)

                # Start the new batch with any from the previous batch
                new_batch = []
                expired_batch = []
                for f in self.frame_batch:
                    if f.frame_num >= f0:
                        new_batch.append(f)
                        f0 = f.frame_num
                    else:
                        expired_batch.append(f)

                # Load the closest expired frame as the last frame state
                self.last_frame_state = expired_batch[-1]
                self.last_frame_state.freeze()

                # Load new frames to fill the batch
                for fn in range(f0 + 1, f1 + 1):
                    f = self.trial_state.init_frame_state(
                        frame_num=fn,
                        trainable=True,
                        load=ra.finetune_mode,
                        prev_frame_state=None if ra.finetune_mode else self.frame_batch[-1],
                        device=self.device,
                    )
                    f.update_ht_data_from_mp()
                    new_batch.append(f)
                self.frame_batch = new_batch

                # Abort if the new batch is too small
                if len(self.frame_batch) < w2:
                    logger.info(f'New batch size {len(new_batch)} < {w2}. Aborting.')
                    break

                # Update master frame state
                mfs.frame_num = self.frame_batch[self.active_idx].frame_num
                mfs.copy_state(self.frame_batch[self.active_idx])

                # Ensure loop skips appropriately
                to_skip = frame_skip - 1

            else:
                # Freeze just-optimised frame
                self.last_frame_state = self.trial_state.init_frame_state(
                    frame_num=frame_num,
                    trainable=False,
                    load=False,
                    prev_frame_state=mfs,
                    device=self.device
                )

                # Roll window
                if i < n_frames - 1:
                    next_frame_num = frame_num + direction * frame_skip  # + w2
                    next_frame = self.trial_state.init_frame_state(
                        frame_num=next_frame_num,
                        trainable=True,
                        load=False,
                        prev_frame_state=mfs,
                        device=self.device
                    )

                    # Reduce curvature
                    if p.curvature_mode and p.curvature_relaxation_factor is not None:
                        K = next_frame.get_state('curvatures')
                        with torch.no_grad():
                            for K_d in K:
                                K_d.data = K_d * p.curvature_relaxation_factor

                    # Shrink length
                    if p.curvature_mode and p.length_shrink_factor is not None:
                        l = next_frame.get_state('length')
                        with torch.no_grad():
                            for d, l_d in enumerate(l):
                                l_d.data = l_d * p.length_shrink_factor
                                self.shrunken_lengths[d] = l_d.detach()

                    # Update master frame state
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
                            threshold=self.parameters.window_image_diff_threshold,
                            direction=direction
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

    def train(self, first_frame: bool = False):
        """
        Train the camera coefficients and multiscale curve.
        """
        logger.info('----- Training the camera coefficients and multiscale curve -----')
        p = self.parameters
        self.optimiser = self._init_optimiser()
        lr_scheduler = self._init_lr_scheduler()
        max_steps = p.n_steps_init if first_frame else p.n_steps_max
        start_step = self.checkpoint.step_frame + 1
        final_step = start_step + max_steps

        # Train the cam coeffs and multiscale curve
        for step in range(start_step, final_step):
            # Centre the worm first so any changes get picked up by the loss
            stats_centre = self._centre_shift()

            # Calculate losses and optimise
            loss, loss_global, losses_depths, stats = self._train_step()

            # At the start of the initialisation stage lock batch to the first frame
            if self.checkpoint.step < self.parameters.n_steps_batch_locked:
                for i, fs in enumerate(self.frame_batch):
                    if i > 0:
                        for k in PARAMETER_NAMES:
                            fs.set_state(k, self.frame_batch[0].get_state(k))

            # Update lr
            if p.algorithm == OPTIMISER_LBFGS_NEW:
                stats['lr'] = self.optimiser_gd.param_groups[1]['lr']
            else:
                stats['lr'] = self.optimiser.param_groups[1]['lr']
            if lr_scheduler is not None:
                lr_scheduler.step(loss.item())

            # Merge stats
            stats = {**stats, **stats_centre}

            # Update steps and checkpoint stats
            self.checkpoint.step += 1
            self.checkpoint.step_frame += 1
            self.checkpoint.loss = loss.item()
            self.checkpoint.metrics = stats
            self.checkpoint.save()

            # Log
            self._log_progress(
                step=step,
                final_step=final_step,
                loss=loss,
                stats=stats
            )

            # Make plots
            self._make_plots(first_frame=first_frame)

            # Update convergence detector
            losses = torch.tensor(
                [loss_global, *losses_depths],
                device=self.device
            )
            self.convergence_detector.forward(losses, first_val=False)

            # When all of the losses have converged and loss has reached target, break
            if not first_frame \
                    and self.convergence_detector.converged.all() \
                    and loss.item() < p.convergence_loss_target \
                    and (p.length_regrow_steps is None or self.checkpoint.step_frame > p.length_regrow_steps):
                break

        self.tb_logger.add_scalar('train_steps', self.checkpoint.step_frame, self.checkpoint.frame_num)

    def _centre_shift(self):
        """
        Shift the curve along its midline so as to centre the scores.
        """
        p = self.parameters
        D = p.depth - p.depth_min
        stats = {}

        if not p.curvature_mode \
                or p.centre_shift_every_n_steps is None \
                or self.checkpoint.step_frame == 0 \
                or (p.length_regrow_steps is not None and self.checkpoint.step_frame < p.length_regrow_steps) \
                or self.checkpoint.step_frame % p.centre_shift_every_n_steps != 0:
            return stats

        points_shifted = 0
        with torch.no_grad():
            for i, fs in enumerate(self.frame_batch):
                if p.curvature_deltas and i != 0:
                    # Centring only needed for the master frame in deltas-mode.
                    break
                scores = fs.get_state('scores')
                X0 = fs.get_state('X0')
                T0 = fs.get_state('T0')
                M10 = fs.get_state('M10')
                length = fs.get_state('length')
                curvatures = fs.get_state('curvatures')

                for d in range(D):
                    # Skip shifting during regrowth phase
                    length_d = length[d]
                    if length_d < p.length_min:
                        continue

                    # Find the midpoint of the tapered scores
                    scores_d = scores[d]
                    N = scores_d.shape[0]
                    old_midpoint = int((N - 1) / 2)
                    scores_d_aa = (scores_d > (scores_d.max() - scores_d.min()) / 2).to(torch.float32)
                    centroid_idx = (torch.arange(N, device=self.device) * scores_d_aa).sum() / scores_d_aa.sum()
                    stats[f'centroid_idx/{d}'] = centroid_idx.item()
                    shift = torch.ceil(centroid_idx).to(torch.int32) - old_midpoint
                    if shift.abs() < p.centre_shift_threshold * N:
                        continue
                    shift.clamp_(min=-p.centre_shift_adj, max=p.centre_shift_adj)
                    new_midpoint = old_midpoint + shift

                    # Compute the curve with the current parameters
                    curvatures_d = curvatures[d]
                    points_d_new, tangents_d_new, M1_new = integrate_curvature(
                        X0[d].unsqueeze(0),
                        T0[d].unsqueeze(0),
                        length[d].unsqueeze(0),
                        curvatures_d.unsqueeze(0),
                        M10[d].unsqueeze(0),
                        integration_algorithm=p.curvature_integration_algorithm,
                    )

                    # Get the new position, tangent and frame values at the new midpoint
                    X0_new = points_d_new[0][new_midpoint]
                    T0_new = tangents_d_new[0][new_midpoint]
                    M10_new = M1_new[0][new_midpoint]

                    # Shift the curvatures and decay towards zero at the new ends
                    decay = torch.linspace(1, 0, shift.abs() + 2, device=self.device)[1:-1]
                    if shift < 0:
                        curvatures_shifted = torch.cat([
                            decay.flip(dims=(0,))[:, None]
                            * torch.repeat_interleave(curvatures_d[0].unsqueeze(0), shift.abs(), dim=0),
                            curvatures_d[:shift],
                        ])
                    else:
                        curvatures_shifted = torch.cat([
                            curvatures_d[shift:],
                            decay[:, None]
                            * torch.repeat_interleave(curvatures_d[-1].unsqueeze(0), shift.abs(), dim=0),
                        ])

                    # Update the parameters to match the shifted curve
                    X0[d].data = X0_new
                    T0[d].data = T0_new
                    M10[d].data = M10_new
                    curvatures[d].data = curvatures_shifted
                    points_shifted += shift.abs()

                # Update the HT data
                fs.update_ht_data_from_mp()

            stats['shifts'] = points_shifted

        return stats

    def _train_step(self) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Train the cam coeffs and multiscale curve for a single step.
        """
        p = self.parameters
        D = p.depth - p.depth_min

        # Outputs
        masks = None
        detection_masks = None
        X0 = None
        T0 = None
        M10 = None
        X0ht = None
        T0ht = None
        M10ht = None
        points_2d = None
        scores = None
        curvatures_smoothed = None
        points_smoothed = None
        sigmas_smoothed = None
        exponents_smoothed = None
        intensities_smoothed = None
        masks_target_residuals = None
        loss = None
        loss_global = None
        losses_depths = None
        stats = None

        def closure():
            nonlocal masks, detection_masks, X0, T0, M10, X0ht, T0ht, M10ht, points_2d, scores, \
                curvatures_smoothed, points_smoothed, sigmas_smoothed, exponents_smoothed, intensities_smoothed, \
                masks_target_residuals, loss, loss_global, losses_depths, stats
            self.optimiser.zero_grad()

            # Collect parameters
            cam_coeffs = torch.stack([f.get_state('cam_coeffs') for f in self.frame_batch])
            cam_shifts = torch.stack([f.get_state('cam_shifts') for f in self.frame_batch])
            cam_rotation_preangles = torch.stack([f.get_state('cam_rotation_preangles') for f in self.frame_batch])
            points_3d_base = torch.stack([f.get_state('points_3d_base') for f in self.frame_batch])
            points_2d_base = torch.stack([f.get_state('points_2d_base') for f in self.frame_batch])
            camera_sigmas = torch.stack([f.get_state('camera_sigmas') for f in self.frame_batch])
            camera_exponents = torch.stack([f.get_state('camera_exponents') for f in self.frame_batch])
            camera_intensities = torch.stack([f.get_state('camera_intensities') for f in self.frame_batch])
            filters = torch.stack([f.get_state('filters') for f in self.frame_batch])
            X0 = [torch.stack([f.get_state('X0')[d] for f in self.frame_batch]) for d in range(D)]
            T0 = [torch.stack([f.get_state('T0')[d] for f in self.frame_batch]) for d in range(D)]
            M10 = [torch.stack([f.get_state('M10')[d] for f in self.frame_batch]) for d in range(D)]
            X0ht = [torch.stack([f.get_state('X0ht')[d] for f in self.frame_batch]) for d in range(D)]
            T0ht = [torch.stack([f.get_state('T0ht')[d] for f in self.frame_batch]) for d in range(D)]
            M10ht = [torch.stack([f.get_state('M10ht')[d] for f in self.frame_batch]) for d in range(D)]
            length = [torch.stack([f.get_state('length')[d] for f in self.frame_batch]) for d in range(D)]
            curvatures = [torch.stack([f.get_state('curvatures')[d] for f in self.frame_batch]) for d in range(D)]
            points = [torch.stack([f.get_state('points')[d] for f in self.frame_batch]) for d in range(D)]
            masks_target = [torch.stack([f.get_state('masks_target')[d] for f in self.frame_batch]) for d in range(D)]
            sigmas = [torch.stack([f.get_state('sigmas')[d] for f in self.frame_batch]) for d in range(D)]
            exponents = [torch.stack([f.get_state('exponents')[d] for f in self.frame_batch]) for d in range(D)]
            intensities = [torch.stack([f.get_state('intensities')[d] for f in self.frame_batch]) for d in range(D)]

            # Check if the length should be fixed
            length_fixed = (p.length_warmup_steps is not None and self.checkpoint.step < p.length_warmup_steps) \
                           or (p.length_regrow_steps is not None and self.checkpoint.step_frame < p.length_regrow_steps)

            # Generate the outputs
            masks, detection_masks, X_raw, T_raw, M1_raw, points_2d, scores, curvatures_smoothed, points_smoothed, sigmas_smoothed, exponents_smoothed, intensities_smoothed = self.model.forward(
                cam_coeffs=cam_coeffs,
                points_3d_base=points_3d_base,
                points_2d_base=points_2d_base,
                X0=X0,
                T0=T0,
                M10=M10,
                X0ht=X0ht,
                T0ht=T0ht,
                M10ht=M10ht,
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
                filters=filters,
                length_warmup=length_fixed,
            )

            # Generate targets with added residuals
            masks_target_residuals = generate_residual_targets(masks_target, masks, detection_masks)

            # Update HT data
            if p.curvature_integration == CURVATURE_INTEGRATION_MIDPOINT:
                X0ht = [
                    torch.stack([X_raw[d][:, 0, 0], X_raw[d][:, 0, -1]], dim=1)
                    for d in range(D)
                ]
                T0ht = [
                    torch.stack([T_raw[d][:, 0, 0], T_raw[d][:, 0, -1]], dim=1)
                    for d in range(D)
                ]
                M10ht = [
                    torch.stack([M1_raw[d][:, 0, 0], M1_raw[d][:, 0, -1]], dim=1)
                    for d in range(D)
                ]

            # Update midpoint data
            elif p.curvature_integration == CURVATURE_INTEGRATION_HT:
                X0 = [
                    X_raw[d][:, 0, int((2**(p.depth_min + d)) / 2)]
                    for d in range(D)
                ]
                T0 = [
                    T_raw[d][:, 0, int((2**(p.depth_min + d)) / 2)]
                    for d in range(D)
                ]
                M10 = [
                    M1_raw[d][:, 0, int((2**(p.depth_min + d)) / 2)]
                    for d in range(D)
                ]

            # Calculate the losses
            loss, loss_global, losses_depths, stats = self._calculate_losses(
                cam_rotation_preangles=cam_rotation_preangles,
                X0=X0,
                T0=T0,
                M10=M10,
                X0ht=X0ht,
                T0ht=T0ht,
                M10ht=M10ht,
                length=length,
                curvatures=curvatures,
                X_raw=X_raw,
                T_raw=T_raw,
                M1_raw=M1_raw,
                points=points,
                masks_target=mtr_for_loss,
                sigmas=sigmas,
                exponents=exponents,
                intensities=intensities,
                camera_sigmas=camera_sigmas,
                camera_exponents=camera_exponents,
                camera_intensities=camera_intensities,
                cam_shifts=cam_shifts,
                masks=masks_for_loss,
                scores=scores_for_loss,
                curvatures_smoothed=curvatures_smoothed,
                points_smoothed=points_smoothed,
                sigmas_smoothed=sigmas_smoothed,
            )

            # Get fix-loss
            if self.runtime_args.fix_mode:
                loss_fix = self._calculate_fix_loss(loss, stats)
                loss_to_minimise = loss_fix
            else:
                loss_to_minimise = loss

            return loss_to_minimise

        # Take optimisation step
        self.optimiser.zero_grad()
        l2m = closure()
        l2m.backward()
        if p.algorithm == OPTIMISER_LBFGS_NEW:
            options = {'closure': closure, 'current_loss': loss, 'ls_debug': False, 'damping': False, 'eps': 1e-3,
                       'eta': 2, 'max_ls': 10}
            self.optimiser.step(options)

            # Update non-lbfgs parameters
            l2m = closure()
            l2m.backward()
            self.optimiser_gd.step()
        else:
            self.optimiser.step()
        if is_bad(loss):
            raise RuntimeError('Bad loss!')

        # Update frames
        self._update_frame_states(masks, masks_target_residuals, X0, T0, M10, X0ht, T0ht, M10ht, points_2d, scores,
                                  sigmas_smoothed, exponents_smoothed, intensities_smoothed,
                                  points_smoothed, curvatures_smoothed, stats)

        # Clamp parameters
        self._clamp_parameters(points_smoothed)

        return loss, loss_global, losses_depths, stats

    def _update_frame_states(
            self,
            masks: List[torch.Tensor],
            masks_target_residuals: List[torch.Tensor],
            X0: List[torch.Tensor],
            T0: List[torch.Tensor],
            M10: List[torch.Tensor],
            X0ht: List[torch.Tensor],
            T0ht: List[torch.Tensor],
            M10ht: List[torch.Tensor],
            points_2d: List[torch.Tensor],
            scores: List[torch.Tensor],
            sigmas_smoothed: List[torch.Tensor],
            exponents_smoothed: List[torch.Tensor],
            intensities_smoothed: List[torch.Tensor],
            points_smoothed: List[torch.Tensor],
            curvatures_smoothed: List[torch.Tensor],
            stats: Dict[str, float],
    ):
        """
        Update the frame states.
        """
        p = self.parameters
        D = p.depth - p.depth_min

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
            self.master_frame_state.set_state('points',
                                              [points_smoothed[d][self.active_idx] for d in range(D)])
            self.master_frame_state.set_state('curvatures_smoothed',
                                              [curvatures_smoothed[d][self.active_idx] for d in range(D)])

            # Update HT data
            if p.curvature_integration == CURVATURE_INTEGRATION_MIDPOINT:
                self.master_frame_state.set_state('X0ht', [X0ht[d][self.active_idx] for d in range(D)])
                self.master_frame_state.set_state('T0ht', [T0ht[d][self.active_idx] for d in range(D)])
                self.master_frame_state.set_state('M10ht', [M10ht[d][self.active_idx] for d in range(D)])

            elif p.curvature_integration == CURVATURE_INTEGRATION_HT:
                self.master_frame_state.set_state('X0', [X0[d][self.active_idx] for d in range(D)])
                self.master_frame_state.set_state('T0', [T0[d][self.active_idx] for d in range(D)])
                self.master_frame_state.set_state('M10', [M10[d][self.active_idx] for d in range(D)])

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

                # Update HT data
                if p.curvature_integration == CURVATURE_INTEGRATION_MIDPOINT:
                    fs.set_state('X0ht', [X0ht[d][i] for d in range(D)])
                    fs.set_state('T0ht', [T0ht[d][i] for d in range(D)])
                    fs.set_state('M10ht', [M10ht[d][i] for d in range(D)])

                elif p.curvature_integration == CURVATURE_INTEGRATION_HT:
                    fs.set_state('X0', [X0[d][i] for d in range(D)])
                    fs.set_state('T0', [T0[d][i] for d in range(D)])
                    fs.set_state('M10', [M10[d][i] for d in range(D)])

            else:
                fs.set_state('curvatures', [curvatures_smoothed[d][i] for d in range(D)])

    def _calculate_losses(
            self,
            cam_rotation_preangles: torch.Tensor,
            X0: List[torch.Tensor],
            T0: List[torch.Tensor],
            M10: List[torch.Tensor],
            X0ht: List[torch.Tensor],
            T0ht: List[torch.Tensor],
            M10ht: List[torch.Tensor],
            length: List[torch.Tensor],
            curvatures: List[torch.Tensor],
            X_raw: List[torch.Tensor],
            T_raw: List[torch.Tensor],
            M1_raw: List[torch.Tensor],
            points: List[torch.Tensor],
            masks_target: List[torch.Tensor],
            sigmas: List[torch.Tensor],
            exponents: List[torch.Tensor],
            intensities: List[torch.Tensor],
            camera_sigmas: torch.Tensor,
            camera_exponents: torch.Tensor,
            camera_intensities: torch.Tensor,
            cam_shifts: torch.Tensor,
            masks: List[torch.Tensor],
            scores: List[torch.Tensor],
            curvatures_smoothed: List[torch.Tensor],
            points_smoothed: List[torch.Tensor],
            sigmas_smoothed: List[torch.Tensor],
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
            M10_prev = self.last_frame_state.get_state('M10')
            X0ht_prev = self.last_frame_state.get_state('X0ht')
            T0ht_prev = self.last_frame_state.get_state('T0ht')
            M10ht_prev = self.last_frame_state.get_state('M10ht')
            length_prev = self.last_frame_state.get_state('length')
            curvatures_prev = self.last_frame_state.get_state('curvatures')
            points_prev = self.last_frame_state.get_state('points')
            cam_shifts_prev = self.last_frame_state.get_state('cam_shifts')
        else:
            X0_prev = None
            T0_prev = None
            M10_prev = None
            X0ht_prev = None
            T0ht_prev = None
            M10ht_prev = None
            length_prev = None
            curvatures_prev = None
            points_prev = None
            cam_shifts_prev = None

        # Losses calculated at each depth
        losses = {
            'masks': calculate_renders_losses(masks, masks_target, p.loss_masks_metric, p.loss_masks_multiscale),
            'scores': calculate_scores_losses(scores),
        }

        if p.curvature_mode:
            losses = {**losses, **{
                'parents': calculate_parents_losses_curvatures(
                    X0, T0, M10, X0ht, T0ht, M10ht, length, curvatures, curvatures_smoothed
                ),
                'smoothness': calculate_smoothness_losses_curvatures(curvatures),
                'intersections': calculate_intersection_losses_curvatures(
                    points_smoothed, sigmas_smoothed, p.curvature_max
                ),
                'alignment': calculate_alignment_losses_curvatures(curvatures, curvatures_prev),
                # 'consistency': calculate_consistency_losses_curvatures(X_raw, T_raw, M1_raw),
                'consistency': calculate_consistency_losses_curvatures_ht(X_raw, T_raw, M1_raw),
            }}
            if p.curvature_deltas:
                losses['temporal'] = calculate_temporal_losses_curvature_deltas(
                    X0, T0, M10, length, curvatures,
                    X0_prev, T0_prev, M10_prev, length_prev, curvatures_prev
                )
                losses['curvature'] = calculate_curvature_losses_curvature_deltas(curvatures)
            else:
                if p.curvature_integration == CURVATURE_INTEGRATION_MIDPOINT:
                    losses['temporal'] = calculate_temporal_losses_curvatures(
                        length, curvatures, X0, T0, M10, None, None, None, cam_shifts,
                        length_prev, curvatures_prev, X0_prev, T0_prev, M10_prev, None, None, None, cam_shifts_prev
                    )
                else:
                    # losses['temporal'] = calculate_temporal_losses_curvatures(
                    #     length, curvatures, None, None, None, X0ht, T0ht, M10ht, cam_shifts,
                    #     length_prev, curvatures_prev, None, None, None, X0ht_prev, T0ht_prev, M10ht_prev,
                    #     cam_shifts_prev
                    # )
                    losses['temporal'] = calculate_temporal_losses_curvatures(
                        length, curvatures, None, None, None, X0ht, None, None, cam_shifts,
                        length_prev, curvatures_prev, None, None, None, X0ht_prev, None, None,
                        cam_shifts_prev
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

    def _calculate_fix_loss(self, loss: torch.Tensor, stats: dict) -> torch.Tensor:
        """
        In fix-mode, the start and end frames are targets that must be reached.
        """
        ra = self.runtime_args
        assert ra.fix_mode
        mfs = self.master_frame_state
        start_fs = self.fix_target_start
        end_fs = self.fix_target_end

        # Calculate target losses - squared distances from all parameters to each end point.
        loss_start = 0.
        loss_end = 0.
        for k in PARAMETER_NAMES:
            if k == 'M10':
                # Trying to join M10 causes more trouble than it's worth
                continue
            if k == 'curvatures':
                # Skip curvature matching for now...
                continue
            p_start = start_fs.get_state(k)
            p_end = end_fs.get_state(k)
            p_curr = mfs.get_state(k)
            if type(p_start) == list:
                if k == 'curvatures':
                    # Just try to match the absolute curvature, not the exact m1/m2 split
                    p_start = [p_start[i].sum(dim=-1) for i in range(len(p_start))]
                    p_end = [p_end[i].sum(dim=-1) for i in range(len(p_end))]
                    p_curr = [p_curr[i].sum(dim=-1) for i in range(len(p_curr))]

                for i in range(len(p_start)):
                    loss_start += torch.sum((p_start[i] - p_curr[i])**2)
                    loss_end += torch.sum((p_end[i] - p_curr[i])**2)
            else:
                loss_start += torch.sum((p_start - p_curr)**2)
                loss_end += torch.sum((p_end - p_curr)**2)

        # Calculate relative loss weightings
        n_frames = abs(start_fs.frame_num - end_fs.frame_num) + 1
        decay = torch.exp(-torch.arange(n_frames) / ra.fix_decay_rate)
        bi_decay = decay + decay.flip(dims=(0,))
        bi_decay = bi_decay / bi_decay.max()
        loss_weighting = 1 - bi_decay
        if self.source_args.direction == 1:
            idx = mfs.frame_num - start_fs.frame_num
        else:
            idx = start_fs.frame_num - mfs.frame_num

        # Weight and sum the losses
        loss_start_weighted = decay[idx] * loss_start
        loss_end_weighted = decay.flip(dims=(0,))[idx] * loss_end
        loss_weighted = loss_weighting[idx] * loss
        loss_fix = loss_start_weighted + loss_end_weighted + loss_weighted

        # Log losses
        stats['loss_fix/start'] = loss_start.item()
        stats['loss_fix/start_weighted'] = loss_start_weighted.item()
        stats['loss_fix/end'] = loss_end.item()
        stats['loss_fix/end_weighted'] = loss_end_weighted.item()
        stats['loss_fix/total'] = loss_fix.item()
        if self.runtime_args.log_level > 0:
            stats['loss_fix/weights_start'] = decay[idx].item()
            stats['loss_fix/weights_end'] = decay.flip(dims=(0,))[idx].item()
            stats['loss_fix/weights_loss'] = loss_weighting[idx].item()

        return loss_fix

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
                    T0 = fs.get_state('T0')
                    M10 = fs.get_state('M10')
                    T0ht = fs.get_state('T0ht')
                    M10ht = fs.get_state('M10ht')
                    length = fs.get_state('length')
                    curvatures = fs.get_state('curvatures')
                    for d in range(D):
                        # Tangents should be normalised
                        T0[d].data = normalise(T0[d])
                        T0ht[d].data = normalise(T0ht[d])

                        # M1 should be orthogonal to T and normalised
                        M10_d = normalise(M10[d])
                        M10_d = orthogonalise(M10_d.unsqueeze(0), T0[d].unsqueeze(0))[0]
                        M10[d].data = normalise(M10_d)

                        M10ht_d = normalise(M10ht[d])
                        M10ht_d = orthogonalise(M10ht_d, T0ht[d])
                        M10ht[d].data = normalise(M10ht_d)

                        # In length warmup phase, linearly grow the length from length_init to length_min
                        if not self.runtime_args.resume \
                                and p.length_warmup_steps is not None \
                                and self.checkpoint.step < p.length_warmup_steps:
                            l = torch.tensor(
                                p.length_init
                                + (p.length_min - p.length_init) * self.checkpoint.step / p.length_warmup_steps,
                                device=self.device
                            )

                        # In regrowth phase, linearly grow the length from length_init to length_min
                        elif p.length_regrow_steps is not None \
                                and self.checkpoint.step_frame < p.length_regrow_steps \
                                and self.shrunken_lengths[d] < p.length_min:
                            l = self.shrunken_lengths[d] \
                                + (p.length_min - self.shrunken_lengths[d]) \
                                * self.checkpoint.step_frame / p.length_regrow_steps

                        # Otherwise, ensure that the worm does not get too long/short.
                        else:
                            l = length[d].clamp(
                                min=p.length_min,
                                max=p.length_max
                            )

                            if p.dl_limit is not None:
                                l_prev = None
                                if i == 0 and self.last_frame_state is not None:
                                    l_prev = self.last_frame_state.get_state('length')[d]
                                elif i > 0:
                                    l_prev = self.frame_batch[i - 1].get_state('length')[d]
                                if l_prev is not None:
                                    l = l.clamp(
                                        min=l_prev - p.dl_limit,
                                        max=l_prev + p.dl_limit,
                                    )
                        length[d].data = l

                        # Ensure curvature doesn't get too large.
                        K = curvatures[d]
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
                    'min': p.sigmas_min + 0.001,
                    'max': p.sigmas_max,
                },
                'exponents': {
                    'min': 0.5,
                    'max': 10
                },
                'intensities': {
                    'min': p.intensities_min + 0.01,
                    'max': 10
                }
            }

            for sei in ['sigmas', 'exponents', 'intensities']:
                if p.use_master:
                    params = [*self.master_frame_state.get_state(sei), ]
                else:
                    params = [pp for f in self.frame_batch for pp in f.get_state(sei)]
                for v in params:
                    v.data = v.clamp(**render_parameters_limits[sei])

                # Camera scaling factors should average 1 and not be more than 30% from the mean
                if p.use_master:
                    params = [self.master_frame_state.get_state(f'camera_{sei}'), ]
                else:
                    params = [f.get_state(f'camera_{sei}') for f in self.frame_batch]
                for v in params:
                    v.data = v / v.mean()
                    sf = 0.3 / ((v - 1).abs()).amax()
                    if sf < 1:
                        v.data = (v - 1) * sf + 1

            # Camera filters should be normalised
            filters = self.master_frame_state.get_state('filters')
            filters.data = filters / filters.norm(dim=(1, 2), keepdim=True)

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
                for di, d in enumerate(range(self.parameters.depth_min, self.parameters.depth + 1)):
                    d_str = 'global' if di == 0 else d
                    for k in state_vars:
                        self.tb_logger.add_scalar(f'detector/{d_str}/{k}', getattr(detector, k)[di].item(), step)

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
            if self.parameters.window_size > 1:
                self._plot_2d_batch_basic()
                # self._plot_2d_batch()
            else:
                self._plot_2d(frame_state, skipped)
            if self.parameters.curvature_mode:
                self._plot_curvatures()
            if self.parameters.filter_size is not None and self.parameters.filter_size > 0:
                self._plot_filters()
            self._plot_point_stats(frame_state, skipped)

    def _plot_3d(self, frame_state: FrameState, skipped: bool = False):
        """
        Make a multiscale curve 3D scatter plot.
        """
        if not self.runtime_args.plot_3d:
            return
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
                    title += f', l={length[d]:.3f}'
                ax.set_title(title)
                # cla(ax)

                # Scale vertices by sigmas and colour by score
                scores_d = to_numpy(scores[d])
                sigmas_d = np.clip(
                    2000 * to_numpy(sigmas[d]),
                    a_min=10,
                    a_max=1000,
                )

                # Scatter smoothed output points generated by integrating the curvature
                vertices = to_numpy(points[d])
                x, y, z = (vertices[:, j] for j in range(3))
                ax.scatter(x, y, z, c=scores_d, cmap=cmap_vertices, s=sigmas_d, alpha=0.4)

                # Draw lines connecting vertices
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
        if not self.runtime_args.plot_2d:
            return
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        n_rows = int(np.ceil(np.sqrt(D - D_min)))
        n_cols = int(np.ceil(np.sqrt(D - D_min)))

        fig = plt.figure(figsize=(1 + n_cols * 3, 1 + n_rows * 3))
        fig.suptitle(self._plot_title(frame_state, skipped))
        gs = GridSpec(n_rows, n_cols)
        d = 0

        masks_target = frame_state.get_state('masks_target_residuals')
        points_2d = frame_state.get_state('points_2d')
        masks_curve = frame_state.get_state('masks_curve')
        length = frame_state.get_state('length')

        images = to_numpy(frame_state.get_state('images'))
        image_triplet = np.concatenate(images, axis=1)
        image_grid = np.concatenate([np.ones_like(image_triplet), image_triplet, np.ones_like(image_triplet)], axis=0)
        M, N = tuple(image_grid.shape)

        scatter_sizes = (np.linspace(1, 0, D)**2 * 100) + 0.3

        for i in range(n_rows):
            for j in range(n_cols):
                if d + D_min >= D:
                    break
                ax = fig.add_subplot(gs[i, j])
                title = f'd={d + D_min}'
                if self.parameters.curvature_mode:
                    title += f', l={length[d]:.3f}'
                ax.set_title(title)
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
                if not np.isnan(X_curve_triplet).any():
                    alphas = X_curve_triplet.copy() * 0.5
                    ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                              extent=(0, N - 1, 2 * int(M / 3), int(M / 3)))

                # Scatter the midline points
                p2d = to_numpy(points_2d[d]).transpose(1, 0, 2)
                for k in range(3):
                    p = p2d[k] + (0, self.trial.crop_size)
                    if k == 1:
                        p += (self.trial.crop_size, 0)
                    elif k == 2:
                        p += (self.trial.crop_size * 2, 0)
                    ax.scatter(p[:, 0], p[:, 1], cmap='jet', c=np.linspace(0, 1, len(p)), s=scatter_sizes[d + D_min],
                               alpha=0.6)

                # Errors
                errors_triplet = X_curve_triplet - X_target_triplet
                errors_triplet = errors_triplet / np.abs(errors_triplet).max()
                ax.imshow(errors_triplet, vmin=-1, vmax=1, cmap='PRGn', aspect='equal',
                          extent=(0, N - 1, M - 1, 2 * int(M / 3)))

                d += 1

        fig.tight_layout()
        self._save_plot(fig, '2D', frame_state)

    def _plot_2d_batch(self):
        """
        Plot a batch of 2D mask renderings of the mutiscale curves.
        """
        if not self.runtime_args.plot_2d:
            return
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
                    p = p2d[k] + (0, self.trial.crop_size)
                    if k == 1:
                        p += (self.trial.crop_size, 0)
                    elif k == 2:
                        p += (self.trial.crop_size * 2, 0)
                    ax.scatter(p[:, 0], p[:, 1], cmap='jet', c=np.linspace(0, 1, len(p)), s=scatter_sizes[d], alpha=0.6)

                # Errors
                errors_triplet = X_curve_triplet - X_target_triplet
                errors_triplet = errors_triplet / np.abs(errors_triplet).max()
                ax.imshow(errors_triplet, vmin=-1, vmax=1, cmap='PRGn', aspect='equal',
                          extent=(0, N - 1, M - 1, 2 * int(M / 3)))

        self._save_plot(fig, '2D_batch', self.master_frame_state)

    def _plot_2d_batch_basic(self):
        """
        Plot a basic version of a batch of 2D mask renderings.
        """
        if not self.runtime_args.plot_2d:
            return
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        n_cols = D - D_min
        if n_cols != 1:
            raise RuntimeError('Basic 2d batch plotting only available at one depth!')
        n_rows = self.parameters.window_size

        fig, ax = plt.subplots(1, figsize=(4, n_rows * 2), gridspec_kw=dict(
            wspace=0.01,
            hspace=0.02,
            top=0.98,
            bottom=0.02,
            left=0.02,
            right=0.98
        ))
        ax.set_title(self._plot_title(self.master_frame_state))

        # Stitch images together and add labels on the left
        frames = []
        bs = len(self.frame_batch)
        label_positions = np.linspace(1 - 1 / bs / 2, 1 / bs / 2, bs)  # + 1)[1:-1]
        for i, frame_state in enumerate(self.frame_batch):
            images = to_numpy(frame_state.get_state('images'))
            frames.append(np.concatenate(1 - images, axis=1))
            ax.text(-0.01, label_positions[i], frame_state.frame_num, transform=ax.transAxes, rotation='vertical',
                    fontsize='x-small', horizontalalignment='center', verticalalignment='bottom')
        images = np.concatenate(frames, axis=0)

        # Draw images
        ax.axis('off')
        ax.imshow(images, cmap='gray', vmin=0, vmax=1)

        # Scatter the midline points
        for i, frame_state in enumerate(self.frame_batch):
            points_2d = frame_state.get_state('points_2d')[0]
            p2d = to_numpy(points_2d).transpose(1, 0, 2)
            for k in range(3):
                p = p2d[k] + (0, i * self.trial.crop_size)
                if k == 1:
                    p += (self.trial.crop_size, 0)
                elif k == 2:
                    p += (self.trial.crop_size * 2, 0)
                ax.scatter(p[:, 0], p[:, 1], cmap='jet', c=np.linspace(0, 1, len(p)), s=0.5, alpha=0.6)

        self._save_plot(fig, '2D_batch', self.master_frame_state)

    def _plot_point_stats(self, frame_state: FrameState, skipped: bool = False):
        """
        Plot the point scores, sigmas, exponents and intensities for the curves.
        """
        if not self.runtime_args.plot_point_stats:
            return
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
        if not self.runtime_args.plot_curvatures:
            return
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        cmap = plt.get_cmap('jet')

        n_rows = 6 if self.parameters.curvature_deltas else 4
        fig, axes = plt.subplots(n_rows, D - D_min, figsize=((D - D_min) * 6 + 2, 10), squeeze=False)
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
        if self.parameters.curvature_deltas:
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
                if self.parameters.curvature_deltas and i > 0:
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

    def _plot_filters(self):
        """
        Plot the filters.
        """
        if not self.runtime_args.plot_filters:
            return
        fig, axes = plt.subplots(len(self.frame_batch), 3, figsize=(10, 4), squeeze=False)
        fig.suptitle(self._plot_title(self.master_frame_state))

        for i, frame_state in enumerate(self.frame_batch):
            filters = to_numpy(frame_state.get_state('filters'))
            for j in range(3):
                ax = axes[i, j]
                if i == 0:
                    ax.set_title(f'Camera {j}')
                if j == 0:
                    ax.set_ylabel(f'Frame {frame_state.frame_num}')
                im = ax.imshow(filters[j], cmap=plt.cm.PRGn, norm=MidpointNormalize(midpoint=0))
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax)

        fig.tight_layout()
        self._save_plot(fig, 'filters', frame_state)

    def _plot_title(self, frame_state: FrameState, skipped: bool = False) -> str:
        """
        Generate a title for the plots.
        """
        title = f'Trial: {self.trial.id}. {self.trial.date:%Y%m%d}. ' \
                f'Frame: {frame_state.frame_num}/{self.trial.n_frames_min}.\n' \
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
            fn = f'{frame_state.frame_num:05d}_{self.step:06d}.{img_extension}'
            if self.runtime_args.prefix_seed_to_plot_names:
                fn = f'{self.runtime_args.seed:02d}_' + fn
            path = save_dir / fn
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(plot_type, fig, self.step)
            self.tb_logger.flush()

        plt.close(fig)
