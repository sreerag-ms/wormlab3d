import os
import os
import time
from typing import Tuple, Union, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mongoengine import DoesNotExist
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from wormlab3d import logger, LOGS_PATH, ROOT_PATH
from wormlab3d.data.model import Trial, Cameras, MFCheckpoint, MFModelParameters, Checkpoint
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_POINTS
from wormlab3d.midlines3d.args_finder import ModelArgs, OptimiserArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.args_finder.optimiser_args import LOSS_CURVE_TARGET_MASKS
from wormlab3d.midlines3d.frame_state import FrameState, TrialState, BUFFER_NAMES, PARAMETER_NAMES
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel, avg_pool_2d
from wormlab3d.nn.args.optimiser_args import LOSS_MSE, LOSS_LOGDIFF, LOSS_KL
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict, hash_data

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')
cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'

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

        # Initialise the skeletoniser
        self.skelnet = self._init_skelnet()

        # Initialise the model
        self.model, self.model_params = self._init_model()

        # Load the trial and initialise trainable parameters
        self.masks = None  # todo
        self._init_trial()

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
        if checkpoint.masks is not None:
            return LOGS_PATH + \
                   f'/trial_{checkpoint.masks.frame.trial.id}' \
                   f'/frame_{checkpoint.masks.frame.frame_num:06d}_{checkpoint.masks.id}' \
                   f'/{checkpoint.model_params.created:%Y%m%d_%H:%M}_{checkpoint.model_params.id}'

        identifiers = {
            'model_params': str(checkpoint.model_params.id),
            **to_dict(checkpoint.source_args),
            **to_dict(checkpoint.optimiser_args)
        }
        arg_hash = hash_data(identifiers)

        return LOGS_PATH + \
               f'/trial_{checkpoint.trial.id}' \
               f'/{arg_hash}'

    @property
    def step(self):
        return self.checkpoint.step_cc + self.checkpoint.step_curve

    # def __getattr__(self, key):
    #     """
    #     Allow parameters to be accessed as member variables.
    #     """
    #     return self.parameters.get(key)

    def _init_skelnet(self) -> BaseNet:
        """
        Build the skeletoniser network.
        """
        if self.source_args.masks_id is not None:
            return

        assert self.model_args.skeletoniser_id is not None, \
            'A checkpoint id for the skeletoniser must be provided.'

        cp = Checkpoint.objects.get(id=self.model_args.skeletoniser_id)
        logger.info(f'Loaded skeletoniser checkpoint (id={cp.id}, created={cp.created}).')

        # Instantiate the network
        net = cp.network_params.instantiate_network()
        logger.info(f'Instantiated SkelNet with {net.get_n_params() / 1e6:.4f}M parameters.')
        logger.debug(f'----------- SkelNet --------------\n\n{net}\n\n')

        # Load the network parameter states
        path = ROOT_PATH + '/logs/scripts/midlines2d/train' \
                           f'/{cp.dataset.created:%Y%m%d_%H:%M}_{cp.dataset.id}' \
                           f'/{cp.network_params.created:%Y%m%d_%H:%M}_{cp.network_params.id}' \
                           f'/checkpoints/{cp.id}.chkpt'
        state = torch.load(path, map_location=torch.device('cpu'))
        net.load_state_dict(state['model_state_dict'], strict=False)
        net.eval()
        logger.info(f'Loaded SkelNet state from "{path}"')

        return net

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
        model = ProjectRenderScoreModel()

        return model, model_params

    def _init_trial(self):
        """
        Load the trial.
        """
        logger.info('Initialising trial state.')
        self.trial: Trial = Trial.objects.get(id=self.source_args.trial_id)

        # Worm length (needed?)
        self.worm_length_db = torch.tensor(self.trial.experiment.worm_length)
        logger.debug(f'Worm length (db) = {self.worm_length_db:.2f}.')

        # Prepare trial state
        self.trial_state = TrialState(
            trial=self.trial,
            start_frame=self.source_args.start_frame,
            end_frame=self.source_args.end_frame,
            model_params=self.model_params,
            optimiser_args=self.optimiser_args
        )

        # Prepare batch state
        self.frame_batch: List[FrameState] = []
        for i in range(self.optimiser_args.window_size):
            self.frame_batch.append(self._init_frame_state(self.trial_state.frame_nums[i]))

        # Initialise parameters
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            # setattr(self, k, torch.stack([fs.get_state(k) for fs in self.frame_batch]))
            setattr(self, k, [fs.get_state(k) for fs in self.frame_batch])

    def _init_frame_state(self, frame_num: int) -> FrameState:
        """
        Load the frame
        """
        frame = self.trial.get_frame(frame_num)
        logger.info(f'Initialising frame state for frame #{frame_num} (id={frame.id}).')

        # Generate segmentation masks for the next frame
        logger.debug('Generating masks.')
        images = torch.from_numpy(np.stack(frame.images))  # .unsqueeze(0)
        masks = torch.zeros_like(images)  # self.skelnet.forward(images)
        masks[masks > self.source_args.masks_target_ceil_threshold] = 1
        masks = masks / masks.sum(axis=(1, 2), keepdim=True)

        # Load cameras
        cameras: Cameras = frame.get_cameras()

        # Initialise frame state
        frame_state = FrameState(
            frame_num=frame_num,
            images=images,
            masks_target=masks,
            cameras=cameras,
            points_3d_base=torch.tensor(frame.centre_3d.point_3d),
            points_2d_base=torch.tensor(frame.centre_3d.reprojected_points_2d),
            worm_length_db=self.worm_length_db,
            model_params=self.model_params,
            optimiser_args=self.optimiser_args,
        )

        return frame_state

    def _init_optimisers(self) -> Tuple[Optimizer, Optimizer]:
        """
        Set up the joint cameras and cloud optimiser and the curve optimiser.
        """
        logger.info('Initialising optimisers.')
        oa = self.optimiser_args

        cls_cc: Optimizer = getattr(torch.optim, oa.algorithm_cc)
        optimiser_cc = cls_cc(
            [
                {'params': self.cam_coeffs, 'lr': oa.lr_cam_coeffs},
                {'params': self.cloud_points, 'lr': oa.lr_cloud_points},
                {'params': self.blur_sigmas_cloud, 'lr': oa.lr_cloud_sigmas},
            ],
            amsgrad=True,
            weight_decay=0
        )

        cls_curve: Optimizer = getattr(torch.optim, oa.algorithm_cc)
        optimiser_curve = cls_curve(
            params=[
                {'params': self.curve_parameters, 'lr': oa.lr_curve_points},
                {'params': self.curve_length, 'lr': oa.lr_curve_points},
                {'params': self.blur_sigmas_curve, 'lr': oa.lr_curve_sigmas},
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
            logger.info('Using GPU.')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
            for frame_state in self.frame_batch:
                frame_state.to(device)
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
                if self.runtime_args.resume_from in ['latest', 'best_cc', 'best_curve']:
                    if self.runtime_args.resume_from == 'latest':
                        order_by = '-created'
                    elif self.runtime_args.resume_from == 'best_cc':
                        order_by = '+loss_cc'
                    else:
                        order_by = '+loss_curve'

                    if self.source_args.masks_id is not None:
                        prev_checkpoints = MFCheckpoint.objects(
                            masks=self.masks,
                            model_params=self.model_params
                        ).order_by(order_by)
                    else:
                        prev_checkpoints = MFCheckpoint.objects(
                            trial=self.trial,
                            model_params=self.model_params
                        ).order_by(order_by)

                    if prev_checkpoints.count() > 0:
                        logger.info(
                            f'Found {prev_checkpoints.count()} previous checkpoints. '
                            f'Using {self.runtime_args.resume_from}.'
                        )
                        prev_checkpoint = prev_checkpoints[0]
                    else:
                        src = f'trial={self.trial.id}' if self.source_args.trial_id is not None else f'masks={self.masks.id}'
                        logger.error(
                            f'Found no checkpoints for {src} and model={self.model_params.id}.'
                        )
                        raise DoesNotExist()
                else:
                    prev_checkpoint = MFCheckpoint.objects.get(
                        id=self.runtime_args.resume_from
                    )
                logger.info(f'Loaded checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}.')
                if self.source_args.trial_id is not None:
                    logger.info(f'Frame number = {prev_checkpoint.frame_num}')
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
            for i, fs in enumerate(self.frame_batch):
                for p in PARAMETER_NAMES:
                    fs.set_state(p, state[p][i])
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
                trial=self.trial,
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
        os.makedirs(self.logs_path + '/videos', exist_ok=True)

    def save_checkpoint(self):
        """
        Save the checkpoint information to the database and the parameters to file.
        """
        logger.info('Saving checkpoint...')
        self.checkpoint.save()
        path = f'{self.logs_path}/checkpoints/{self.checkpoint.id}.chkpt'
        params = {
            p: torch.stack([fs.get_state(p) for fs in self.frame_batch])
            for p in PARAMETER_NAMES
        }
        torch.save({
            **params,
            'optimiser_cc_state_dict': self.optimiser_cc.state_dict(),
            'optimiser_curve_state_dict': self.optimiser_curve.state_dict(),
        }, path)

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint
        self.checkpoint = self.checkpoint.clone()

        # Update checkpoint in TrialState
        self.trial_state.checkpoint = self.checkpoint

    def process_trial(self):
        """
        Process the trial.
        """
        oa = self.optimiser_args
        self._configure_paths()
        self._init_tb_logger()

        # Initial plots
        self._make_plots(pre_step=True, show_curve=self.checkpoint.step_curve > 0)

        # Train
        w2 = int((oa.window_size - 1) / 2)
        first_frame = self.checkpoint.frame_num
        n_frames = len(self.trial_state) - first_frame + 1

        for i, frame_num in enumerate(range(first_frame, self.trial_state.frame_nums[-1])):
            logger.info(f'======== Training frame #{frame_num} ({i}/{n_frames}) ========')

            # Reset counters and train the batch
            self.checkpoint.frame_num = frame_num
            self.checkpoint.step_cc = 0
            self.checkpoint.step_curve = 0

            if i == 0:
                self.train(oa.n_steps_cc_init, oa.n_steps_curve_init)
            else:
                self.train(oa.n_steps_cc, oa.n_steps_curve)

            # Save the state
            active_idx = min(i, w2)
            self.trial_state.update_frame_state(frame_num, self.frame_batch[active_idx])
            self.trial_state.save()

            # Roll window
            if i > w2:
                for j in range(oa.window_size):
                    curr_frame = self.frame_batch[j]

                    if j + 1 < oa.window_size:
                        next_frame = self.frame_batch[j + 1]
                    elif i + w2 < len(self.trial_state):
                        next_frame = self._init_frame_state(i + w2)

                    with torch.no_grad():
                        for n in BUFFER_NAMES + PARAMETER_NAMES:
                            curr_frame.set_state(n, next_frame.get_state(n))
                    curr_frame.frame_num = next_frame.frame_num

                    if j < active_idx:
                        self.frame_batch[j].freeze()

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_frames > 0 \
                    and frame_num % self.runtime_args.checkpoint_every_n_frames == 0:
                self.save_checkpoint()

    def train(self, n_steps_cc: int, n_steps_curve: int):
        """
        Train a batch of frames.
        """
        oa = self.optimiser_args

        # Train cc
        if n_steps_cc > 0 and (oa.optimise_cloud or oa.optimise_cam_coeffs):
            self._train_cc(n_steps_cc)

        # Train curve
        if n_steps_curve > 0 and oa.optimise_curve:
            self._train_curve(n_steps_curve)

    def _train_cc(self, n_steps: int):
        """
        Train the camera coefficients and cloud points.
        """
        logger.info('----- Training the camera coefficients and cloud points -----')
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
        cam_coeffs = torch.stack(self.cam_coeffs)
        cloud_points = torch.stack(self.cloud_points)
        masks_target = torch.stack(self.masks_target)
        blur_sigmas_cloud = torch.stack(self.blur_sigmas_cloud)
        points_3d_base = torch.stack(self.points_3d_base)
        points_2d_base = torch.stack(self.points_2d_base)

        # Run the parameters through the model to get the outputs
        masks_cloud, cloud_points_scores = self.model.forward_cloud(
            cam_coeffs=cam_coeffs,
            cloud_points=cloud_points,
            masks_target=masks_target,
            blur_sigmas_cloud=blur_sigmas_cloud,
            points_3d_base=points_3d_base,
            points_2d_base=points_2d_base,
        )

        # Calculate gradients and take optimisation step
        loss, stats = self._calculate_renders_losses(
            render=masks_cloud,
            target=masks_target,
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
            'blur_sigmas_cloud/mean': blur_sigmas_cloud.mean(),
            'blur_sigmas_cloud/var': blur_sigmas_cloud.var(),
            'n_relocated': n_relocated
        }}

        self.masks_cloud = masks_cloud
        self.cloud_points_scores = cloud_points_scores

        # Update batch state
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('masks_cloud', masks_cloud[i])
            fs.set_state('cloud_points_scores', cloud_points_scores[i])
            fs.set_stats(stats)

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
        logger.info('----- Training the curve parameters -----')
        start_step = self.checkpoint.step_curve + 1
        final_step = start_step + n_steps

        # Calculate the cloud masks (this doesn't change now)
        with torch.no_grad():
            self.masks_cloud, self.cloud_points_scores = self.model.forward_cloud(
                cam_coeffs=torch.stack(self.cam_coeffs),
                cloud_points=torch.stack(self.cloud_points),
                masks_target=torch.stack(self.masks_target),
                blur_sigmas_cloud=torch.stack(self.blur_sigmas_cloud),
                points_3d_base=torch.stack(self.points_3d_base),
                points_2d_base=torch.stack(self.points_2d_base),
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

    def _train_step_curve(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train the curve parameters for a single step.
        """

        # Run the parameters through the model to get the outputs
        masks_curve, curve_points_scores = self.model.forward_curve(
            cam_coeffs=torch.stack(self.cam_coeffs),
            curve_points=torch.stack(self.curve_points),
            blur_sigmas_curve=torch.stack(self.blur_sigmas_curve),
            points_3d_base=torch.stack(self.points_3d_base),
            points_2d_base=torch.stack(self.points_2d_base),
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
            'curve_points_scores/mean': curve_points_scores.mean(),
            'curve_points_scores/var': curve_points_scores.var(),
            'blur_sigmas_curve/mean': self.blur_sigmas_curve.mean(),
            'blur_sigmas_curve/var': self.blur_sigmas_curve.var(),
            # 'max_revolutions': self.max_revolutions,
        }}

        # Update batch state
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('masks_curve', self.masks_curve[i])
            fs.set_state('curve_points_scores', self.curve_points_scores[i])
            fs.set_state('curve_points', self.curve_points[i])
            fs.set_stats(stats)

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
                l = F.mse_loss(x, y, reduction='mean')
            elif m == LOSS_KL:
                l = F.kl_div(x, y, reduction='batchmean')
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
                masks_rep = avg_pool_2d(masks_rep, oob_grad_val=0)
                target_rep = avg_pool_2d(target_rep, oob_grad_val=0)
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
        logger.info('Plotting.')
        bs = len(self.frame_batch)

        # Make initial plots for all batch elements
        if pre_step:
            if bs > 1:
                self._plot_3d_batch(show_cloud, show_curve)
            else:
                self._plot_3d(0, show_cloud, show_curve)
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and self.step % self.runtime_args.plot_every_n_steps == 0
        ):
            if bs > 1:
                self._plot_3d_batch(show_cloud, show_curve,
                                    self.optimiser_args.loss_3d_cloud_threshold if show_curve else 0)
            else:
                self._plot_3d(0, show_cloud, show_curve,
                              self.optimiser_args.loss_3d_cloud_threshold if show_curve else 0)
                # self._plot_2d(0, show_cloud, show_curve)

            for idx in range(bs):
                self._plot_2d(idx, show_cloud, show_curve)
                if plot_sigmas:
                    self._plot_sigmas(idx)
                if plot_scores:
                    self._plot_scores(idx)

    def _plot_3d_batch(
            self,
            show_cloud: bool = True,
            show_curve: bool = True,
            cloud_point_threshold: float = 0,
    ):
        """
        Make a grid of 3D scatter plots showing either or both of the cloud points and the curve points.
        """
        ws = self.optimiser_args.window_size
        n_rows = int(np.floor(np.sqrt(ws)))
        n_cols = int(np.ceil(np.sqrt(ws)))

        # Rotate the perspective on every plot
        self.plot_3d_azim += 15

        fig = plt.figure()
        gs = GridSpec(n_rows, n_cols)
        idx = 0

        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= len(self.frame_batch):
                    break
                ax = fig.add_subplot(gs[i, j], projection='3d')
                ax.view_init(azim=self.plot_3d_azim)
                ax.set_title(f'Frame #{self.frame_batch[idx].frame_num}')

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
                    # fig.colorbar(s1)

                # Curve points
                if show_curve:
                    curve_points = self.curve_points[idx]
                    scores = self.curve_points_scores[idx]
                    x, y, z = (to_numpy(curve_points[:, j]) for j in range(3))
                    # s2 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='YlGnBu', s=50, marker='x', alpha=0.9)
                    s2 = ax.scatter(x, y, z, color='black', s=75, marker='x', alpha=0.9)
                    # fig.colorbar(s2)

                idx += 1

        fig.suptitle(f'step_cc: {self.checkpoint.step_cc}. step_curve: {self.checkpoint.step_curve}.')
        fig.tight_layout()
        self._save_plot(fig, '3d_batch')

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
