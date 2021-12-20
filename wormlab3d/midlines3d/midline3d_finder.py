import os
import os
import time
from typing import Tuple, Union, Dict, List

from skimage.filters import threshold_otsu
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
import torchbnn as bnn
from simple_worm.plot3d import cla
from wormlab3d import logger, LOGS_PATH, ROOT_PATH, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import Trial, Cameras, MFCheckpoint, MFModelParameters, Checkpoint, SegmentationMasks
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_POINTS, ENCODING_MODE_MSC
from wormlab3d.midlines3d.args_finder import ModelArgs, OptimiserArgs, RuntimeArgs, SourceArgs
from wormlab3d.midlines3d.args_finder.optimiser_args import LOSS_CURVE_TARGET_MASKS
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, CAM_PARAMETER_NAMES
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel, avg_pool_2d
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.nn.args.optimiser_args import LOSS_MSE, LOSS_LOGDIFF, LOSS_KL, LOSS_BCE
from wormlab3d.nn.detector import Detector, ConvergenceDetector
from wormlab3d.nn.ema import EMA
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.toolkit.util import is_bad, to_numpy, to_dict, hash_data

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')
cmap_cloud = 'autumn_r'
cmap_curve = 'YlGnBu'
img_extension = 'png'

PRINT_KEYS = [
    'msc/loss_masks',
    # 'msc/scores_total',
    'msc/loss_dists_aunts',
    'msc/loss_dists_parents',
    'msc/loss_dists_neighbours',
    'msc/smoothness',
    'loss/sigmas_sfs_loss',
    'loss/intensities_sfs_loss',
    # 'msc/intensities',
    # 'msc/intensities_ordering',
    # 'msc/sigmas_ordering',
    # 'cloud_points_scores/mean',
    # 'n_relocated',
    # 'n_clustered',
    # 'loss/3d',
    # 'loss/render',
    # 'loss/neighbours',
    # 'curve_points_scores/mean',
    # 'loss/preangles',
    # 'loss/temporal',
    # 'loss/worm_length',
    # 'worm_length'
]

# torch.autograd.set_detect_anomaly(True)

GPU_ID = 1
EMA_RATE_CC = 0.9
EMA_RATE_CURVE = 0.99

EMA_CONVERGENCE_TAU_FAST = 10
EMA_CONVERGENCE_TAU_SLOW = 100


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
        # self.skelnet = self._init_skelnet()
        self._init_skelnet()

        # Initialise the model
        self.model, self.model_params = self._init_model()

        # Initialise convergence detector
        # self.convergence_detector = Detector(
        #     tau_init=EMA_CONVERGENCE_TAU,
        #     shape=(self.model_params.ms_curve_depth,)
        # )
        self.convergence_detector = ConvergenceDetector(
            shape=(self.model_params.ms_curve_depth,),
            tau_fast=EMA_CONVERGENCE_TAU_FAST,
            tau_slow=EMA_CONVERGENCE_TAU_SLOW,
            threshold=0.1,
            patience=25
        )
        # self.convergence_detector.configure()

        # Runtime params
        self.device = self._init_devices()

        # Load the trial and initialise trainable parameters
        self.masks = None  # todo
        self._init_trial()

        # Optimisers
        self.optimiser_cc, self.optimiser_curve = self._init_optimisers()

        # Checkpoints
        self.checkpoint = self._init_checkpoint()

        # Plotting
        self.plot_3d_azim = -60

        # Loop vars
        self.frame_num = 0
        self.active_idx = 0

        # Register exponential moving averages
        ema = EMA()
        ema.register(f'loss.{EMA_RATE_CC}/cc', decay=EMA_RATE_CC, val=1e-3)
        ema.register(f'loss.{EMA_RATE_CURVE}/curve', decay=EMA_RATE_CURVE, val=1)
        self.ema = ema

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

    def __getattr__(self, key):
        """
        Allow batched parameters to be accessed as member variables.
        """
        if key in PARAMETER_NAMES + BUFFER_NAMES:
            return [fs.get_state(key) for fs in self.frame_batch]

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
        self.skelnet_cp = cp
        return

        # Instantiate the network
        net = cp.network_params.instantiate_network()
        logger.info(f'Instantiated SkelNet with {net.get_n_params() / 1e6:.4f}M parameters.')
        logger.debug(f'----------- SkelNet --------------\n\n{net}\n\n')

        # Load the network parameter states
        path = ROOT_PATH + '/logs/scripts/midlines2d/train' \
                           f'/{cp.dataset.created:%Y%m%d_%H:%M}_{cp.dataset.id}' \
                           f'/{cp.network_params.created:%Y%m%d_%H:%M}_{cp.network_params.id}' \
                           f'/checkpoints/{cp.id}.chkpt'

        path = cp
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

        # Master state
        self.master_frame_state = self._init_frame_state(self.trial_state.frame_nums[0])

        # # Prepare batch state
        # self.frame_batch: List[FrameState] = []
        # for i in range(self.optimiser_args.window_size):
        #     fs = self._init_frame_state(self.trial_state.frame_nums[i], master_frame_state=self.master_frame_state)
        #     self.frame_batch.append(fs)

        # for fs1 in self.frame_batch:
        #     for fs2 in self.frame_batch:
        #         if fs1.frame_num != fs2.frame_num:
        #             assert not torch.allclose(fs1.images, fs2.images)

        # # Initialise parameters
        # for k in BUFFER_NAMES + PARAMETER_NAMES:
        #     # setattr(self, k, torch.stack([fs.get_state(k) for fs in self.frame_batch]))
        #     setattr(self, k, [fs.get_state(k) for fs in self.frame_batch])

    def _init_frame_state(self, frame_num: int, prev_frame_state: FrameState = None,
                          master_frame_state: FrameState = None) -> FrameState:
        """
        Load the frame
        """
        frame = self.trial.get_frame(frame_num)
        logger.info(f'Initialising frame state for frame #{frame_num} (id={frame.id}).')

        # Load segmentation masks for the next frame
        # logger.debug('Loading masks.')
        images = torch.from_numpy(np.stack(frame.images))  # '.unsqueeze(1)

        # masks = torch.zeros_like(images)
        # masks = self.skelnet.forward(images.to(self.device))

        # try:
        #     masks = SegmentationMasks.objects.get(
        #         trial=self.trial,
        #         frame=frame,
        #         checkpoint=self.model_args.skeletoniser_id
        #     )
        # except DoesNotExist:
        #     raise RuntimeError(f'Could not find segmentation masks for frame #{frame_num} (id={frame.id}).')

        # masks = self.skelnet.forward(images)
        # masks = []
        # for img in images:
        #     masks.append(self.skelnet.forward(img.to(self.device).unsqueeze(0)))
        # masks = torch.stack(masks)

        # masks = torch.from_numpy(masks.X)
        masks = images.clone().detach()

        # Threshold
        # masks = []
        # for img in frame.images:
        #     threshold = threshold_otsu(img) * 0.8
        #     masks.append(torch.from_numpy(img > threshold).to(torch.float32))
        # masks = torch.stack(masks)
        # im = np.stack(frame.images)
        # print(im.shape, im.dtype)
        # images_thresh = threshold_otsu(np.stack(frame.images))
        # print(images_thresh.shape, images_thresh.dtype)
        # masks = torch.from_numpy(images_thresh)

        masks[masks >= self.source_args.masks_target_ceil_threshold] = 1
        masks[masks < self.source_args.masks_target_ceil_threshold] = 0

        # blur_sigma = self.model_args.blur_sigma_masks_cloud_init
        # ks = int(blur_sigma * 5)
        # if ks % 2 == 0:
        #     ks += 1
        # masks = gaussian_blur(masks, kernel_size=ks, sigma=blur_sigma)

        # masks = masks / masks.sum(dim=(1, 2), keepdim=True)
        # masks = masks.squeeze(1)
        # masks = masks.clone().detach()

        # print('masks.shape', masks.shape)
        # print('masks.sum(dim=(1,2)).shape', masks.sum(dim=(1,2)).shape)
        # print('masks.sum(dim=(1,2))', masks.sum(dim=(1,2)))
        # exit()

        if self.model_params.curve_mode == ENCODING_MODE_MSC:
            masks_full_res = masks.unsqueeze(0)
            # masks_full_res = images.unsqueeze(0)

            sizes = torch.linspace(8, PREPARED_IMAGE_SIZE[0], self.model_params.ms_curve_depth).to(torch.int32)
            masks = []
            for d in range(self.model_params.ms_curve_depth):
                # image_size = max(8, int(PREPARED_IMAGE_SIZE[0] / (2**(self.model_params.ms_curve_depth - d - 1))))
                image_size = sizes[d]
                # masks_ds = avg_pool_2d(masks_rep, oob_grad_val=0)
                masks_ds = F.interpolate(masks_full_res, (image_size, image_size), mode='nearest')  #, align_corners=False)
                masks_rs = F.interpolate(masks_ds, PREPARED_IMAGE_SIZE, mode='bilinear', align_corners=False)

                blur_sigma = 1 / (2**(d+1))
                ks = int(blur_sigma * 5)
                if ks % 2 == 0:
                    ks += 1
                masks_rs = gaussian_blur(masks_rs, kernel_size=ks, sigma=blur_sigma)

                masks_rs = masks_rs.squeeze(0)
                # masks_rs = masks_rs / masks_rs.sum(dim=(1, 2), keepdim=True)
                masks.append(masks_rs)

        # Load cameras
        cameras: Cameras = frame.get_cameras()

        # Initialise frame state
        frame_state = FrameState(
            frame_num=frame_num,
            images=images,  # .squeeze(1),
            masks_target=masks,
            cameras=cameras,
            points_3d_base=torch.tensor(frame.centre_3d.point_3d),
            points_2d_base=torch.tensor(frame.centre_3d.reprojected_points_2d),
            worm_length_db=self.worm_length_db,
            model_params=self.model_params,
            optimiser_args=self.optimiser_args,
            prev_frame_state=prev_frame_state,
            master_frame_state=master_frame_state
        )
        frame_state.to(self.device)

        return frame_state

    def _init_optimisers(self) -> Tuple[Optimizer, Optimizer]:
        """
        Set up the joint cameras and cloud optimiser and the curve optimiser.
        """
        logger.info('Initialising optimisers.')
        oa = self.optimiser_args

        if 1:
            params = {
                k: self.master_frame_state.get_state(k)
                for k in PARAMETER_NAMES
            }
        else:
            params = {
                k: [fs.get_state(k) for fs in self.frame_batch]
                for k in PARAMETER_NAMES
            }

        cls_cc: Optimizer = getattr(torch.optim, oa.algorithm_cc)
        cam_params = [params[f'cam_{k}'] for k in CAM_PARAMETER_NAMES]
        optimiser_cc = cls_cc(
            [
                {'params': cam_params, 'lr': oa.lr_cam_coeffs},
                # {'params': params['cloud_points'], 'lr': oa.lr_cloud_points},
                # {'params': params['blur_sigmas_cloud'], 'lr': oa.lr_cloud_sigmas},
            ],
            # amsgrad=True,
            # centered=False,
            weight_decay=0
        )

        curve_optim_params = [
            {'params': params['curve_parameters'], 'lr': oa.lr_curve_points},
            # {'params': params['curve_length'], 'lr': oa.lr_curve_length},
            {'params': params['blur_sigmas_curve'], 'lr': oa.lr_curve_sigmas},
            {'params': params['blur_intensities_curve'], 'lr': oa.lr_curve_intensities},
            {'params': params['blur_sigmas_cameras_sfs'], 'lr': oa.lr_curve_sigmas},
            {'params': params['blur_intensities_cameras_sfs'], 'lr': oa.lr_curve_intensities},
        ]
        if self.model_params.curve_mode == ENCODING_MODE_MSC:
            curve_optim_params.append(
                {'params': cam_params, 'lr': oa.lr_cam_coeffs},
            )

        cls_curve: Optimizer = getattr(torch.optim, oa.algorithm_curve)
        optimiser_curve = cls_curve(
            params=curve_optim_params,
            # amsgrad=True,
            # centered=True,
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
            device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info('Using GPU.')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
            # for fs in self.frame_batch:
            #     fs.to(device)
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
            raise RuntimeError('Not ready!')
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
            ws = self.optimiser_args.window_size
            w2 = int(ws / 2)
            for i in range(ws):
                fs = self.frame_batch[i]
                load_fs = self._init_frame_state(checkpoint.frame_num - w2 + i)
                for k in BUFFER_NAMES:
                    fs.set_state(k, load_fs.get_state(k))
                for k in PARAMETER_NAMES:
                    fs.set_state(k, state[k][i])
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
                frame_num=self.source_args.start_frame,
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
        # self.checkpoint.save()
        # path = f'{self.logs_path}/checkpoints/{self.checkpoint.id}.chkpt'
        # params = {
        #     p: torch.stack([fs.get_state(p) for fs in self.frame_batch])
        #     for p in PARAMETER_NAMES
        # }
        # torch.save({
        #     **params,
        #     'optimiser_cc_state_dict': self.optimiser_cc.state_dict(),
        #     'optimiser_curve_state_dict': self.optimiser_curve.state_dict(),
        # }, path)

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
        self._make_plots(
            pre_step=True,
            show_cloud=False,
            show_curve=True
        )

        # Train
        w2 = int((oa.window_size - 1) / 2)
        first_frame = self.checkpoint.frame_num
        n_frames = len(self.trial_state) - first_frame + 1

        for i, frame_num in enumerate(range(first_frame, self.trial_state.frame_nums[-1])):
            logger.info(f'======== Training frame #{frame_num} ({i}/{n_frames}) ========')

            active_idx = min(i, w2)
            self.frame_num = frame_num
            self.active_idx = active_idx

            # Reset counters and train the batch
            self.checkpoint.frame_num = frame_num
            # self.checkpoint.step_cc = 0
            # self.checkpoint.step_curve = 0

            if self.step == 0:
                self.train(oa.n_steps_cc_init, oa.n_steps_curve_init, frame_num == first_frame)
            else:
                self.train(oa.n_steps_cc, oa.n_steps_curve, frame_num == first_frame)

            # Save the state
            # self.trial_state.update_frame_state(frame_num, self.frame_batch[active_idx])
            self.trial_state.update_frame_state(frame_num, self.master_frame_state)
            self.trial_state.save()

            self.convergence_detector.convergence_count.zero_()

            # Make plots
            if self.runtime_args.plot_every_n_frames > -1 \
                    and (i + 1) % self.runtime_args.plot_every_n_frames == 0:
                self._make_plots(
                    final_step=True,
                    plot_sigmas=oa.optimise_curve_sigmas,
                    plot_intensities=oa.optimise_curve_intensities,
                    plot_scores=True,
                    show_cloud=False,
                    show_curve=True
                )

            # Roll window
            next_frame = self._init_frame_state(frame_num + w2 + 1, self.master_frame_state)
            self.master_frame_state.frame_num = next_frame.frame_num
            with torch.no_grad():
                for k in BUFFER_NAMES + PARAMETER_NAMES:
                    self.master_frame_state.set_state(k, next_frame.get_state(k))

            #
            # for j in range(oa.window_size):
            #     curr_frame = self.frame_batch[j]
            #     # if j < min(i+1, w2):
            #     #     curr_frame.freeze()
            #
            #     if i >= w2:
            #         # print('j',j, w2, oa.window_size,len(self.trial_state))
            #         if j + 1 < oa.window_size:
            #             # print('copying from next in window')
            #             next_frame = self.frame_batch[j + 1]
            #         elif i + w2 < len(self.trial_state):
            #             # print('instantiating new frame')
            #             next_frame = self._init_frame_state(frame_num + w2 + 1, curr_frame)
            #             # print(f'{curr_frame.frame_num} images == {next_frame.frame_num} images')
            #             # print(f'{curr_frame.images.shape} images shape == {next_frame.images.shape} shape')
            #             # print(f'{curr_frame.images.sum()} images sum == {next_frame.images.sum()} sum')
            #         else:
            #             # print('leaving next frame untouched')
            #             continue
            #
            #         with torch.no_grad():
            #             for k in BUFFER_NAMES + PARAMETER_NAMES:
            #                 # print(k)
            #                 curr_frame.set_state(k, next_frame.get_state(k))
            #         curr_frame.frame_num = next_frame.frame_num

            # for fs1 in self.frame_batch:
            #     for fs2 in self.frame_batch:
            #         if fs1.frame_num != fs2.frame_num:
            #             if torch.allclose(fs1.images, fs2.images):
            #                 print(f'{fs1.frame_num} images == {fs2.frame_num} images')
            #                 print(f'{fs1.images.shape} images shape == {fs2.images.shape} shape')
            #                 print(f'{fs1.images.sum()} images sum == {fs2.images.sum()} sum')
            #                 raise RuntimeError('fucked it')

            # if 1:
            #     self.master_frame_state.set_state('cam_coeffs', next_frame.cam_coeffs_db)

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_frames > 0 \
                    and frame_num % self.runtime_args.checkpoint_every_n_frames == 0:
                self.save_checkpoint()

    def train(self, n_steps_cc: int, n_steps_curve: int, first_frame: bool = False):
        """
        Train a batch of frames.
        """
        oa = self.optimiser_args

        self.optimiser_cc, self.optimiser_curve = self._init_optimisers()

        # Train MSC
        if self.model_params.curve_mode == ENCODING_MODE_MSC:
            self._train_msc(n_steps_curve, first_frame)
            return

        # Train cc
        if n_steps_cc > 0 and (oa.optimise_cloud or oa.optimise_cam_coeffs):
            self._train_cc(n_steps_cc)

        #
        # self.save_checkpoint()
        # with torch.no_grad():
        #     fs = self._init_frame_state(0)
        #     cp = fs.get_state('curve_parameters')
        #     cl = fs.get_state('curve_length')
        #     self.master_frame_state.set_state('curve_parameters', cp)
        #     self.master_frame_state.set_state('curve_length', cl)

        # Train curve
        # n_steps_curve = 20000
        # print('n_steps_curve', n_steps_curve)
        if n_steps_curve > 0 and oa.optimise_curve:
            self._train_curve(n_steps_curve)
        # exit()

    def _train_cc(self, n_steps: int):
        """
        Train the camera coefficients and cloud points.
        """
        logger.info('----- Training the camera coefficients and cloud points -----')
        oa = self.optimiser_args
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
                show_curve=self.checkpoint.step_curve > 0,
                plot_sigmas=oa.optimise_cloud_sigmas,
                plot_scores=True
            )

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and self.step % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

            # If moving average is below threshold then break
            if oa.loss_target_cc is not None and self.ema[f'loss.{EMA_RATE_CC}/cc'] < oa.loss_target_cc:
                logger.info('Loss moving average reached target, breaking.')
                break

        n_steps_taken = step - start_step + 1
        self.tb_logger.add_scalar('train_steps/cc', n_steps_taken, self.checkpoint.step_cc)

    def _train_step_cc(self) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Train the cam coeffs and cloud points for a single step.
        """
        oa = self.optimiser_args
        batch_args = {
            k: torch.stack([fs.get_state(k) for fs in self.frame_batch])
            for k in
            ['cam_coeffs', 'cloud_points', 'masks_target', 'blur_sigmas_cloud', 'points_3d_base', 'points_2d_base']
        }

        # Run the parameters through the model to get the outputs
        masks_cloud, cloud_points_scores = self.model.forward_cloud(**batch_args)

        # Calculate gradients and take optimisation step
        loss, stats = self._calculate_renders_losses(
            render=masks_cloud,
            target=batch_args['masks_target'],
            metric=oa.loss_cc,
            multiscale=oa.loss_cc_multiscale,
        )
        loss_temporal = self._calculate_temporal_losses('cloud')
        loss += loss_temporal

        # Cloud points scores
        loss_cloud_scores = ((1 - cloud_points_scores)**2).mean()
        stats['loss/cloud_scores'] = loss_cloud_scores
        if oa.loss_cloud_scores > 0:
            loss += oa.loss_cloud_scores * loss_cloud_scores

        # Cloud sigmas losses
        cs = self.master_frame_state.get_state('blur_sigmas_cloud')
        loss_cloud_sigmas = torch.mean((cs - cs.mean())**2)
        stats['loss/cloud_sigmas'] = loss_cloud_sigmas
        if oa.loss_cloud_sigmas > 0:
            loss += oa.loss_cloud_sigmas * loss_cloud_sigmas

        # Neighbourhood losses
        cp = self.master_frame_state.get_state('cloud_points')
        # loss_cloud_neighbours = torch.mean(torch.norm(cp[1:] - cp[:1]))

        # Calculate pairwise distances between the cloud points to identify outliers
        dists_cloud = torch.cdist(cp, cp, p=2)
        dists_cloud, idxs_cloud = torch.sort(dists_cloud, dim=1)

        # # Take closest 1% distances from each cloud point
        # dists_cloud = dists_cloud[:, max(1, int(len(cp)*0.01))]
        # loss_cloud_neighbours = dists_cloud.mean(dim=0)

        # Weight distances exponentially so closest distances are minimised the most
        t = torch.exp(-torch.linspace(0, 1, len(cp), device=self.device) / oa.loss_cloud_neighbours_rate)
        weighted_dists = dists_cloud * t[None, :]
        loss_cloud_neighbours = weighted_dists.mean()

        stats['loss/neighbours'] = loss_cloud_neighbours
        if oa.loss_cloud_neighbours > 0:
            loss += oa.loss_cloud_neighbours * loss_cloud_neighbours

        # Camera rotation preangles loss
        pre_angles = self.master_frame_state.get_state('cam_rotation_preangles')
        preangles_loss = torch.mean((1 - torch.norm(pre_angles, dim=2))**2)
        loss += preangles_loss
        stats['loss/cam_preangles'] = preangles_loss

        if is_bad(loss):
            if is_bad(preangles_loss):
                logger.warning('Bad preangles loss, skipping parameter update.')
            elif is_bad(loss_cloud_scores):
                logger.warning('Bad loss cloud scores, skipping parameter update.')
            elif is_bad(loss_cloud_sigmas):
                logger.warning('Bad loss cloud sigmas, skipping parameter update.')
            else:
                logger.warning('Bad renders loss, skipping parameter update.')
        else:
            self.optimiser_cc.zero_grad()
            loss.backward()
            self.optimiser_cc.step()

        # Do some point relocations
        n_relocated = self._relocate_cloud_points(cloud_points_scores)

        # Do some point clustering
        n_clustered = self._cluster_cloud_points(cloud_points_scores)

        # Update camera coefficients to the batch average
        with torch.no_grad():
            # cam_coeffs_avg = torch.stack(self.cam_coeffs).mean(dim=0)
            for k in CAM_PARAMETER_NAMES:
                cam_param_avg = torch.stack(getattr(self, f'cam_{k}')).mean(dim=0)
                for fs in self.frame_batch:
                    fs.set_state(f'cam_{k}', cam_param_avg)

        stats = {**stats, **{
            'loss/temporal': loss_temporal.item(),
            'cloud_points_scores/mean': cloud_points_scores.mean(),
            'cloud_points_scores/var': cloud_points_scores.var(),
            'blur_sigmas_cloud/mean': batch_args['blur_sigmas_cloud'].mean(),
            'blur_sigmas_cloud/var': batch_args['blur_sigmas_cloud'].var(),
            'n_relocated': n_relocated,
            'n_clustered': n_clustered,
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
        total_relocated = 0
        for i, fs in enumerate(self.frame_batch):
            if fs.is_frozen:
                continue

            # Check which points scored below threshold
            n_scored_too_low = (cloud_points_scores[i] <= self.optimiser_args.relocate_score_threshold).sum()
            n_to_relocate = min(max_turnover, n_scored_too_low)
            total_relocated += n_to_relocate

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
                        # std=0.001
                    )
                    # self.blur_sigmas_cloud[i][src_idxs] = 0.01

            # Random relocations
            n_random = self.optimiser_args.relocate_points_randomly
            if n_random > 0:
                scored_idxs = torch.argsort(cloud_points_scores[i], descending=True)

                # Take from random idxs and drop at high scoring idxs
                random_idxs = torch.randperm(n_random)
                src_idxs = scored_idxs[random_idxs]
                dest_idxs = scored_idxs[:n_random]

                with torch.no_grad():
                    # Relocate points
                    self.cloud_points[i][src_idxs] = torch.normal(
                        mean=self.cloud_points[i][dest_idxs],
                        std=self.blur_sigmas_cloud[i][dest_idxs][:, None].expand_as(self.cloud_points[i][dest_idxs])
                        # std=0.001
                    )
                    # self.blur_sigmas_cloud[i][src_idxs] = 0.01

                total_relocated += n_random

        return total_relocated

    def _cluster_cloud_points(self, cloud_points_scores: torch.Tensor) -> int:
        """
        Every so often, any outlier points should clone into a cluster.
        """
        if not (self.optimiser_args.cluster_every_n_steps > 0
                and (self.checkpoint.step_cc + 1) % self.optimiser_args.cluster_every_n_steps == 0):
            return 0

        # Default the maximum number of points to relocate to 1% of the total points if not defined
        max_turnover = self.optimiser_args.cluster_max_points
        if max_turnover is None:
            max_turnover = int(self.model_params.n_cloud_points * 0.01)

        # Default the neighbourhood size to 1% of the total points if not defined
        nhd_size = self.optimiser_args.cluster_nhd_size
        if nhd_size is None:
            nhd_size = int(self.model_params.n_cloud_points * 0.01)

        # Iterate over batch
        total_relocated = 0
        for i, fs in enumerate(self.frame_batch):
            if fs.is_frozen:
                continue

            # Get average distance to nearest neighbours
            cps = self.cloud_points[i]
            dists = torch.cdist(cps, cps)
            dists_sorted, _ = torch.sort(dists, dim=1)
            nhd_mean_dist = dists_sorted[:, :nhd_size].mean(dim=1)

            # Check which points had densities below the threshold
            n_outliers = (nhd_mean_dist >= self.optimiser_args.cluster_density_threshold).sum()
            n_to_relocate = min(max_turnover, n_outliers)
            total_relocated += n_to_relocate

            if n_to_relocate > 0:
                # Sort neighbourhood densities
                densities_sorted, point_density_ranked_idxs = torch.sort(nhd_mean_dist, descending=False)
                src_idxs = point_density_ranked_idxs[-n_to_relocate:]
                dest_idxs = point_density_ranked_idxs[:n_to_relocate]

                # Randomise destinations
                random_idxs = torch.randperm(n_to_relocate)
                dest_idxs = dest_idxs[random_idxs]

                with torch.no_grad():
                    # Relocate points
                    self.cloud_points[i][src_idxs] = torch.normal(
                        mean=self.cloud_points[i][dest_idxs],
                        std=self.blur_sigmas_cloud[i][dest_idxs][:, None].expand_as(self.cloud_points[i][dest_idxs])
                        # std=0.001
                    )
                    # self.blur_sigmas_cloud[i][src_idxs] = 0.01

        return total_relocated

    def _train_curve(self, n_steps: int):
        """
        Train the curve parameters.
        """
        logger.info('----- Training the curve parameters -----')
        oa = self.optimiser_args
        start_step = self.checkpoint.step_curve + 1
        final_step = start_step + n_steps

        # Calculate the cloud masks (this doesn't change now)
        with torch.no_grad():
            cloud_points = torch.stack(self.cloud_points)
            self.masks_cloud, self.cloud_points_scores = self.model.forward_cloud(
                cam_coeffs=torch.stack(self.cam_coeffs),
                cloud_points=cloud_points,
                masks_target=torch.stack(self.masks_target),
                blur_sigmas_cloud=torch.stack(self.blur_sigmas_cloud),
                points_3d_base=torch.stack(self.points_3d_base),
                points_2d_base=torch.stack(self.points_2d_base),
            )

            # Calculate the initial curve points
            self._update_curve_points()
            # curve_points = self._get_curve_coordinates()
            # for i, fs in enumerate(self.frame_batch):
            #     fs.set_state('curve_points', curve_points[i])

            # Determine target
            if oa.loss_curve_target == LOSS_CURVE_TARGET_MASKS:
                self.curve_target = torch.stack(self.masks_target)
            elif oa.loss_curve_3d:
                T = oa.loss_3d_cloud_threshold
                if T > 0:
                    # todo - can't stack different shapes!
                    # print('self.cloud_points_scores > T', (self.cloud_points_scores > T).shape)
                    cts = []
                    for i in range(len(cloud_points)):
                        cts.append(cloud_points[i][self.cloud_points_scores[i] > T])
                    # cts = torch.stack(cts)
                    self.curve_target = cts
                    # self.curve_target = cloud_points[self.cloud_points_scores > T]
                else:
                    threshold = 0.01
                    centroids = []

                    # for i in range(len(self.frame_batch)):
                    #     # Reduce the points so that they are not too clustered together
                    #     points = cloud_points[i]
                    #     clusters = {}
                    #     centroids_i = torch.zeros((0,3), device=self.device)
                    #
                    #     while len(points) > 0:
                    #         if len(centroids_i) == 0:
                    #             p = points[0]
                    #             print('Adding first centroid at ', p)
                    #             centroids_i = torch.stack([*centroids_i, p])
                    #             clusters[0] = [p]
                    #             points = points[1:]
                    #             continue
                    #
                    #         # Check distances to centroids
                    #         dists = torch.cdist(points, centroids_i, p=2)
                    #         # dists, _ = dists.sort(dim=1)
                    #         idxs = (dists < threshold).nonzero()
                    #
                    #         if len(idxs):
                    #             # If any points are close enough to any centroids then allocate them there
                    #             for idx in idxs:
                    #                 point_idx, centroid_idx = idx
                    #                 clusters[int(centroid_idx)].append(points[point_idx])
                    #
                    #             # Remove points
                    #             remove_idxs = idxs[:, 0]
                    #             mask = torch.ones(len(points), dtype=torch.bool)
                    #             mask[remove_idxs] = 0
                    #
                    #             all_idxs = torch.arange(len(points))
                    #             # print('all_idxs == remove_idxs')
                    #             # print('torch.where(all_idxs == remove_idxs,torch.tensor(1),torch.tensor(0))')
                    #             # w = torch.where(all_idxs == remove_idxs,torch.tensor(1),torch.tensor(0))
                    #             # print(w)
                    #             # print(all_idxs == remove_idxs)
                    #             # print('(all_idxs == remove_idxs).nonzero()')
                    #             # print((all_idxs == remove_idxs).nonzero())
                    #             # print('1 - (all_idxs == remove_idxs).nonzero()')
                    #             # print(1 - (all_idxs == remove_idxs).nonzero())
                    #             # keep_idxs = all_idxs[1 - (all_idxs == remove_idxs).nonzero()]
                    #             keep_idxs = all_idxs[mask]
                    #             # print('keep_idxs')
                    #             # print(keep_idxs)
                    #             points = points[keep_idxs]
                    #
                    #             # Update centroids
                    #             for ci in range(len(centroids_i)):
                    #                 centroids_i[ci] = torch.mean(torch.stack(clusters[ci]), dim=0)
                    #
                    #         else:
                    #             # Nothing close enough so add a new centroid
                    #             p = points[0]
                    #             centroids_i = torch.stack([*centroids_i, p])
                    #             clusters[len(clusters)] = [p]
                    #             points = points[1:]
                    #
                    #     print('len(clusters)', len(clusters))
                    #
                    #     centroids.append(centroids_i)
                    #
                    # self.curve_target = centroids
                    self.curve_target = cloud_points

                # print('self.cloud_points_scores.shape', self.cloud_points_scores.shape)
                # print('cloud_points.shape', cloud_points.shape)
                # print(self.curve_target.shape)
                # exit()
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

            # If moving average is below threshold then break
            if oa.loss_target_curve is not None and self.ema[f'loss.{EMA_RATE_CURVE}/curve'] < oa.loss_target_curve:
                logger.info('Loss moving average reached target, breaking.')
                break

        n_steps_taken = step - start_step + 1
        self.tb_logger.add_scalar('train_steps/curve', n_steps_taken, self.checkpoint.step_curve)

    def _train_step_curve(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train the curve parameters for a single step.
        """
        # curve_points = self._get_curve_coordinates()
        # assert curve_points.requires_grad
        # self._update_curve_points()
        cc = self._get_curve_coordinates()

        # Run the parameters through the model to get the outputs
        masks_curve, curve_points_scores = self.model.forward_curve(
            cam_coeffs=torch.stack(self.cam_coeffs),
            # curve_points=torch.stack(self.curve_points),
            # curve_points=self._get_curve_coordinates(),
            # curve_points=curve_points,
            curve_points=cc,
            blur_sigmas_curve=torch.stack(self.blur_sigmas_curve),
            points_3d_base=torch.stack(self.points_3d_base),
            points_2d_base=torch.stack(self.points_2d_base),
        )

        loss_render = torch.tensor(0., device=self.device)
        loss_3d = torch.tensor(0., device=self.device)
        stats_render = {}
        stats_3d = {}

        if self.optimiser_args.loss_curve_3d:
            loss_3d, stats_3d = self._calculate_3d_losses(cc)
        else:
            loss_render, stats_render = self._calculate_renders_losses(
                render=masks_curve,
                target=self.curve_target,
                metric=self.optimiser_args.loss_curve,
                multiscale=self.optimiser_args.loss_curve_multiscale,
            )

        loss_curve, stats_curve = self._calculate_curve_losses()
        loss_temporal = self._calculate_temporal_losses('curve', cc)

        loss = loss_3d + loss_render + loss_curve + loss_temporal

        self.optimiser_curve.zero_grad()
        loss.backward()
        self.optimiser_curve.step()

        # self.masks_curve = masks_curve
        # self.curve_points_scores = curve_points_scores

        stats = {**stats_3d, **stats_render, **stats_curve, **{
            'loss/3d': loss_3d.item(),
            'loss/render': loss_render.item(),
            'loss/curve': loss_curve.item(),
            'loss/temporal': loss_temporal.item(),
            'worm_length': torch.stack(self.curve_length).mean(),
            'curve_points_scores/mean': curve_points_scores.mean(),
            'curve_points_scores/var': curve_points_scores.var(),
            'blur_sigmas_curve/mean': torch.stack(self.blur_sigmas_curve).mean(),
            'blur_sigmas_curve/var': torch.stack(self.blur_sigmas_curve).var(),
            # 'max_revolutions': self.max_revolutions,
        }}

        # Update batch state
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('masks_curve', masks_curve[i])
            fs.set_state('curve_points_scores', curve_points_scores[i])
            # fs.set_state('curve_points', curve_points[i])
            fs.set_stats(stats)

        return loss, stats

    def _calculate_3d_losses(self, cc_batch):
        """
        Calculate the how well the curve fits the control points.
        """
        # cc = torch.stack(self.curve_points)

        # # Calculate pairwise distances between the cloud points and the curve points
        # dists = torch.cdist(cc, ct, p=2)
        # # dists = torch.cdist(cc, self.curve_target, p=2)
        # min_points = 3  # todo: investigate
        #
        # # Only consider distances within a segment-length of each point
        # segment_lengths = torch.norm(cc[:, 1:] - cc[:, :-1], dim=-1)
        # max_distance = segment_lengths.mean() * 2  # todo: investigate
        #
        # # Sort the distances relative to the each
        # dists_curve_control, _ = dists.sort(dim=2)
        # dists_control_curve, _ = dists.sort(dim=1)
        #
        # # Set any distances greater than the max to 0
        # dists_curve_control[dists_curve_control > max_distance] = 0
        # # dists_control_curve[dists_control_curve > max_distance] = 0
        #
        # # Only consider the nearest 10% of the filtered cloud points to each curve point within distance cutoff
        # n_cloud_points = max(min_points, int(self.curve_target.shape[1] * 0.1))
        # dists_curve_control_filtered = dists_curve_control[:, :, :n_cloud_points]
        #
        # # Only consider the nearest 10% curve points to each cloud point
        # n_curve_points = max(min_points, int(self.model_params.n_curve_points * 0.1))
        # dists_control_curve_filtered = dists_control_curve[:, :n_curve_points]
        #
        # # Loss pulls in both directions
        # # loss_curve_control = 0  #torch.mean(dists_curve_control_filtered**2)
        # # loss_control_curve = torch.mean(dists_control_curve_filtered**2)
        # loss_curve_control = torch.sum(dists_curve_control_filtered)
        # loss_control_curve = torch.sum(dists_control_curve_filtered)
        # loss = loss_curve_control + loss_control_curve
        #
        # # Add centre-point pull  # todo: is this necessary / does it help?
        # loss_cp = torch.mean((cc.mean(dim=1) - self.curve_target.mean(dim=1))**2)
        # loss = loss + loss_cp

        loss_all = 0
        M = self.model_params.n_cloud_points
        N = self.model_params.n_curve_points

        for i in range(len(self.frame_batch)):
            cc = cc_batch[i]
            ct = self.curve_target[i]

            # # Calculate pairwise distances between the cloud points to identify outliers
            # dists_cloud = torch.cdist(ct, ct, p=2)
            # # print('dists_cloud')
            # # print(dists_cloud.shape)
            # # print(dists_cloud)
            # dists_cloud, idxs_cloud = torch.sort(dists_cloud, dim=1)
            # # print(dists_cloud)
            #
            # # Take closest 5% distances from each cloud point
            # dists_cloud = dists_cloud[:, max(1, int(M*0.05))]
            # # print(dists_cloud.shape)
            #
            # # Identify cloud points with large closest distances
            # avg_cloud_dists = dists_cloud.mean(dim=0)
            # # print('avg_cloud_dists.shape', avg_cloud_dists.shape)
            # mean_avg_cloud_dists = avg_cloud_dists.mean()
            # # print('mean_avg_cloud_dist', mean_avg_cloud_dists)
            # std_avg_cloud_dists = avg_cloud_dists.std()
            # # print('std_avg_cloud_dists', std_avg_cloud_dists)
            # inlier_idxs = idxs_cloud[
            #     (avg_cloud_dists > mean_avg_cloud_dists - 2 *std_avg_cloud_dists) & (avg_cloud_dists < mean_avg_cloud_dists + 2*std_avg_cloud_dists)
            # ]
            # # print('inlier_idxs')
            # # print(inlier_idxs.shape)
            # # print(inlier_idxs)
            #
            # # Remove the outliers
            # cc = cc[inlier_idxs]
            # # print('cc')
            # # print(cc.shape)
            # # print(cc)
            # # exit()

            # Calculate pairwise distances between the cloud points and the curve points
            dists = torch.cdist(cc, ct, p=2)

            # Only consider distances within a segment-length of each point
            # segment_lengths = torch.norm(cc[1:] - cc[:-1], dim=-1)
            # max_distance = segment_lengths.mean() * 2  # todo: investigate

            # Sort the distances relative to the each
            dists_curve_control, idxs_curve_control = dists.sort(dim=1)
            dists_control_curve, idxs_control_curve = dists.sort(dim=0)

            # Each curve point should be only pulled by max one control point
            # - the furthest away of all the points pulling on it

            # so that means masking the distances where idxs are duplicated...
            loss_control_curve = 0
            closest_curve_idxs_to_controls = idxs_control_curve[0]
            closest_dists_to_controls = dists_control_curve[0]
            closest_control_idxs_to_curve = idxs_curve_control[0]
            closest_dists_to_curve = dists_curve_control[0]

            # print('closest_dists_to_controls')
            # print(closest_dists_to_controls.shape)
            # print(closest_dists_to_controls)
            # forces_control_curve = torch.amax(closest_dists_to_controls, dim=1)
            # forces_curve_control = closest_dists_to_curve
            # forces = forces_curve_control.clone()
            # forces[closest_curve_idxs_to_controls] = forces_control_curve[closest_curve_idxs_to_controls]
            # forces = (forces**2).sum()

            for curve_idx in range(N):
                curve_forces_from_controls_at_idx = closest_dists_to_controls[
                    closest_curve_idxs_to_controls == curve_idx]

                # If the curve is being pulled by some controls at this point then just move towards the farthest
                if len(curve_forces_from_controls_at_idx) > 0:

                    if curve_idx in [0, N - 1]:
                        w = 10
                    elif curve_idx in [1, N - 2]:
                        w = 5
                    else:
                        w = 1

                    loss_control_curve += w * torch.amax(curve_forces_from_controls_at_idx)**2

                # # If curve point is NOT being pulled on, then move the curve point towards it's nearest control point
                # else:
                #     closest_dists_to_curve[closest_control_idxs_to_curve]
                #     loss_control_curve += closest_dists_to_curve[curve_idx]**2 * 1e-1

                # If curve point is NOT being pulled on, then move the curve point towards it's neighbouring vertex on the central-side
                else:
                    # Head-end
                    if curve_idx < N / 2:
                        neighbour_idx = curve_idx + 1
                    else:
                        neighbour_idx = curve_idx - 1
                    loss_control_curve += torch.norm(cc[curve_idx] - cc[neighbour_idx].detach())**2

            # Set any distances greater than the max to 0
            # dists_curve_control[dists_curve_control > max_distance] = 0
            # dists_control_curve[dists_control_curve > max_distance] = 0

            # Only consider the nearest 10% of the filtered cloud points to each curve point within distance cutoff
            # n_cloud_points = max(min_points, int(ct.shape[0] * 0.1))
            # dists_curve_control_filtered = dists_curve_control[:, :n_cloud_points]

            # Only consider the nearest 10% curve points to each cloud point
            # n_curve_points = max(min_points, int(self.model_params.n_curve_points * 0.1))
            # dists_control_curve_filtered = dists_control_curve[:n_curve_points]
            # dists_control_curve_filtered = dists_control_curve[0]

            # dists_control_curve = dists_control_curve[:5]
            # random_idxs = torch.randperm(5)
            # dists_control_curve_filtered = dists_control_curve[random_idxs]

            # Loss pulls in both directions
            loss_curve_control = 0
            # loss_curve_control = torch.sum(dists_curve_control[0]**2) * 1e-5
            # loss_control_curve = torch.mean(dists_control_curve_filtered**2)
            # loss_curve_control = torch.sum(dists_curve_control_filtered)
            # loss_control_curve = torch.sum(dists_control_curve_filtered**2)
            # loss_control_curve = torch.sum(dists_control_curve_filtered)
            loss = loss_curve_control + loss_control_curve

            # # Add centre-point pull  # todo: is this necessary / does it help?
            # loss_cp = torch.mean((cc.mean(dim=0) - ct.mean(dim=0))**2)
            # loss = loss + loss_cp

            loss_all += loss

        stats = {
            'loss/3d_curve_control': loss_curve_control,
            'loss/3d_control_curve': loss_control_curve,
            # 'loss/loss_cp': loss_cp,
            '3d_dists/curve_control/mean': dists.mean(),
            '3d_dist/curve_control/var': dists.var(),
        }

        return loss_all, stats

    def _calculate_curve_losses(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Losses for curve length and curvatures.
        """
        loss = 0
        stats = {}
        cp = torch.stack(self.curve_parameters)

        if self.model_params.curve_mode == ENCODING_MODE_DELTA_ANGLES:
            pre_angles = cp[:, 3:7]
            theta0_reg_loss = torch.mean((1 - torch.norm(pre_angles[:, :2], dim=1))**2)
            phi0_reg_loss = torch.mean((1 - torch.norm(pre_angles[:, 2:4], dim=1))**2)
            preangles_loss = theta0_reg_loss + phi0_reg_loss
            loss += preangles_loss
            stats['loss/preangles'] = preangles_loss

            # # Add worm-length loss - bigger worms are preferred
            # wl_loss = -torch.log(1 + self.curve_length.mean())
            # loss += wl_loss
            # stats['loss/worm_length'] = wl_loss

            # Add curve-length loss - regularise to be close to the db value
            cl = torch.stack(self.curve_length)
            cl_loss = torch.mean((cl - self.worm_length_db)**2)  # * 1e-1
            stats['loss/curve_length'] = cl_loss
            if self.optimiser_args.loss_curve_length > 0:
                loss += cl_loss * self.optimiser_args.loss_curve_length

            # Add curvature regularisation
            delta_angles = cp[:, 7:]
            curvature = torch.sum(delta_angles**2)
            stats['loss/curvature'] = curvature
            if self.optimiser_args.loss_curve_curvature > 0:
                loss += curvature * self.optimiser_args.loss_curve_curvature

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

        def loss_(m, x, y, reduce=True):
            if m == LOSS_MSE:
                # l = F.mse_loss(x, y, reduction='mean')
                l = F.mse_loss(x, y, reduction='mean' if reduce else 'none')
            elif m == LOSS_KL:
                l = F.kl_div(x, y, reduction='batchmean' if reduce else 'none')
            elif m == LOSS_LOGDIFF:
                l = torch.sum((torch.log(1 + x) - torch.log(1 + y))**2)
            elif m == LOSS_BCE:
                l = F.binary_cross_entropy(x, y, reduction='mean' if reduce else 'none')
            return l

        stats = {}

        if multiscale:
            # Multiscale loss
            loss = 0
            masks_rep = render.clone()
            target_rep = target.clone()
            k = 1
            while masks_rep.shape[-1] > 1:
                l = loss_(metric, masks_rep, target_rep)
                loss += l
                # l = loss_(metric, masks_rep, target_rep, reduce=False)
                #
                # # Only try to improve worst
                # cam_losses = l.mean(dim=(2,3))  # shape=(bs, 3)
                # worst_cam_loss = cam_losses.amax(dim=1).mean()
                # loss += worst_cam_loss

                stats[f'loss/{metric}_{masks_rep.shape[-1]}'] = l.item()

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

    def _calculate_temporal_losses(self, cloud_or_curve: str, points=None) -> float:
        assert cloud_or_curve in ['cloud', 'curve']

        if cloud_or_curve == 'cloud':
            # (bs, n_cloud_points, 3)
            params = torch.stack(self.cloud_points)
            w = self.optimiser_args.loss_cloud_temporal_smoothing

        else:
            # (bs, n_curve_points, 3)
            if points is not None:
                params = points
            else:
                params = torch.stack(self.curve_points)
            w = self.optimiser_args.loss_curve_temporal_smoothing

        # print('temporal losses')
        # print(params.shape)

        # norm2 = torch.sum((params[1:] - params[:-1])**2, dim=-1)
        # print(norm2.shape)
        # print(norm2.sum())
        # print(w)

        # return w * norm2.mean()

        # Difference from means
        norm2 = torch.mean(torch.sum((params - params.mean(dim=0, keepdim=True))**2, dim=-1))

        return w * norm2

        # avg_variance = params.var(dim=0).mean()
        # return w * avg_variance

    def _train_msc(self, n_steps: int, first_frame: bool = False):
        """
        Train the camera coefficients and multiscale curve.
        """
        logger.info('----- Training the camera coefficients and multiscale curve -----')
        oa = self.optimiser_args
        start_step = self.checkpoint.step_curve + 1
        final_step = start_step + n_steps

        # Train the cam coeffs and multiscale curve
        for step in range(start_step, final_step):
            loss, losses_masks, stats = self._train_step_msc()

            # Update step and checkpoint loss
            self.checkpoint.step_curve += 1
            self.checkpoint.loss_curve = loss.item()
            self.checkpoint.metrics_cc = stats

            # Log
            self._log_progress(
                cc_or_curve='curve',
                step=step,
                final_step=final_step,
                loss=loss,
                stats=stats
            )

            # Make plots
            self._make_plots(
                show_cloud=False,
                show_curve=True,
                plot_sigmas=oa.optimise_curve_sigmas,
                plot_intensities=oa.optimise_curve_intensities,
                plot_scores=True
            )

            # Checkpoint
            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and self.step % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

            # Update convergence detector
            losses_masks = torch.tensor(
                [stats[f'msc/masks/{oa.loss_curve}/{d}'] for d in range(self.model_params.ms_curve_depth)],
                device=self.device
            )
            self.convergence_detector.forward(losses_masks)

            # When all of the mask losses have converged, break
            if not first_frame and self.convergence_detector.converged.all():
                break

        n_steps_taken = step - start_step + 1
        self.tb_logger.add_scalar('train_steps/msc', n_steps_taken, self.checkpoint.step_curve)

    def _train_step_msc(self) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Train the cam coeffs and multiscale curve for a single step.
        """
        oa = self.optimiser_args
        D = self.model_params.ms_curve_depth

        batch_args ={}
        for k in ['cam_coeffs', 'points_3d_base', 'points_2d_base', 'blur_sigmas_cameras_sfs', 'blur_intensities_cameras_sfs']:
            # batch_args[k] = torch.stack([f.get_state(k) for f in self.frame_batch])
            batch_args[k] = self.master_frame_state.get_state(k).unsqueeze(0)
        for k in ['curve_parameters', 'masks_target', 'blur_sigmas_curve', 'blur_intensities_curve']:
            batch_args[k] = [
                self.master_frame_state.get_state(k)[d].unsqueeze(0)
                # torch.stack([f.get_state(k)[d] for f in self.frame_batch])
                for d in range(D)
            ]

        stats = {}
        loss = 0
        losses_masks = []
        losses_dists_neighbours = []
        losses_dists_parent = []
        losses_dists_aunts = []
        losses_scores = []
        losses_sigmas = []
        losses_intensities = []
        losses_smoothnesses = []
        # targets = []
        masks = []
        points_2d = []
        points_scores = []
        curve_points_smoothed=[]
        sigmas_smoothed=[]
        intensities_smoothed=[]

        sigmas_sfs = batch_args['blur_sigmas_cameras_sfs']
        intensities_sfs = batch_args['blur_intensities_cameras_sfs']

        # Run the parameters through the model at each scale to get the outputs
        for d in range(D):
            curve_points_d = batch_args['curve_parameters'][d]  # [max(0,2**(d-1)):2**d]
            masks_target_d = batch_args['masks_target'][d]
            sigmas_d = batch_args['blur_sigmas_curve'][d]
            intensities_d = batch_args['blur_intensities_curve'][d]

            # Smooth things
            if d > 1:
                # print(d)
                # print('curve_points_d', curve_points_d.shape)
                ks = int(d/2)*2+1
                pad_size = int(ks /2)

                cp = torch.cat([
                    torch.repeat_interleave(curve_points_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    curve_points_d,
                    torch.repeat_interleave(curve_points_d[:, -1].unsqueeze(1), pad_size, dim=1)
                ], axis=1)
                cp = cp.permute(0,2,1)
                # cp = curve_points_d.permute(0,2,1)
                # print('cp', cp.shape)
                cps = F.avg_pool1d(input=cp, kernel_size=ks, stride=1, padding=0)
                # print('ag', ag.shape)
                curve_points_d = cps.permute(0,2,1)
                # print('ag', ag.shape)

                sigs = torch.cat([
                    sigmas_d[:, 1:pad_size+1].flip(dims=(1,)),
                    # torch.repeat_interleave(sigmas_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    sigmas_d,
                    sigmas_d[:, -pad_size-1:-1].flip(dims=(1,)),
                    # torch.repeat_interleave(sigmas_d[:, -1].unsqueeze(1), pad_size, dim=1)
                ], dim=1)
                sigs = sigs[:, None, :]
                sigs = F.avg_pool1d(input=sigs, kernel_size=ks, stride=1, padding=0)
                sigmas_d = sigs.squeeze(1)

                ints = torch.cat([
                    intensities_d[:, 1:pad_size+1].flip(dims=(1,)),
                    # torch.repeat_interleave(intensities_d[:, 0].unsqueeze(1), pad_size, dim=1),
                    intensities_d,
                    intensities_d[:, -pad_size-1:-1].flip(dims=(1,)),
                    # torch.repeat_interleave(intensities_d[:, -1].unsqueeze(1), pad_size, dim=1)
                ], dim=1)
                ints = ints[:, None, :]
                ints = F.avg_pool1d(input=ints, kernel_size=ks, stride=1, padding=0)
                intensities_d = ints.squeeze(1)

            # print('sigmas_d.shape', sigmas_d.shape)
            # print('sigmas_sfs.shape', sigmas_sfs.shape)

            masks_d, points_2d_d, points_scores_d = self.model.forward_cloud(
                cam_coeffs=batch_args['cam_coeffs'],
                # cloud_points=curve_points_d,
                cloud_points=curve_points_d,
                masks_target=masks_target_d,
                blur_sigmas_cloud=sigmas_d,
                blur_intensities_cloud=intensities_d,
                blur_sigmas_cameras_sfs=sigmas_sfs,
                blur_intensities_cameras_sfs=intensities_sfs,
                points_3d_base=batch_args['points_3d_base'],
                points_2d_base=batch_args['points_2d_base'],
            )
            masks.append(masks_d)
            points_2d.append(points_2d_d)
            points_scores.append(points_scores_d)
            curve_points_smoothed.append(curve_points_d)
            sigmas_smoothed.append(sigmas_d)
            intensities_smoothed.append(intensities_d)

            # Calculate losses
            # print('\n')
            # print('masks_d', masks_d.shape, masks_d.max(), masks_d.min(), masks_d.sum())
            # print('masks_target_d', masks_target_d.shape, masks_target_d.max(), masks_target_d.min(), masks_target_d.sum())
            # target = torch.bernoulli(masks_target_d)  # * masks_d.sum()/masks_target_d.sum())
            # print('target',  target.shape, target.dtype, target.max())
            # exit()
            # print('target', target.shape, target.max(), target.min(), target.sum())
            # x = torch.normal(mean=torch.zeros_like(masks_target_d), std=1 / (2**(d+1)))

        targets = {}

        for d in range(D - 1, -1, -1):
            curve_points_d = batch_args['curve_parameters'][d]  # [max(0,2**(d-1)):2**d]
            masks_target_d = batch_args['masks_target'][d]
            sigmas_d = batch_args['blur_sigmas_curve'][d]
            intensities_d = batch_args['blur_intensities_curve'][d]
            masks_d = masks[d]
            points_scores_d = points_scores[d]
            curve_points_smoothed_d = curve_points_smoothed[d]
            sigmas_smoothed_d = sigmas_smoothed[d]
            intensities_smoothed_d = intensities_smoothed[d]

            if 1:
                target = masks_target_d
                # if d == D - 1:
                #     residual_next = torch.zeros_like(target)
                # else:
                #     # residual_next_d = torch.clamp(batch_args['masks_target'][d+1] - masks[d+1].detach(), min=0)
                #     residual_next_d = torch.clamp(batch_args['masks_target'][d+1] - masks[d+1], min=0)
                #     # residual_next_d = masks[d+1]
                #     if d < D - 2:
                #         residual_next = 0.5 * residual_next + 0.5 * residual_next_d
                #     else:
                #         residual_next = residual_next_d
                res_eps = 0.01

                residual_next = torch.zeros_like(target)
                masks_next = torch.zeros_like(target)
                # print('\nnexts', d)
                for d2 in range(d+1, D):
                    sf = 1/(2**(d2-d))
                    res_d2 = torch.clamp(batch_args['masks_target'][d2] - masks[d2], min=0)
                    res_d2[res_d2 < res_eps] = 0
                    masks_next += sf * masks[d2]
                    residual_next += sf * res_d2
                    # print(d2, 1/(2**(d2-d)), res_d2.max().item(), res_d2.min().item(), masks[d2].max().item(), masks[d2].min().item())

                # print('\nprevs', d)
                residual_prev = torch.zeros_like(target)
                masks_prev = torch.zeros_like(target)
                for d2 in range(d-1, -1, -1):
                    sf = 1/(2**(d-d2))
                    res_d2 = torch.clamp(batch_args['masks_target'][d2] - masks[d2], min=0)
                    res_d2[res_d2 < res_eps] = 0
                    masks_prev += sf * masks[d2]
                    residual_prev += sf * res_d2
                    # print(d2, 1/(2**(d2-d)), res_d2.max().item(), res_d2.min().item(), masks[d2].max().item(), masks[d2].min().item())
                    # residual_prev += 1/(2**(d-d2)) * torch.clamp(batch_args['masks_target'][d2] - masks[d2], min=0)

                # print('\nsummary')
                # print(d)
                # print(masks_prev.max().item(), masks_prev.min().item(), residual_prev.max().item(), residual_prev.min().item())
                # print(masks_next.max().item(), masks_next.min().item(), residual_next.max().item(), residual_next.min().item())

                # target += -residual_prev #+ 0.5 * residual_next
                # target = target - residual_next - residual_prev
                # target = target - 0.5 * masks_next  # - residual_prev
                # target = target - 0.5 * masks_next + 0.5 * residual_prev
                # target = target - 0.5 * masks_prev + residual_next
                target = target + residual_next
                # target = target + residual_next - 0.5*residual_prev
                # target = target + residual_next - residual_prev
                # target = target / target.max()
                target = torch.clamp(target, min=0, max=1)
                target = target.detach()
                # target[target < 0.05] = 0
                # target = torch.bernoulli(target)  #.detach())

            elif 0:
                target = torch.bernoulli(masks_target_d)
            elif 0:
                if 0 and d > 0:
                    residual_prev = torch.clamp(masks_target_d - masks[d-1].detach(), min=0)
                    # residual_prev = torch.clamp(batch_args['masks_target'][d-1] - masks[d-1].detach(), min=0)
                else:
                    residual_prev = torch.zeros_like(masks_target_d)
                if d < D - 1:
                    # residual_next = torch.clamp(masks_target_d - masks[d+1].detach(), min=0)
                    residual_next = torch.clamp(batch_args['masks_target'][d+1] - masks[d+1].detach(), min=0)
                else:
                    residual_next = torch.zeros_like(masks_target_d)

                # target = masks_target_d + 0.5 * residual_prev + 0.5 * residual_next
                target = masks_target_d + residual_next
                target = target.clamp(min=0, max=1)
                target = torch.bernoulli(target)

            elif 0 and d < self.model_params.ms_curve_depth - 1:
                # residual = masks_target_d - masks[-1].detach()
                residual = masks_target_d - masks[d+1].detach()
                residual = residual.clamp(min=0)
                # print('residual', residual.shape, residual.max(), residual.min(), residual.sum())
                # residual = residual / residual.sum(dim=(2,3), keepdim=True)
                residual = residual / torch.amax(residual, dim=(2,3), keepdim=True)
                # print('residual', residual.shape, residual.max(), residual.min(), residual.sum())
                # target = residual
                target = torch.bernoulli(torch.clamp(masks_target_d + residual, max=1, min=0))
            else:
                target = masks_target_d



            # Add some mask to the target so only give errors where the projection is
            # target[masks_d < 0.01] = 0


            targets[d] = target

            loss_masks_d, stats_d = self._calculate_renders_losses(
                render=masks_d,
                # target=masks_target_d,
                target=target,
                metric=oa.loss_curve,
                multiscale=oa.loss_curve_multiscale,
                calculate_all=False
            )
            stats[f'msc/masks/{oa.loss_curve}/{d}'] = loss_masks_d

            # # Only put mask losses in when the parent's mask losses have converged
            # if d > 0:
            #     # print('d', d, self.convergence_detector.detecting[d-1])
            #     if self.convergence_detector.detecting[:d].any():
            #         loss_masks_d = torch.zeros_like(loss_masks_d)

            losses_masks.append(loss_masks_d.clone())

            if d > 0:
                # Track distance to neighbours
                dist_neighbours = torch.norm(curve_points_d[:, 1:] - curve_points_d[:, -1], dim=-1)
                stats[f'msc/dist_neighbours/{d}/mean'] = dist_neighbours.mean()
                # if d > 1:
                #     stats[f'msc/dist_neighbours/{d}/var'] = dist_neighbours.var()

                # # Siblings should stay "either side" of the parent
                # if d > 1:
                #     curve_points_prev = batch_args['curve_parameters'][d-1].detach()
                #
                #     curve_points_d[:, 1:-1] - curve_points_d[:, :-2]
                #
                #     # Draw line between parent's neighbours
                #     directions = curve_points_prev[:, 1:-1] - curve_points_prev[:, :-2]
                #
                #     #

                # Distance to immediate neighbours should be the same - take central differences
                dist_ltr = torch.norm(curve_points_d[:, 1:-1] - curve_points_d[:, :-2], dim=-1)
                dist_rtl = torch.norm(curve_points_d[:, 2:] - curve_points_d[:, 1:-1], dim=-1)
                # dist_neighbours_diffs = (dist_ltr - dist_rtl)**2
                # loss_dists_neighbours = dist_neighbours_diffs.sum()
                loss_dists_neighbours = torch.sum((torch.log(1 + dist_ltr) - torch.log(1 + dist_rtl))**2)
                # stats[f'msc/dist_neighbours_diffs/{d}/mean'] = dist_neighbours_diffs.mean()
                # stats[f'msc/dist_neighbours_diffs/{d}/var'] = dist_neighbours_diffs.var()

                # # Distances to neighbours should be larger than distances to the next neighbours
                # if d > 1:
                #     dist_neighbours2 = torch.norm(curve_points_d[:, 2:] - curve_points_d[:, :-2], dim=-1)
                #     a = dist_neighbours[:, 1:]
                #     b = dist_neighbours2
                #     loss_dists_neighbours += torch.where(a > b, a, torch.zeros_like(a)).sum()
                # loss_dists_neighbours += torch.clamp(dist_neighbours.amax() - dist_neighbours2.amin(), min=0)

                losses_dists_neighbours.append(loss_dists_neighbours)

                # Track distance to parent
                curve_points_parent = torch.repeat_interleave(batch_args['curve_parameters'][d - 1], repeats=2, dim=1)
                # print('\n\n')
                # print('batch_args[\'curve_parameters\'][d - 1]', batch_args['curve_parameters'][d - 1].shape)
                # print('curve_points_parent', curve_points_parent.shape, curve_points_parent.max(), curve_points_parent.min())
                dist_parent = torch.norm(curve_points_d - curve_points_parent, dim=-1)
                # dist_parent = torch.norm(curve_points_d - curve_points_parent.detach(), dim=-1)
                # print('dist_parent', dist_parent.shape, dist_parent.max(), dist_parent.min())
                # stats[f'msc/dist_parent/{d}/mean'] = dist_parent.mean()
                # stats[f'msc/dist_parent/{d}/var'] = dist_parent.var()

                # Distance to parent should be same for siblings
                # dist_parent_diff = (dist_parent[:, ::2] - dist_parent[:, 1::2])**2
                dist_parent_diff = torch.sum((torch.log(1 + dist_parent[:, ::2]) - torch.log(1 + dist_parent[:, 1::2]))**2)
                # print('dist_parent_diff', dist_parent_diff.shape)  #, dist_parent_diff.max(), dist_parent_diff.min())
                # stats[f'msc/dist_parent_diff/{d}/mean'] = dist_parent_diff.mean()
                # stats[f'msc/dist_parent_diff/{d}/var'] = dist_parent_diff.var()
                losses_dists_parent.append(dist_parent_diff)

                # # losses_dists_parent.append(dist_parent.var())
                # a = dist_parent
                # b = dist_parent.mean().detach()
                # loss_dists_parent_d = torch.sum((torch.log(1 + a) - torch.log(1 + b))**2)
                # losses_dists_parent.append(loss_dists_parent_d)

                # Add aunt-pulls
                parents = batch_args['curve_parameters'][d - 1]
                left_children = curve_points_d[:, ::2]
                right_children = curve_points_d[:, 1::2]
                left_children_to_left_aunt = torch.norm(left_children[:, 1:] - parents[:, :-1], dim=-1)**2
                right_children_to_right_aunt = torch.norm(right_children[:, :-1] - parents[:, 1:], dim=-1)**2

                left_children_to_right_children = torch.norm(left_children[:, 1:] - right_children[:, :-1], dim=-1)**2
                #
                # loss_aunts_d = torch.sum(
                #     (left_children_to_right_children - (left_children_to_left_aunt + right_children_to_right_aunt))**2
                # )

                # loss_aunts_d = left_children_to_left_aunt.sum() + right_children_to_right_aunt.sum()
                a = left_children_to_right_children
                b = (left_children_to_left_aunt + right_children_to_right_aunt)
                loss_aunts_d = torch.sum((torch.log(1 + a) - torch.log(1 + b))**2)

                # print('loss_aunts_d', loss_aunts_d.item())
                # stats[f'msc/dist_aunts/{d}'] = loss_aunts_d
                losses_dists_aunts.append(loss_aunts_d)

            # loss_temporal = self._calculate_temporal_losses('cloud') # todo

            # Curve points scores
            # loss_scores_d = ((1 - points_scores_d)**2).mean()
            loss_scores_d = torch.sum((torch.log(1 + points_scores_d) - torch.log(1 + points_scores_d.mean().detach()))**2)
            # stats[f'msc/scores/{d}'] = loss_scores_d  #loss_d
            losses_scores.append(loss_scores_d)

            # Curve sigmas losses
            # loss_sigmas_d = torch.mean((sigmas_d - sigmas_d.mean())**2)
            # loss_sigmas_d = (torch.amax(sigmas_d) - torch.amin(sigmas_d))**2

            loss_sigmas_d = 0.1 * torch.sum((torch.log(1 + sigmas_d) - torch.log(1 + sigmas_d.mean().detach()))**2)

            # if d > 1:
            #     n = sigmas_d.shape[1]
            #     mp = int(n/2)
            #     sd1 = torch.clamp(sigmas_d[:, :mp-1] - sigmas_d[:, 1:mp], min=0).sum()
            #     sd2 = torch.clamp(sigmas_d[:, mp+1:] - sigmas_d[:, mp:-1], min=0).sum()
            #
            #     qp = int(n/4)
            #     middle_section = sigmas_d[:, qp:3*qp]
            #     sd3 = torch.sum((torch.log(1 + middle_section) - torch.log(1 + middle_section.mean().detach()))**2)
            #
            #     loss_sigmas_d = 0.1*sd1 + 0.1*sd2 + sd3
            #
            # else:
            #     loss_sigmas_d = torch.sum((torch.log(1 + sigmas_d) - torch.log(1 + sigmas_d.mean().detach()))**2)

            loss_sigmas_d += torch.sum((sigmas_d - sigmas_smoothed_d)**2)

            # print('loss_sigmas_d', loss_sigmas_d.shape, loss_sigmas_d.max(), loss_sigmas_d.min(), loss_sigmas_d.sum())
            # stats[f'msc/sigmas/{d}'] = loss_sigmas_d
            losses_sigmas.append(loss_sigmas_d)

            # Curve rendering intensity losses
            loss_intensities_d = torch.sum((torch.log(1 + intensities_d) - torch.log(1 + intensities_d.mean().detach()))**2)
            # n = sigmas_d.shape[1]
            # mp = int(n/2)
            # id1 = torch.clamp(intensities_d[:, :mp-1] - intensities_d[:, 1:mp], min=0)
            # id2 = torch.clamp(intensities_d[:, mp+1:] - intensities_d[:, mp:-1], min=0)
            # loss_intensities_d = torch.sum( id1 + id2)
            # loss_intensities_d = torch.sum((intensities_d - intensities_d.mean(dim=1, keepdim=True))**2)

            # loss_intensities_d = torch.sum((intensities_d - intensities_smoothed_d)**2)
            losses_intensities.append(loss_intensities_d)

            # Smoothness loss
            if d > 3:
                loss_smoothness_d = torch.sum((curve_points_d - curve_points_smoothed_d)**2)
                sf = (d-3)/(D-3)
                losses_smoothnesses.append(sf * loss_smoothness_d)

            # stats = {**stats, **{
            #     f'msc/scores/{d}/mean': points_scores_d.mean(),
            #     f'msc/scores/{d}/var': points_scores_d.var(),
            #     f'msc/sigmas/{d}/mean': sigmas_d.mean(),
            #     f'msc/sigmas/{d}/var': sigmas_d.var(),
            # }}

        self.targets = targets
        self.points_2d = points_2d
        # exit()

        # Masks losses
        losses_masks = list(reversed(losses_masks))
        loss_masks = sum(losses_masks)
        stats['msc/loss_masks'] = loss_masks
        if oa.loss_curve_masks > 0:
            loss += oa.loss_curve_masks * loss_masks

        # Distance losses to neighbours
        loss_dists_neighbours = sum(losses_dists_neighbours)
        stats['msc/loss_dists_neighbours'] = loss_dists_neighbours
        if oa.loss_curve_dists_neighbours > 0:
            loss += oa.loss_curve_dists_neighbours * loss_dists_neighbours

        # Distance losses to parents
        loss_dists_parents = sum(losses_dists_parent)
        stats['msc/loss_dists_parents'] = loss_dists_parents
        if oa.loss_curve_dists_parents > 0:
            loss += oa.loss_curve_dists_parents * loss_dists_parents

        # Distance losses to aunts
        loss_dists_aunts = sum(losses_dists_aunts)
        stats['msc/loss_dists_aunts'] = loss_dists_aunts
        if oa.loss_curve_dists_aunts > 0:
            loss += oa.loss_curve_dists_aunts * loss_dists_aunts

        # Curve points scores
        loss_scores = sum(losses_scores)
        stats['msc/loss_scores'] = loss_scores
        stats['msc/scores_total'] = torch.cat(points_scores, dim=1).sum()
        if oa.loss_curve_scores > 0:
            loss += oa.loss_curve_scores * loss_scores

        # Curve sigmas losses
        loss_sigmas = sum(losses_sigmas)
        stats['msc/sigmas'] = loss_sigmas
        if oa.loss_curve_sigmas > 0:
            loss += oa.loss_curve_sigmas * loss_sigmas

        # Curve rendering intensities losses
        loss_intensities = sum(losses_intensities)
        stats['msc/intensities'] = loss_intensities
        if oa.loss_curve_intensities > 0:
            loss += oa.loss_curve_intensities * loss_intensities

        # Intensities should be ordered (?)
        # loss_intensities_ordering = 0
        # for d in range(1, D):
        #     intensities_ub = batch_args['blur_intensities_curve'][d-1].min()
        #     intensities_d = batch_args['blur_intensities_curve'][d]
        #     lio = torch.where(intensities_d > intensities_ub, intensities_d, torch.zeros_like(intensities_d))
        #     loss_intensities_ordering += lio.sum()
        # intensities_means = torch.tensor([
        #     batch_args['blur_intensities_curve'][d].mean()
        #     for d in range(D)
        # ])
        # loss_intensities_ordering = torch.sum(torch.clamp(intensities_means[1:] - intensities_means[:-1], min=0))
        # loss_intensities_ordering = torch.exp(loss_intensities_ordering) - 1
        # stats['msc/intensities_ordering'] = loss_intensities_ordering
        # loss += loss_intensities_ordering

        # Curve smoothness losses
        loss_smoothness = sum(losses_smoothnesses)
        stats['msc/smoothness'] = loss_smoothness
        if oa.loss_curve_smoothness > 0:
            loss += oa.loss_curve_smoothness * loss_smoothness

        # Sigmas should be ordered
        # loss_sigma_ordering = 0
        # for d in range(1, D):
        #     sigmas_ub = batch_args['blur_sigmas_curve'][d-1].min()
        #     sigmas_d = batch_args['blur_sigmas_curve'][d]
        #     lso = torch.where(sigmas_d > sigmas_ub, sigmas_d, torch.zeros_like(sigmas_d))
        #     loss_sigma_ordering += lso.sum()

        # loss_sigma_ordering = 0
        # for d in range(D-1, 0, -1):
        #     sd = batch_args['blur_sigmas_curve'][d]
        #     sd_parent = torch.repeat_interleave(batch_args['blur_sigmas_curve'][d-1], repeats=2, dim=1)
        #     loss_sigma_ordering += torch.clamp(sd - sd_parent, min=0).sum()
        #     # sd_too_large = torch.where(sd > sd_parent, sd, torch.zeros_like(sd))
        #     # sd_too_small = torch.where(sd > sd_parent, torch.zeros_like(sd), sd_parent)
        #     # loss_sigma_ordering += sd_too_large.sum()
        #     # loss_sigma_ordering += -sd_too_small.sum()

        # sigmas_means = torch.tensor([
        #     batch_args['blur_sigmas_curve'][d].mean()
        #     for d in range(D)
        # ])
        # loss_sigma_ordering = torch.sum(torch.clamp(sigmas_means[1:] - sigmas_means[:-1], min=0))
        # stats['msc/sigmas_ordering'] = loss_sigma_ordering
        # loss += loss_sigma_ordering

        # Sigma cameras sfs should average 1
        sigmas_sfs_loss = torch.sum((sigmas_sfs - 1)**2)
        loss += sigmas_sfs_loss
        stats['loss/sigmas_sfs_loss'] = sigmas_sfs_loss

        # Intensities cameras sfs should average 1
        intensities_sfs_loss = torch.sum((intensities_sfs - 1)**2)
        loss += intensities_sfs_loss
        stats['loss/intensities_sfs_loss'] = intensities_sfs_loss

        # Log the scale factors
        for i in range(3):
            stats[f'sfs/sigmas/{i}'] = sigmas_sfs[:, i].mean()
            stats[f'sfs/intensities/{i}'] = intensities_sfs[:, i].mean()

        # Camera rotation preangles loss
        pre_angles = self.master_frame_state.get_state('cam_rotation_preangles')
        preangles_loss = torch.mean((1 - torch.norm(pre_angles, dim=2))**2)
        loss += preangles_loss
        stats['loss/cam_preangles'] = preangles_loss

        if is_bad(loss):
            if is_bad(preangles_loss):
                logger.warning('Bad preangles loss, skipping parameter update.')
            elif is_bad(loss_scores):
                logger.warning('Bad loss scores, skipping parameter update.')
            elif is_bad(loss_sigmas):
                logger.warning('Bad loss sigmas, skipping parameter update.')
            else:
                logger.warning('Bad renders loss, skipping parameter update.')
        else:
            self.optimiser_curve.zero_grad()
            loss.backward()
            self.optimiser_curve.step()

            with torch.no_grad():
                sigmas = self.master_frame_state.get_state('blur_sigmas_curve')
                for sigmas_d in sigmas:
                    sigmas_d.data = sigmas_d.clamp(min=5e-3)
                intensities = self.master_frame_state.get_state('blur_intensities_curve')
                for intensities_d in intensities:
                    intensities_d.data = intensities_d.clamp(min=1e-2, max=1.5)

        # self.masks_curve = masks
        # self.curve_points_scores = points_scores


        self.master_frame_state.set_state('masks_curve', [masks[d][0] for d in range(D)])
        self.master_frame_state.set_state('curve_points_scores', [points_scores[d][0] for d in range(D)])
        self.master_frame_state.set_stats(stats)

        # # Update batch state
        # for i, fs in enumerate(self.frame_batch):
        #     fs.set_state('masks_curve', [masks[d][i] for d in range(D)])
        #     fs.set_state('curve_points_scores', [points_scores[d][i] for d in range(D)])
        #     fs.set_stats(stats)

        return loss, losses_masks, stats

    def _get_curve_coordinates(self) -> torch.Tensor:
        """
        Decode the curve parameters into the 3D coordinates.
        """
        if self.model_params.curve_mode == ENCODING_MODE_POINTS:
            # Parameters are the curve coordinates
            return torch.stack(self.curve_parameters)

        cp = torch.stack(self.curve_parameters)
        bs = cp.shape[0]

        # First 3 parameters are the offset
        offset = cp[:, :3]
        parameters = cp[:, 3:]

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
            # max_revs = torch.linspace(self.model_params.max_revolutions * 2, 0.5, bs, device=self.device)
            # max_delta_angle = max_revs * 2 * np.pi / N
            max_delta_angle = self.model_params.max_revolutions * 2 * np.pi / N

            # Remaining parameters are the delta angles
            delta_angles = torch.tanh(parameters.reshape((bs, 2, -1))) * max_delta_angle  # [:, None, None]

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
        cl = torch.stack(self.curve_length)[:, None, None]
        tau = e0s * cl / N

        # Start at the offset and add the tangent vectors to form the curve
        curve_coordinates = offset[:, None, :] + torch.cumsum(tau, dim=1)

        return curve_coordinates

    def _update_curve_points(self):
        cc = self._get_curve_coordinates()
        for i, fs in enumerate(self.frame_batch):
            fs.set_state('curve_points', cc[i])

    def _log_progress(self, cc_or_curve: str, step: int, final_step: int, loss: float, stats: dict):
        """
        Log the loss and stats to the tensorboard logger and command line.
        """
        log_msg = f'[{step}/{final_step - 1}]\tLoss: {loss:.5E}'
        checkpoint_step = getattr(self.checkpoint, f'step_{cc_or_curve}')
        self.tb_logger.add_scalar(f'{cc_or_curve}/loss', loss, checkpoint_step)
        for key, val in stats.items():
            self.tb_logger.add_scalar(f'{cc_or_curve}/{key}', val, checkpoint_step)
            if key in PRINT_KEYS:
                log_msg += f'\t{key}: {val:.3E}'
        logger.info(log_msg)

        # Update moving average
        rate = EMA_RATE_CC if cc_or_curve == 'cc' else EMA_RATE_CURVE
        loss_ema = self.ema(f'loss.{rate}/{cc_or_curve}', loss.item())
        self.tb_logger.add_scalar(f'loss.{rate}/{cc_or_curve}', loss_ema, checkpoint_step)

        # Log camera coefficients
        if cc_or_curve == 'cc' or self.model_params.curve_mode == ENCODING_MODE_MSC:
            # coefficients = self.master_frame_state.get_state('cam_coeffs')

            # Extract parameters
            # intrinsics = self.master_frame_state.get_state('cam_intrinsics')
            # fx = intrinsics[:, 0]
            # fy = intrinsics[:, 1]
            # cx = intrinsics[:, 2]
            # cy = intrinsics[:, 3]
            # # rotation = coefficients[:, 4:13]
            # # rotation = self.master_frame_state.get_state('cam_rotations')
            # rotation = self.frame_batch[0].get_state('cam_rotations')
            # # translation = coefficients[:, 13:16]
            # translation = self.master_frame_state.get_state('cam_translations')
            # # distortion = coefficients[:, 16:21]
            # distortion = self.master_frame_state.get_state('cam_distortions')
            # # shifts = coefficients[:, 21]
            # shifts = self.master_frame_state.get_state('cam_shifts')

            # Rotation angles
            rotation_preangles = self.master_frame_state.get_state('cam_rotation_preangles')

            # rotation_preangles[i] = torch.tensor([
            #     [cos_phi, sin_phi],
            #     [cos_theta, sin_theta],
            #     [cos_psi, sin_psi],
            # ])
            rotation_angles = torch.atan2(rotation_preangles[:, :, 0], rotation_preangles[:, :, 1])

            # Log
            for i in range(3):
                # self.tb_logger.add_scalar(f'cam_coeffs/fx/{i}', fx[i], checkpoint_step)
                # self.tb_logger.add_scalar(f'cam_coeffs/fy/{i}', fy[i], checkpoint_step)
                # self.tb_logger.add_scalar(f'cam_coeffs/cx/{i}', cx[i], checkpoint_step)
                # self.tb_logger.add_scalar(f'cam_coeffs/cy/{i}', cy[i], checkpoint_step)
                # for j in range(9):
                #     self.tb_logger.add_scalar(f'cam_coeffs/R[{j}]/{i}', rotation[i, j], checkpoint_step)
                # for j in range(3):
                #     self.tb_logger.add_scalar(f'cam_coeffs/t[{j}]/{i}', translation[i, j], checkpoint_step)
                # for j in range(5):
                #     self.tb_logger.add_scalar(f'cam_coeffs/d[{j}]/{i}', distortion[i, j], checkpoint_step)
                # self.tb_logger.add_scalar(f'cam_coeffs/shifts/{i}', shifts[i], checkpoint_step)

                for j, angle in enumerate(['phi', 'theta', 'psi']):
                    self.tb_logger.add_scalar(f'cam_rotation_angles/{i}/{angle}', rotation_angles[i, j], checkpoint_step)

        detector = self.convergence_detector
        if detector is not None:
            for d in range(self.model_params.ms_curve_depth):
                state_vars = ['mu_fast', 'mu_slow', 'convergence_count', 'converged']
                for k in state_vars:
                    self.tb_logger.add_scalar(f'detector/{d}/{k}', getattr(detector, k)[d].item(), checkpoint_step)

    def _make_plots(
            self,
            pre_step: bool = False,
            final_step: bool = False,
            show_cloud: bool = True,
            show_curve: bool = True,
            plot_sigmas: bool = False,
            plot_intensities: bool = False,
            plot_scores: bool = False,
    ):
        """
        Generate some plots.
        """
        bs = 1  #len(self.frame_batch)

        # Make initial plots for all batch elements
        if pre_step:
            if self.model_params.curve_mode == ENCODING_MODE_MSC:
                self._plot_3d_msc()
            elif bs > 1:
                self._plot_3d_batch(show_cloud, show_curve)
            else:
                self._plot_3d(0, show_cloud, show_curve)
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and self.step % self.runtime_args.plot_every_n_steps == 0
        ):
            logger.info('Plotting.')
            if self.model_params.curve_mode == ENCODING_MODE_MSC:
                self._plot_3d_msc()
                self._plot_2d_msc()
                if plot_sigmas:
                    self._plot_sigmas_msc()
                if plot_intensities:
                    self._plot_intensities_msc()
                if plot_scores:
                    self._plot_scores_msc()
            elif bs > 1:
                self._plot_3d_batch(show_cloud, show_curve,
                                    self.optimiser_args.loss_3d_cloud_threshold if show_curve else 0)
                self._plot_2d_batch()
                self._plot_scores_batch()
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
        self.plot_3d_azim += 10
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
            # scores = self.curve_points_scores[idx]
            x, y, z = (to_numpy(curve_points[:, j]) for j in range(3))
            # s2 = ax.scatter(x, y, z, c=to_numpy(scores), cmap='YlGnBu', s=50, marker='x', alpha=0.9)
            s2 = ax.scatter(x, y, z, color='black', s=75, marker='x', alpha=0.9)
            # fig.colorbar(s2)

        # Fix equal aspect ratio
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis=1))

        ax.set_title(f'step_cc: {self.checkpoint.step_cc}. step_curve: {self.checkpoint.step_curve}.')
        fig.tight_layout()
        self._save_plot(fig, '3d')

    def _plot_3d_msc(self):
        """
        Make a multiscale curve 3D scatter plot.
        """
        cmap_vertices = 'autumn_r'
        cmap_curve = 'jet'
        D = self.model_params.ms_curve_depth
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        # Rotate the perspective on every plot
        self.plot_3d_azim += 1

        fig = plt.figure(figsize=(n_cols * 4, 1 + n_rows * 4))
        fig.suptitle(
            f'Trial: {self.trial.id}. Frame: {self.master_frame_state.frame_num}/{self.trial_state.frame_nums[-1]}.\n'
            f'Step: {self.checkpoint.step_curve}. '
        )
        gs = GridSpec(n_rows, n_cols)
        d = 0

        curve_parameters = self.master_frame_state.get_state('curve_parameters')
        curve_points_scores = self.master_frame_state.get_state('curve_points_scores')
        blur_sigmas_curve = self.master_frame_state.get_state('blur_sigmas_curve')

        for i in range(n_rows):
            for j in range(n_cols):
                if d >= D:
                    break
                ax = fig.add_subplot(gs[i, j], projection='3d')
                ax.view_init(azim=self.plot_3d_azim)
                ax.set_title(f'd={d}')
                # cla(ax)

                # Scatter vertices
                vertices = to_numpy(curve_parameters[d])
                scores = to_numpy(curve_points_scores[d])
                sigmas = np.clip(
                    # np.exp(10+to_numpy(blur_sigmas_curve[d])),
                    2000*to_numpy(blur_sigmas_curve[d]),
                    a_min=10,
                    a_max=1000,
                )
                # print('blur_sigmas_curve', blur_sigmas_curve[d].shape, blur_sigmas_curve[d])
                # print('sigmas', sigmas.shape, sigmas)
                x, y, z = (vertices[:, j] for j in range(3))
                s1 = ax.scatter(x, y, z, c=scores, cmap=cmap_vertices, s=sigmas, alpha=0.4)  #, zorder=-1)
                # s1 = ax.scatter(x, y, z, c=scores, cmap=cmap_vertices, s=20, alpha=0.8, zorder=-1)
                # fig.colorbar(s1)

                # # Draw lines connecting points
                colours = np.linspace(0, 1, len(vertices))
                points = vertices[:, None, :]
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=colours, cmap=cmap_curve, zorder=-2, alpha=0.2)
                ax.add_collection(lc)



                # Find axes limits
                if d > 0:
                    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
                    ax.set_box_aspect(np.ptp(limits, axis = 1))

                    # limits = {
                    #     ['x', 'y', 'z'][i]: {
                    #         'min': vertices[:, i].min(),
                    #         'max': vertices[:, i].max()
                    #     }
                    #     for i in range(3)
                    # }
                    # max_range = np.max([l['max'] - l['min'] for _, l in limits.items()])
                    #
                    # # print('vertices', vertices.shape, vertices)
                    # # print('limits', limits)
                    #
                    # # Adjust limits to match
                    # for xyz, l in limits.items():
                    #     r = l['max'] - l['min']
                    #     sf = max_range / r
                    #     limits[xyz]['max'] = l['max'] * sf
                    #     limits[xyz]['min'] = l['min'] * sf
                    #
                    # ax.set_xlim(limits['x']['min'], limits['x']['max'])
                    # ax.set_ylim(limits['y']['min'], limits['y']['max'])
                    # ax.set_zlim(limits['z']['min'], limits['z']['max'])

                d += 1

        fig.tight_layout()
        self._save_plot(fig, '3d_msc')

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
        n_rows = int(np.ceil(np.sqrt(ws)))
        n_cols = int(np.ceil(np.sqrt(ws)))

        # Rotate the perspective on every plot
        self.plot_3d_azim += 5

        fig = plt.figure(figsize=(n_cols * 3, 1 + n_rows * 3))
        fig.suptitle(
            f'Trial: {self.trial.id}. \n'
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )
        gs = GridSpec(n_rows, n_cols)
        idx = 0

        # Find axes limits
        cloud_points = to_numpy(torch.stack(self.cloud_points))
        curve_points = to_numpy(torch.stack(self.curve_points))
        # print('cloud_points.shape', cloud_points.shape)
        # print('curve_points.shape', curve_points.shape)

        limits = {
            ['x', 'y', 'z'][i]: {
                'min': np.min([cloud_points[:, :, i].min(), curve_points[:, :, i].min()]),
                'max': np.max([cloud_points[:, :, i].max(), curve_points[:, :, i].max()])
            }
            for i in range(3)
        }
        max_range = np.max([l['max'] - l['min'] for _, l in limits.items()])

        # Adjust limits to match
        for xyz, l in limits.items():
            r = l['max'] - l['min']
            sf = max_range / r
            limits[xyz]['max'] = l['max'] * sf
            limits[xyz]['min'] = l['min'] * sf

        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= len(self.frame_batch):
                    break
                fs = self.frame_batch[idx]
                ax = fig.add_subplot(gs[i, j], projection='3d')
                ax.view_init(azim=self.plot_3d_azim)
                ax.set_title(f'Frame #{fs.frame_num}')
                cla(ax)

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
                    s2 = ax.scatter(x, y, z, color='darkgrey', s=30, marker='x', alpha=0.4)
                    # fig.colorbar(s2)

                ax.set_xlim(limits['x']['min'], limits['x']['max'])
                ax.set_ylim(limits['y']['min'], limits['y']['max'])
                ax.set_zlim(limits['z']['min'], limits['z']['max'])

                idx += 1

        fig.tight_layout()
        self._save_plot(fig, '3d_batch')

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
        images = to_numpy(self.images[idx])

        n_rows = 1 + int(show_cloud) + int(show_curve)
        fig, axes = plt.subplots(n_rows, figsize=(8, 1 + 2 * n_rows))
        fig.suptitle(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame_num}. '
            f'idx: {idx}. '
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )

        # Stitch images and masks together
        image_triplet = np.concatenate(images, axis=1)
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

        if show_curve and X_curve.max() > 0:
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

    def _plot_2d_msc(self):
        """
        Plot the 2D mask renderings of the mutiscale curves.
        """
        D = self.model_params.ms_curve_depth
        n_rows = int(np.ceil(np.sqrt(D)))
        n_cols = int(np.ceil(np.sqrt(D)))

        fig = plt.figure(figsize=(n_cols * 3, 1 + n_rows * 3))
        fig.suptitle(
            f'Trial: {self.trial.id}. Frame: {self.master_frame_state.frame_num}/{self.trial_state.frame_nums[-1]}.\n'
            f'Step: {self.checkpoint.step_curve}. '
        )
        gs = GridSpec(n_rows, n_cols)
        d = 0

        # masks_target = self.master_frame_state.get_state('masks_target')
        # masks_target = self.frame_batch[0].get_state('masks_target')
        points_2d = self.points_2d
        masks_target = self.targets
        # masks_curve = self.master_frame_state.get_state('masks_curve')
        masks_curve = self.master_frame_state.get_state('masks_curve')


        images = to_numpy(self.master_frame_state.get_state('images'))
        image_triplet = np.concatenate(images, axis=1)
        # image_grid = np.concatenate([image_triplet,image_triplet, np.ones_like(image_triplet)], axis=0)
        image_grid = np.concatenate([np.ones_like(image_triplet),image_triplet, np.ones_like(image_triplet)], axis=0)
        # image_grid = np.concatenate([image_triplet, np.zeros_like(image_triplet), np.ones_like(image_triplet)], axis=0)
        M, N = tuple(image_grid.shape)

        scatter_sizes = (np.linspace(1, 0, D)**2 * 100)+0.3
        # print('scatter_sizes', scatter_sizes)

        for i in range(n_rows):
            for j in range(n_cols):
                if d >= D:
                    break
                ax = fig.add_subplot(gs[i, j])
                ax.set_title(f'd={d}')
                ax.axis('off')

                X_target = to_numpy(masks_target[d][0])
                X_curve = to_numpy(masks_curve[d])

                # Stitch images and masks together
                ax.imshow(image_grid, cmap='gray', vmin=0, vmax=1)
                ax.set_xlim((0, N))
                ax.set_ylim((M, 0))

                # Target overlay
                X_target_triplet = np.concatenate(X_target, axis=1)  # / X_target.max()
                alphas = X_target_triplet.copy()
                # alphas[alphas < 0.1] = 0
                # alphas[alphas > 0.1] = 1
                # (left, right, bottom, top)
                ax.imshow(X_target_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal',  #alpha=alphas,
                          extent=(0, N - 1, int(M / 3), 0))

                # Curve overlay
                X_curve_triplet = np.concatenate(X_curve, axis=1) / X_curve.max()
                alphas = X_curve_triplet.copy() * 0.5
                # alphas[alphas < 0.1] = 0
                # alphas[alphas > 0.2] = 1
                ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                          extent=(0, N - 1, 2 * int(M / 3), int(M / 3)))

                # scatter the midline points
                p2d = to_numpy(points_2d[d][0])
                # print(p2d.shape)

                # p2d = np.concatenate([
                #     p2d[0],
                #     p2d[1] + (0, 100),
                #     p2d[2] + (0, 200),
                # ])
                # print(p2d.shape)

                # if d < 5:
                #     print(p2d)

                # exit()
                for k in range(3):
                    p = p2d[k] + (0, 200)
                    if k == 1:
                        p += (200, 0)
                    elif k == 2:
                        p += (400, 0)
                    ax.scatter(p[:, 0], p[:, 1], cmap='jet', c=np.linspace(0, 1, len(p)), s=scatter_sizes[d], alpha=0.6)

                # Errors
                # print('X_curve_triplet', X_curve_triplet.shape, X_curve_triplet.min(), X_curve_triplet.max(), X_curve_triplet.sum())
                # print('X_target_triplet', X_target_triplet.shape, X_target_triplet.min(), X_target_triplet.max(), X_target_triplet.sum())
                errors_triplet = X_curve_triplet - X_target_triplet
                errors_triplet = errors_triplet / np.abs(errors_triplet).max()
                ax.imshow(errors_triplet, vmin=-1, vmax=1, cmap='PRGn', aspect='equal',
                          extent=(0, N - 1, M - 1, 2 * int(M / 3)))

                d += 1

        fig.tight_layout()
        self._save_plot(fig, '2d_msc')

    def _plot_2d_batch(self):
        """
        Plot a batch of 2D mask renderings; target, cloud and curve.
        """
        ws = self.optimiser_args.window_size
        n_rows = int(np.ceil(np.sqrt(ws)))
        n_cols = int(np.ceil(np.sqrt(ws)))

        fig = plt.figure(figsize=(n_cols * 3, 1 + n_rows * 3))
        gs = GridSpec(n_rows, n_cols)
        fig.suptitle(
            f'Trial: {self.trial.id}. \n'
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )
        idx = 0

        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= len(self.frame_batch):
                    break
                fs = self.frame_batch[idx]
                ax = fig.add_subplot(gs[i, j])
                is_frozen = ' (frozen)' if fs.is_frozen else ''
                ax.set_title(f'Frame #{fs.frame_num}{is_frozen}')
                ax.axis('off')

                images = to_numpy(self.images[idx])
                X_target = to_numpy(self.masks_target[idx])
                X_cloud = to_numpy(self.masks_cloud[idx])
                X_curve = to_numpy(self.masks_curve[idx])

                # Stitch images and masks together
                image_triplet = np.concatenate(images, axis=1)
                image_grid = np.concatenate([image_triplet] * 3, axis=0)
                M, N = tuple(image_grid.shape)
                ax.imshow(image_grid, cmap='gray', vmin=0, vmax=1)
                ax.set_xlim((0, N))
                ax.set_ylim((M, 0))

                # Target overlay
                X_target_triplet = np.concatenate(X_target, axis=1) / X_target.max()
                alphas = X_target_triplet.copy()
                alphas[alphas < 0.1] = 0
                alphas[alphas > 0.1] = 1
                # (left, right, bottom, top)
                ax.imshow(X_target_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                          extent=(0, N - 1, int(M / 3), 0))

                # Cloud overlay
                X_cloud_triplet = np.concatenate(X_cloud, axis=1) / X_cloud.max()
                alphas = X_cloud_triplet.copy()
                # alphas[alphas < 0.1] = 0
                alphas[alphas > 0.2] = 1
                ax.imshow(X_cloud_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                          extent=(0, N - 1, 2 * int(M / 3), int(M / 3)))

                # Curve overlay
                if X_curve.max() > 0:
                    X_curve_triplet = np.concatenate(X_curve, axis=1) / X_curve.max()
                    alphas = X_curve_triplet.copy()
                    alphas[alphas < 0.1] = 0
                    alphas[alphas > 0.2] = 1
                    ax.imshow(X_curve_triplet, vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas,
                              extent=(0, N - 1, M - 1, 2 * int(M / 3)))

                idx += 1

        fig.tight_layout()
        self._save_plot(fig, '2d_batch')

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
            f'Frame: {self.frame_num}. '
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


    def _plot_sigmas_msc(self):
        """
        Plot the sigmas for the mutiscale curves.
        """
        D = self.model_params.ms_curve_depth
        cmap = plt.get_cmap('jet')
        sigmas_curve = self.master_frame_state.get_state('blur_sigmas_curve')
        sigmas_cameras_sfs = self.master_frame_state.get_state('blur_sigmas_cameras_sfs')
        sfs = ", ".join([f"{sf:.3f}" for sf in sigmas_cameras_sfs])

        fig, ax = plt.subplots(1, figsize=(6, 8))
        ax.set_title(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame_num}. '
            f'Step: {self.checkpoint.step_curve}. '
            f'\nCameras sfs: {sfs}.'
        )
        colours = [cmap(d) for d in np.linspace(0, 1, D)]

        for d in range(D):
            sigmas = to_numpy(sigmas_curve[d])
            positions = np.linspace(0, 1, len(sigmas) + 2)[1:-1]
            ax.plot(positions, sigmas, label=f'd={d}', color=colours[d], alpha=0.5)
            ax.scatter(x=positions, y=sigmas, color=colours[d], s=10, alpha=0.8)

        fig.tight_layout()
        self._save_plot(fig, 'sigmas_msc')

    def _plot_intensities_msc(self):
        """
        Plot the intensities for the mutiscale curves.
        """
        D = self.model_params.ms_curve_depth
        cmap = plt.get_cmap('jet')
        intensities_curve = self.master_frame_state.get_state('blur_intensities_curve')
        intensities_cameras_sfs = self.master_frame_state.get_state('blur_intensities_cameras_sfs')
        sfs = ", ".join([f"{sf:.3f}" for sf in intensities_cameras_sfs])

        fig, ax = plt.subplots(1, figsize=(6, 8))
        ax.set_title(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame_num}. '
            f'Step: {self.checkpoint.step_curve}. '
            f'\nCameras sfs: {sfs}.'
        )
        colours = [cmap(d) for d in np.linspace(0, 1, D)]

        for d in range(D):
            intensities = to_numpy(intensities_curve[d])
            positions = np.linspace(0, 1, len(intensities) + 2)[1:-1]
            ax.plot(positions, intensities, label=f'd={d}', color=colours[d], alpha=0.5)
            ax.scatter(x=positions, y=intensities, color=colours[d], s=10, alpha=0.8)

        fig.tight_layout()
        self._save_plot(fig, 'intensities_msc')

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
            f'Frame: {self.frame_num}. '
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

    def _plot_scores_msc(self):
        """
        Plot the scores for the mutiscale curves.
        """
        D = self.model_params.ms_curve_depth
        cmap = plt.get_cmap('jet')
        scores_curve = self.master_frame_state.get_state('curve_points_scores')

        fig, ax = plt.subplots(1, figsize=(6, 8))
        ax.set_title(
            f'{self.trial.date:%Y%m%d} #{self.trial.trial_num}. \n'
            f'Frame: {self.frame_num}. '
            f'Step: {self.checkpoint.step_curve}. '
        )
        colours = [cmap(d) for d in np.linspace(0, 1, D)]

        for d in range(D):
            scores = to_numpy(scores_curve[d])
            positions = np.linspace(0, 1, len(scores) + 2)[1:-1]
            ax.plot(positions, scores, label=f'd={d}', color=colours[d], alpha=0.5)
            ax.scatter(x=positions, y=scores, color=colours[d], s=10, alpha=0.8)

        fig.tight_layout()
        self._save_plot(fig, 'scores_msc')

    def _plot_scores_batch(self):
        """
        Plot the scores across the batch.
        """
        point_scores = to_numpy(self.cloud_points_scores)
        n_rows = 2
        n_cols = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 6))
        fig.suptitle(
            f'Trial: {self.trial.id}. Frame: {self.frame_num}.\n'
            f'step_cc: {self.checkpoint.step_cc}. '
            f'step_curve: {self.checkpoint.step_curve}.'
        )

        ax_raw = axes[0]
        ax_raw.set_title('point_scores')
        ax_sorted = axes[1]
        ax_sorted.set_title('sort(point_scores)')

        for i, fs in enumerate(self.frame_batch):
            ax_raw.plot(point_scores[i], label=f'#{fs.frame_num}', alpha=0.5)
            ax_sorted.plot(np.sort(point_scores[i]), label=f'#{fs.frame_num}', alpha=0.5)
        ax_raw.legend()
        ax_sorted.legend()

        fig.tight_layout()
        self._save_plot(fig, 'scores_batch')

    def _save_plot(self, fig: Figure, plot_type: str):
        """
        Either log the figure to the tensorboard logger or save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path + f'/plots/{plot_type}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir + f'/{self.frame_num:05d}_{self.step:06d}.{img_extension}'
            plt.savefig(path, bbox_inches='tight')

        else:
            self.tb_logger.add_figure(plot_type, fig, self.step)
            self.tb_logger.flush()

        plt.close(fig)
