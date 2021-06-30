import math
import os
import shutil
import time
from datetime import timedelta
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from mongoengine import DoesNotExist
from torch.utils.tensorboard import SummaryWriter

from simple_worm.controls import CONTROL_KEYS
from simple_worm.controls_torch import ControlSequenceTorch, ControlSequenceBatchTorch
from simple_worm.frame_torch import FrameTorch, FrameSequenceBatchTorch, FrameSequenceTorch, FrameBatchTorch
from simple_worm.losses import REG_LOSS_TYPES, REG_LOSS_VARS
from simple_worm.losses_torch import LossesTorch
from simple_worm.plot3d import plot_X_vs_target, plot_FS_3d, generate_scatter_diff_clip, plot_CS, plot_frame_3d, \
    plot_frame_components
from simple_worm.util_torch import expand_tensor
from simple_worm.worm_torch import WormModule
from wormlab3d import LOGS_PATH, logger
from wormlab3d.data.model import Checkpoint, FrameSequence, Trial
from wormlab3d.data.model.sw_checkpoint import SwCheckpoint
from wormlab3d.data.model.sw_regularisation_parameters import SwRegularisationParameters
from wormlab3d.data.model.sw_run import SwRun, SwControlSequence, SwFrameSequence
from wormlab3d.data.model.sw_simulation_parameters import SwSimulationParameters
from wormlab3d.simple_worm.args import RuntimeArgs, FrameSequenceArgs, SimulationArgs, OptimiserArgs, RegularisationArgs
from wormlab3d.toolkit.util import to_dict, to_numpy

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')


class Manager:
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            frame_sequence_args: FrameSequenceArgs,
            simulation_args: SimulationArgs,
            optimiser_args: OptimiserArgs,
            regularisation_args: RegularisationArgs,
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.frame_sequence_args = frame_sequence_args
        self.simulation_args = simulation_args
        self.optimiser_args = optimiser_args
        self.regularisation_args = regularisation_args

        # Target frame sequence (from data)
        self.FS_db, self.FS_target = self._init_frame_sequence()

        # Initialise configurations
        self.sim_params, self.reg_params = self._init_configuration()

        # Initialise worm simulator
        self.worm = self._init_worm()

        # Initialise initial conditions and trainable parameters
        self.F0, self.CS = self._init_params()

        # Checkpoints
        self.checkpoint = self._init_checkpoint()

        # Outputs
        self.FS_out: FrameSequenceTorch
        self.FS_outs: List[FrameSequenceTorch] = []
        self.FS_labels: List[str] = []

    @property
    def logs_path(self) -> str:
        return self.get_logs_path(self.checkpoint)

    @staticmethod
    def get_logs_path(checkpoint: SwCheckpoint) -> str:
        sim_dir = f'/N={checkpoint.sim_args["worm_length"]:03d}' \
                  f'_T={checkpoint.sim_args["duration"]:06.2f}' \
                  f'_dt={checkpoint.sim_args["dt"]:04.2f}' \
                  f'_{checkpoint.sim_params.id}'

        FS_dir = f'/{checkpoint.frame_sequence_args["trial_id"]:03d}' \
                 f'_{checkpoint.frame_sequence_args["start_frame"]}' \
                 f'_{checkpoint.frame_sequence_args["midline_source"]}' \
                 f'_{checkpoint.frame_sequence.id}'

        regs_dir = f'/{checkpoint.reg_params.created:%Y%m%d_%H:%M}' \
                   f'_{checkpoint.reg_params.id}'

        return LOGS_PATH + sim_dir + FS_dir + regs_dir

    def _init_frame_sequence(self) -> Tuple[FrameSequence, FrameSequenceTorch]:
        """
        Load or create the target frame sequence.
        """
        FS_db = None

        # Try to load an existing frame sequence
        if self.frame_sequence_args.load:
            try:
                FS_db = self._load_frame_sequence()
            except DoesNotExist:
                logger.info('Frame sequence not found in database.')

        # Not loaded frame sequence, so create one
        if FS_db is None:
            FS_db = self._generate_frame_sequence()  # persists to the database

        # Create the simulation target frame sequence
        x = torch.from_numpy(FS_db.X)

        # Resample the worm points if required
        T = int(self.simulation_args.duration / self.simulation_args.dt)
        T_data = x.shape[0]
        N = self.simulation_args.worm_length
        N_data = x.shape[1]
        if T != T_data or N != N_data:
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = F.interpolate(x, size=(T, N), mode='bilinear', align_corners=True)
            x = x.squeeze(0).permute(1, 2, 0)

        # Permute dimensions for compatibility with simple-worm
        x = x.permute(0, 2, 1)
        assert x.shape == (T, 3, N)
        FS = FrameSequenceTorch(x=x)

        return FS_db, FS

    def _load_frame_sequence(self) -> FrameSequence:
        """
        Load an existing frame sequence from the database.
        """
        # If we have a fs id then load this from the database
        if self.frame_sequence_args.fs_id is not None:
            FS_db = FrameSequence.objects.get(id=self.frame_sequence_args.fs_id)
        else:
            # Otherwise, try to find one matching the same parameters
            trial = Trial.objects.get(id=self.frame_sequence_args.trial_id)
            n_frames = math.ceil(self.simulation_args.duration * trial.fps)
            FS_db = FrameSequence.find_from_args(self.frame_sequence_args, n_frames)
            if FS_db.count() > 0:
                logger.info(f'Found {len(FS_db)} matching frame sequences in database, using most recent.')
                FS_db = FS_db[0]
            else:
                raise DoesNotExist()

        logger.info(f'Loaded frame sequence id={FS_db.id}.')

        return FS_db

    def _generate_frame_sequence(self) -> FrameSequence:
        """
        Generate a frame sequence matching the given arguments and save it to the database.
        """
        logger.info('Generating frame sequence.')
        trial = Trial.objects.get(id=self.frame_sequence_args.trial_id)
        n_frames = math.ceil(self.simulation_args.duration * trial.fps)
        f0 = self.frame_sequence_args.start_frame
        fn = f0 + n_frames
        frame_nums = range(f0, fn)
        seq = []
        for frame_num in frame_nums:
            frame = trial.get_frame(frame_num)
            midlines = frame.get_midlines3d(filters={'source': self.frame_sequence_args.midline_source})
            if len(midlines) > 1:
                logger.info(
                    f'Found {len(midlines)} 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}. Picking at random..')
                midline = midlines[np.random.randint(len(midlines))]
            elif len(midlines) == 1:
                midline = midlines[0]
            else:
                raise RuntimeError(
                    f'Could not find any 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}.')
            seq.append(midline)

        # Create and save the database record
        FS_db = FrameSequence()
        FS_db.set_from_sequence(seq)
        FS_db.save()

        return FS_db

    def _init_configuration(self) -> Tuple[SwSimulationParameters, SwRegularisationParameters]:
        """
        Load or create the simulation and regularisation parameters.
        """

        # Find or create the database record for this simulation configuration
        if self.simulation_args.sim_id is not None:
            sim_params = SwSimulationParameters.get(id=self.simulation_args.sim_id)
        else:
            sim_config = self.simulation_args.get_config_dict()
            sim_params = SwSimulationParameters.objects(**sim_config)
            if sim_params.count() > 0:
                logger.info(
                    f'Found {len(sim_params)} matching simulation parameter records in database, using most recent.')
                sim_params = sim_params[0]
                logger.info(f'Loaded simulation parameters id={sim_params.id}.')
            else:
                logger.info('No suitable simulation parameter records found in database, creating new.')
                sim_params = SwSimulationParameters(**sim_config)
                sim_params.save()

        # Find or create the database record for the regularisation configuration
        if self.regularisation_args.reg_id is not None:
            reg_params = SwRegularisationParameters.get(id=self.regularisation_args.reg_id)
        else:
            rw = self.regularisation_args.get_reg_weights()
            reg_params = SwRegularisationParameters.objects(**rw)
            if reg_params.count() > 0:
                logger.info(
                    f'Found {len(reg_params)} matching regularisation parameter records in database, using most recent.')
                reg_params = reg_params[0]
                logger.info(f'Loaded regularisation parameters id={reg_params.id}.')
            else:
                logger.info('No suitable regularisation parameter records found in database, creating new.')
                reg_params = SwRegularisationParameters(**rw)
                reg_params.save()

        return sim_params, reg_params

    def _init_worm(self) -> WormModule:
        """
        Create the worm simulator.
        """
        worm = WormModule(
            N=self.sim_params.worm_length,
            dt=self.sim_params.dt,
            batch_size=self.optimiser_args.batch_size,
            reg_weights=self.regularisation_args.get_reg_weights(),
            inverse_opt_max_iter=self.optimiser_args.inverse_opt_max_iter,
            inverse_opt_tol=self.optimiser_args.inverse_opt_tol,
            parallel=self.runtime_args.parallel_solvers > 0,
            n_workers=self.runtime_args.parallel_solvers,
            quiet=False
        )
        return worm

    def _init_params(self) -> Tuple[FrameTorch, ControlSequenceTorch]:
        """
        Initialise the optimisable initial frame and control sequence.
        """
        bs = self.optimiser_args.batch_size

        # Generate optimisable initial frame, using known x0 and an estimate of psi0
        F0 = FrameBatchTorch(
            x=expand_tensor(self.FS_target[0].x, bs),
            estimate_psi=self.optimiser_args.estimate_psi,
            optimise=self.optimiser_args.optimise_F0,
            batch_size=bs
        )

        # Generate optimisable control sequence
        CS = ControlSequenceBatchTorch(
            worm=self.worm.worm_solver,
            n_timesteps=self.FS_target.x.shape[0],
            optimise=self.optimiser_args.optimise_CS,
            batch_size=bs
        )

        # Add some noise
        with torch.no_grad():
            if self.optimiser_args.init_noise_std_psi0 > 0:
                F0.psi += torch.normal(torch.zeros_like(F0.psi), std=self.optimiser_args.init_noise_std_psi0)
            for abg in CONTROL_KEYS:
                std = getattr(self.optimiser_args, f'init_noise_std_{abg}')
                if std > 0:
                    v = getattr(CS, abg)
                    v.data += torch.normal(torch.zeros_like(v), std=self.optimiser_args.init_noise_std_alpha)

        return F0, CS

    def _init_checkpoint(self):
        """
        The current checkpoint instance contains the most up to date instance of the model.
        This is not persisted to the database until we actually want to checkpoint it, so should
        be thought of more as a checkpoint-buffer.
        """

        # Load previous checkpoint
        prev_checkpoint: SwCheckpoint = None
        if self.runtime_args.resume:
            try:
                if self.runtime_args.resume_from in ['latest', 'best']:
                    order_by = '-created' if self.runtime_args.resume_from == 'latest' else '+loss'
                    prev_checkpoints = SwCheckpoint.objects(
                        frame_sequence=self.FS_db,
                        sim_params=self.sim_params,
                        reg_params=self.reg_params
                    ).order_by(order_by)
                    if prev_checkpoints.count() > 0:
                        logger.info(
                            f'Found {prev_checkpoints.count()} previous checkpoints. '
                            f'Using {self.runtime_args.resume_from}.'
                        )
                        prev_checkpoint = prev_checkpoints[0]
                    else:
                        logger.error(
                            f'Found no checkpoints for FS={self.FS_db.id}, sim={self.sim_params.id} and reg={self.reg_params.id}'
                        )
                        raise DoesNotExist()
                else:
                    prev_checkpoint = Checkpoint.objects.get(
                        id=self.runtime_args.resume_from
                    )
                logger.info(f'Loaded checkpoint id={prev_checkpoint.id}, created={prev_checkpoint.created}')
                logger.info(f'Loss = {prev_checkpoint.loss:.6f}')
                for key, val in prev_checkpoint.metrics.items():
                    logger.info(f'\t{key}: {val:.4E}')
            except DoesNotExist:
                raise RuntimeError(f'Could not load checkpoint={self.runtime_args.resume_from}')

        # Either clone the previous checkpoint to use as the starting point
        if prev_checkpoint is not None:
            checkpoint = prev_checkpoint.clone()

            # Update the simulation and regularisation references to the ones now in use
            if checkpoint.sim_params.id != self.sim_params.id:
                logger.warning('Simulation parameters have changed! This may cause problems!')
                checkpoint.sim_params = self.sim_params
            if checkpoint.reg_params.id != self.reg_params.id:
                logger.warning('Regularisation parameters have changed! This may cause problems!')
                checkpoint.reg_params = self.reg_params

            # Args are stored against the checkpoint, so just override them
            checkpoint.frame_sequence_args = to_dict(self.frame_sequence_args)
            checkpoint.optimiser_args = to_dict(self.optimiser_args)
            checkpoint.runtime_args = to_dict(self.runtime_args)
            checkpoint.sim_args = to_dict(self.simulation_args)
            checkpoint.reg_args = to_dict(self.regularisation_args)

            # Load the parameter states
            runs = SwRun.objects(checkpoint=prev_checkpoint)
            if runs.count() != self.optimiser_args.batch_size:
                raise RuntimeError(f'Number of runs matching checkpoint "{prev_checkpoint.id}" ({runs.count()}) '
                                   f'not equal to batch size ({self.optimiser_args.batch_size}). Unable to resume.')
            self.F0 = FrameBatchTorch.from_list([
                FrameTorch(
                    x=torch.from_numpy(run.F0.x),
                    psi=torch.from_numpy(run.F0.psi)
                )
                for run in runs
            ], optimise=self.optimiser_args.optimise_F0)
            self.CS = ControlSequenceBatchTorch.from_list([
                ControlSequenceTorch(
                    alpha=torch.from_numpy(run.CS.alpha),
                    beta=torch.from_numpy(run.CS.beta),
                    gamma=torch.from_numpy(run.CS.gamma)
                )
                for run in runs
            ], optimise=self.optimiser_args.optimise_CS)
            logger.info(f'Loaded batch state from {runs.count()} runs.')

        # ..or start a new checkpoint
        else:
            checkpoint = SwCheckpoint(
                frame_sequence=self.FS_db,
                sim_params=self.sim_params,
                reg_params=self.reg_params,
                frame_sequence_args=to_dict(self.frame_sequence_args),
                optimiser_args=to_dict(self.optimiser_args),
                runtime_args=to_dict(self.runtime_args),
                sim_args=to_dict(self.simulation_args),
                reg_args=to_dict(self.regularisation_args),
            )

        return checkpoint

    def _init_tb_logger(self):
        """
        Initialise the tensorboard writers - one for each batch item plus one master.
        """
        self.tb_logger_main = SummaryWriter(self.logs_path + f'/events/{START_TIMESTAMP}_agg', flush_secs=5)
        self.tb_loggers = [
            SummaryWriter(self.logs_path + f'/events/{START_TIMESTAMP}_{idx:03d}', flush_secs=5)
            for idx in range(self.optimiser_args.batch_size)
        ]

    def configure_paths(self, renew_logs: bool = False):
        """Create the directories."""
        if renew_logs:
            logger.warning('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path + '/events', exist_ok=True)
        os.makedirs(self.logs_path + '/plots', exist_ok=True)

    def save_checkpoint(self):
        """
        Save the checkpoint information and run output to the database.
        """
        logger.info('Saving checkpoint and run record...')
        self.checkpoint.save()

        # Create the run record of inputs and optimal outputs for the most recent runs.
        for idx in range(self.optimiser_args.batch_size):
            run = SwRun()
            run.checkpoint = self.checkpoint
            run.sim_params = self.sim_params
            run.frame_sequence = self.FS_db
            run.F0 = SwFrameSequence()
            run.F0.x = to_numpy(self.F0[idx].x)
            run.F0.psi = to_numpy(self.F0[idx].psi)
            run.CS = SwControlSequence()
            run.CS.alpha = to_numpy(self.CS[idx].alpha)
            run.CS.beta = to_numpy(self.CS[idx].beta)
            run.CS.gamma = to_numpy(self.CS[idx].gamma)
            run.FS = SwFrameSequence()
            run.FS.x = to_numpy(self.FS_out[idx].x)
            run.FS.psi = to_numpy(self.FS_out[idx].psi)
            run.save()

        # Replace the current checkpoint-buffer with a clone of the just-saved checkpoint.
        self.checkpoint = self.checkpoint.clone()

    def train(self, n_steps: int):
        """
        Run the inverse optimisation.
        """
        self.configure_paths()
        self._init_tb_logger()
        start_step = self.checkpoint.step
        final_step = start_step + n_steps - 1

        # Initial plots
        self._make_plots(pre_step=True)

        for step in range(start_step, final_step + 1):
            start_time = time.time()
            L = self._train_step()
            time_per_step = time.time() - start_time
            seconds_left = float((final_step - step) * time_per_step)
            logger.info(
                f'[{step + 1}/{final_step + 1}]. '
                f'Total loss = {L.total.mean():.5E}. '
                f'Data loss = {L.data.mean():.5e}. '
                f'Regularisation loss = {L.reg.mean():.5e}. '
                f'Time taken: {timedelta(seconds=time_per_step)}. '
                f'Est. complete in: {timedelta(seconds=seconds_left)}.'
            )

            if self.runtime_args.checkpoint_every_n_steps > 0 \
                    and (step + 1) % self.runtime_args.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

    def _train_step(self) -> LossesTorch:
        """
        Run a single inverse optimisation step.
        """

        # Make target pseudo-batch
        FS_target_batch = FrameSequenceBatchTorch.from_list([self.FS_target] * self.optimiser_args.batch_size)

        # Forward simulation
        FS, L, F0_opt, CS_opt, FS_opt, L_opt = self.worm.forward(
            F0=self.F0,
            CS=self.CS,
            calculate_inverse=True,
            FS_target=FS_target_batch
        )

        # Update the inputs/outputs to the found optimals
        self.FS_out = FS_opt
        self.FS_outs.append(FS_opt)
        self.FS_labels.append(f'X_{self.checkpoint.step}')

        # Update F0 and CS to the found optimals
        if self.optimiser_args.optimise_F0:
            self.F0 = F0_opt
        if self.optimiser_args.optimise_CS:
            self.CS = CS_opt

        # Increment step counter
        self.checkpoint.step += 1

        # Log losses and make plots
        self.checkpoint.loss = float(L.total.mean())
        self._log_losses(L)
        self._make_plots()

        return L

    def _log_losses(self, L: LossesTorch):
        """
        Log the losses to tensorboard.
        The main tb_logger logs means and variances of each metric.
        The other loggers log the metrics for each item in the batch.
        """
        step = self.checkpoint.step

        # Add aggregate losses - means and variances of each key
        self.tb_logger_main.add_scalar('loss_batch/total/mean', L.total.mean(), step)
        self.tb_logger_main.add_scalar('loss_batch/total/var', L.total.var(), step)
        self.tb_logger_main.add_scalar('loss_batch/data/mean', L.data.mean(), step)
        self.tb_logger_main.add_scalar('loss_batch/data/var', L.data.var(), step)
        self.tb_logger_main.add_scalar('loss_batch/reg/mean', L.reg.mean(), step)
        self.tb_logger_main.add_scalar('loss_batch/reg/var', L.reg.var(), step)
        for loss in REG_LOSS_TYPES:
            for k in REG_LOSS_VARS:
                self.tb_logger_main.add_scalar(
                    f'reg_weighted_batch/{loss}/{k}/mean',
                    L.reg_losses_weighted[loss][k].mean(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_weighted_batch/{loss}/{k}/var',
                    L.reg_losses_weighted[loss][k].var(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_unweighted_batch/{loss}/{k}/mean',
                    L.reg_losses_unweighted[loss][k].mean(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_unweighted_batch/{loss}/{k}/var',
                    L.reg_losses_unweighted[loss][k].var(),
                    step
                )

        # Log individual losses for each item in the batch
        for idx in range(self.optimiser_args.batch_size):
            tbl = self.tb_loggers[idx]
            Li = L[idx]
            tbl.add_scalar('loss/total', Li.total, step)
            tbl.add_scalar('loss/data', Li.data, step)
            tbl.add_scalar('loss/reg', Li.reg, step)
            for loss in REG_LOSS_TYPES:
                for k in REG_LOSS_VARS:
                    tbl.add_scalar(
                        f'reg_weighted/{loss}/{k}',
                        Li.reg_losses_weighted[loss][k],
                        step
                    )
                    tbl.add_scalar(
                        f'reg_unweighted/{loss}/{k}',
                        Li.reg_losses_unweighted[loss][k],
                        step
                    )

    def _make_plots(self, pre_step: bool = False, final_step: bool = False):
        """
        Generate some example plots and videos.
        """

        # Select the idxs to plot
        bs = self.optimiser_args.batch_size
        n_examples = min(self.runtime_args.plot_n_examples, bs)
        idxs = np.random.choice(bs, n_examples, replace=False)

        if pre_step:
            for idx in idxs:
                self._plot_F0_components(idx)
                self._plot_F0_3d(idx)
                self._plot_CS(idx)
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and (self.checkpoint.step + 1) % self.runtime_args.plot_every_n_steps == 0
        ):
            for idx in idxs:
                self._plot_X(idx)
                self._plot_F0_components(idx)
                self._plot_F0_3d(idx)
                self._plot_CS(idx)

        if final_step or (
                self.runtime_args.videos_every_n_steps > -1
                and (self.checkpoint.step + 1) % self.runtime_args.videos_every_n_steps == 0
        ):
            for idx in idxs:
                # self._plot_FS_3d(idx)
                self._make_diff_vids(idx)

    def _plot_X(self, idx: int):
        """
        Plot the x,y,z midline positions over time as matrices.
        """
        fig = plot_X_vs_target(
            FS=self.FS_out[idx].to_numpy(),
            FS_target=self.FS_target.to_numpy()
        )
        self._save_plot(fig, f'X/{idx:03d}')

    def _plot_F0_components(self, idx: int):
        """
        Plot the psi/e0/e1/e2 frame components as matrices.
        """
        fig = plot_frame_components(
            F=self.F0[idx].to_numpy(worm=self.worm.worm_solver, calculate_components=True)
        )
        self._save_plot(fig, f'F0/{idx:03d}')

    def _plot_F0_3d(self, idx: int):
        """
        Plot a 3x3 grid of 3D plots of the same worm frame from different angles.
        """
        fig = plot_frame_3d(
            F0=self.F0[idx].to_numpy(worm=self.worm.worm_solver, calculate_components=True)
        )
        self._save_plot(fig, f'F0_3D/{idx:03d}')

    def _plot_CS(self, idx: int):
        """
        Plot the control sequences as matrices.
        """
        fig = plot_CS(CS=self.CS[idx].to_numpy())
        self._save_plot(fig, f'CS/{idx:03d}')

    def _plot_FS_3d(self, idx: int):
        fig = plot_FS_3d(
            FSs=[self.FS_out[idx].to_numpy(), self.FS_target.to_numpy()],
            CSs=[self.CS[idx].to_numpy(), None],
            labels=['Attempt', 'Target']
        )
        self._save_plot(fig, f'3D/{idx:03d}')

    def _save_plot(self, fig: Figure, plot_type: str):
        """
        Log the figure to the tensorboard logger and optionally save it to disk.
        """
        if self.runtime_args.save_plots:
            save_dir = self.logs_path + f'/plots/{plot_type}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir + f'/{self.checkpoint.step:05d}.svg'
            plt.savefig(path, bbox_inches='tight')

        self.tb_logger_main.add_figure(plot_type, fig, self.checkpoint.step)
        self.tb_logger_main.flush()

        plt.close(fig)

    def _make_diff_vids(self, idx: int):
        """
        Generate a video clip showing a target sequence, an attempt and the difference.
        """
        logger.info(f'Generating scatter diff clip.')
        generate_scatter_diff_clip(
            FS_target=self.FS_target.to_numpy(),
            FS_attempt=self.FS_out[idx].to_numpy(),
            save_dir=self.logs_path + f'/vid_diffs/{idx:03d}',
            save_fn=str(self.checkpoint.step),
            n_arrows=12,
        )
