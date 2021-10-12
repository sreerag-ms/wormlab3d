import math
import os
import shutil
import time
from collections import OrderedDict
from datetime import timedelta
from typing import Tuple, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from mongoengine import DoesNotExist
from simple_worm.control_gates_torch import ControlGateTorch
from simple_worm.controls import CONTROL_KEYS
from simple_worm.controls_torch import ControlSequenceTorch, ControlSequenceBatchTorch
from simple_worm.frame_torch import FrameTorch, FrameSequenceBatchTorch, FrameSequenceTorch, FrameBatchTorch
from simple_worm.losses import REG_LOSS_TYPES, REG_LOSS_VARS
from simple_worm.losses_torch import LossesTorch
from simple_worm.material_parameters import MP_KEYS
from simple_worm.material_parameters_torch import MaterialParametersTorch, MaterialParametersBatchTorch
from simple_worm.plot3d import plot_X_vs_target, plot_FS_3d, generate_scatter_diff_clip, plot_CS, plot_frame_3d, \
    plot_frame_components, plot_CS_vs_output, plot_gates
from simple_worm.util_torch import expand_tensor
from simple_worm.worm_torch import WormModule
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from wormlab3d import LOGS_PATH, logger
from wormlab3d.data.model import FrameSequence, Trial
from wormlab3d.data.model.sw_checkpoint import SwCheckpoint
from wormlab3d.data.model.sw_regularisation_parameters import SwRegularisationParameters
from wormlab3d.data.model.sw_run import SwRun, SwControlSequence, SwFrameSequence, SwMaterialParameters
from wormlab3d.data.model.sw_simulation_parameters import SwSimulationParameters
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.simple_worm.args import RuntimeArgs, FrameSequenceArgs, SimulationArgs, OptimiserArgs, RegularisationArgs
from wormlab3d.toolkit.util import to_dict, to_numpy, hash_data, is_bad

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
        self.FS_db, self.FS_target, self.FS_target_chunks, self.FS_target_chunk_steps = self._init_frame_sequence()

        # Initialise configurations
        self.sim_params, self.reg_params = self._init_configuration()

        # Initialise worm simulator(s)
        self.worm, self.worm_stitched = self._init_worms()

        # Initialise initial conditions and trainable parameters
        self.MP, self.F0, self.CS, self.CS_stitched = self._init_params()

        # Checkpoints
        self.checkpoint = self._init_checkpoint()

        # Outputs
        self.FS_out: FrameSequenceTorch
        self.FS_out_stitched: FrameSequenceTorch
        self.MP_log: Dict[int, MaterialParametersBatchTorch] = OrderedDict()

    @property
    def logs_path(self) -> str:
        return self.get_logs_path(self.checkpoint)

    @property
    def batch_size(self) -> int:
        if self.optimiser_args.chunked_mode:
            return self.optimiser_args.n_chunks
        else:
            return self.optimiser_args.batch_size

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

        solver_dir = f'/{checkpoint.optimiser_args["inverse_opt_library"]}' \
                     f'_{checkpoint.optimiser_args["inverse_opt_method"]}' \
                     f'_{hash_data(checkpoint.optimiser_args["inverse_opt_opts"])}'

        if 'chunked_mode' in checkpoint.optimiser_args and checkpoint.optimiser_args['chunked_mode']:
            solver_dir += f'_chunks={checkpoint.optimiser_args["n_chunks"]}'

        return LOGS_PATH + sim_dir + FS_dir + regs_dir + solver_dir

    def _init_frame_sequence(self) \
            -> Tuple[FrameSequence, FrameSequenceTorch, List[FrameSequenceTorch], Optional[np.ndarray]]:
        """
        Load or create the target frame sequence and the chunks if required.
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

        # Create the simulation target frame sequence (leaving the first frame for F0)
        x = torch.from_numpy(FS_db.X[1:])

        # Resample the worm points if required
        duration = self.simulation_args.duration
        dt = self.simulation_args.dt
        n_frames = int(duration / dt)
        n_frames_data = x.shape[0]
        N = self.simulation_args.worm_length
        N_data = x.shape[1]
        if n_frames != n_frames_data or N != N_data:
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = F.interpolate(x, size=(n_frames, N), mode='bilinear', align_corners=True)
            x = x.squeeze(0).permute(1, 2, 0)

        # Permute dimensions for compatibility with simple-worm
        x = x.permute(0, 2, 1)
        assert x.shape == (n_frames, 3, N)
        FS = FrameSequenceTorch(x=x)

        # Chunk the FS if required
        FS_chunks = []
        FS_chunk_steps = []
        if self.optimiser_args.chunked_mode:
            n_chunks = self.optimiser_args.n_chunks
            chunk_duration = duration / n_chunks
            overlap = max(dt * 2, chunk_duration * 0.1)
            n_frames_chunk = int((chunk_duration + overlap) / dt)
            timesteps = np.arange(0, duration, dt)
            start_times = list(np.linspace(0, duration - chunk_duration, n_chunks) - overlap / 2)
            start_idxs = []
            for i, t in enumerate(timesteps):
                if t >= start_times[0] or i + n_frames_chunk == n_frames:
                    start_times.pop(0)
                    start_idxs.append(i)
                    if len(start_times) == 0:
                        break
            start_idxs = np.array(start_idxs)
            end_idxs = start_idxs + n_frames_chunk

            for c in range(n_chunks):
                FS_chunks.append(FS[start_idxs[c]:end_idxs[c]])
                if c > 0:
                    assert start_idxs[c] < end_idxs[c - 1]
                    assert len(FS_chunks[c]) == len(FS_chunks[c - 1])

            FS_chunk_steps = np.array([start_idxs, end_idxs]).T

        return FS_db, FS, FS_chunks, FS_chunk_steps

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
            n_frames = math.ceil(self.simulation_args.duration * trial.fps) + 1
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
        fsa = self.frame_sequence_args
        trial = Trial.objects.get(id=fsa.trial_id)
        n_frames = math.ceil(self.simulation_args.duration * trial.fps) + 1
        f0 = fsa.start_frame
        fn = f0 + n_frames
        frame_nums = range(f0, fn)
        seq = []
        for frame_num in frame_nums:
            frame = trial.get_frame(frame_num)
            logger.debug(f'Loading 3D midline for frame #{frame_num} (id={frame.id}).')
            filters = {'source': fsa.midline_source}
            if fsa.midline_source_file is not None:
                filters['source_file'] = fsa.midline_source_file
            midlines = frame.get_midlines3d(filters)
            if len(midlines) > 1:
                logger.info(
                    f'Found {len(midlines)} 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}. Picking with lowest error..')
                midline = midlines[0]
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

            # Update duration to the chunk duration when in chunked mode
            if self.optimiser_args.chunked_mode:
                sim_config['duration'] = self.FS_target_chunks[0].n_frames * self.simulation_args.dt

            sim_params = SwSimulationParameters.objects(**sim_config)
            if sim_params.count() > 0:
                logger.info(
                    f'Found {sim_params.count()} matching simulation parameter records in database, using most recent.')
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

    def _init_worms(self) -> Union[WormModule, Optional[WormModule]]:
        """
        Create the worm simulators.
        """
        oa = self.optimiser_args
        worm = WormModule(
            N=self.sim_params.worm_length,
            dt=self.sim_params.dt,
            batch_size=self.batch_size,
            **oa.get_mp_opt_flags('optimise_MP_'),
            optimise_F0=oa.optimise_F0,
            optimise_CS=oa.optimise_CS,
            reg_weights=self.regularisation_args.get_reg_weights(),
            max_alpha_beta=oa.max_alpha_beta,
            max_gamma=oa.max_gamma,
            inverse_opt_library=oa.inverse_opt_library,
            inverse_opt_method=oa.inverse_opt_method,
            inverse_opt_max_iter=oa.inverse_opt_max_iter,
            inverse_opt_tol=oa.inverse_opt_tol,
            inverse_opt_opts=oa.inverse_opt_opts,
            mkl_threads=oa.mkl_threads,
            parallel=self.runtime_args.parallel_solvers > 0,
            n_workers=self.runtime_args.parallel_solvers,
            quiet=False
        )

        worm_stitched = None
        if oa.chunked_mode:
            worm_stitched = WormModule(
                N=self.sim_params.worm_length,
                dt=self.sim_params.dt,
                batch_size=1,
                optimise_F0=False,
                optimise_CS=False,
                reg_weights=self.regularisation_args.get_reg_weights(),
                quiet=False
            )

        return worm, worm_stitched

    def _init_params(self) \
            -> Tuple[
                MaterialParametersBatchTorch,
                FrameBatchTorch,
                ControlSequenceBatchTorch,
                Optional[ControlSequenceBatchTorch]
            ]:
        """
        Initialise the optimisable initial frame and control sequence.
        """
        oa = self.optimiser_args

        # Generate optimisable material parameters
        MP = MaterialParametersTorch(**self.simulation_args.get_mp_dict())
        MP = MaterialParametersBatchTorch.from_list(
            batch=[MP] * self.batch_size,
            **oa.get_mp_opt_flags('optimise_')
        )

        # Generate the NaturalFrame sequence to approximate twist and curvatures
        NFS_target: List[NaturalFrame] = []
        NFS_target_chunks: List[List[NaturalFrame]] = []
        NF_prev: List[NaturalFrame] = []
        buffer_size = 20
        for F in self.FS_target:
            NF = NaturalFrame(X=F.x.numpy().T)

            # Correct any sign flips
            if len(NF_prev) > 0:
                m1_prev = np.mean([p.m1 for p in NF_prev], axis=0)
                m2_prev = np.mean([p.m2 for p in NF_prev], axis=0)

                # Check distances from previous frame to all pi/2 rotations of current frame
                d0 = np.sum((NF.m1 - m1_prev)**2 + (NF.m2 - m2_prev)**2)
                d1 = np.inf  # np.sum((NF.m1 + m2_prev)**2 +(NF.m2 - m1_prev)**2)
                d2 = np.sum((NF.m1 + m1_prev)**2 + (NF.m2 + m2_prev)**2)
                d3 = np.inf  # np.sum((NF.m1 - m2_prev)**2 +(NF.m2 + m1_prev)**2)

                # If the distance is smallest to a rotated frame then rotate the frame to fix
                closest_rotation_idx = np.argmin(np.array([d0, d1, d2, d3]))
                if closest_rotation_idx != 0:
                    if closest_rotation_idx == 1:
                        NF.m1 = -NF.m2
                        NF.m2 = NF.m1
                        NF.psi += np.pi / 2
                    elif closest_rotation_idx == 2:
                        NF.m1 = -NF.m1
                        NF.m2 = -NF.m2
                        NF.psi += np.pi
                    elif closest_rotation_idx == 3:
                        NF.m1 = NF.m2
                        NF.m2 = -NF.m1
                        NF.psi += 3 * np.pi / 2

            NFS_target.append(NF)
            NF_prev.append(NF)
            if len(NF_prev) > buffer_size:
                NF_prev.pop(0)

        # Chunk the NaturalFrame sequence if needed
        if oa.chunked_mode:
            for i in range(oa.n_chunks):
                idxs = self.FS_target_chunk_steps[i]
                NFS_target_chunks.append(
                    NFS_target[idxs[0]:idxs[1]]
                )

        # Generate optimisable initial frame, using known x0 and an estimate of psi0
        psi0 = torch.zeros(self.batch_size, self.sim_params.worm_length)
        if oa.chunked_mode:
            x0 = torch.zeros(oa.n_chunks, 3, self.sim_params.worm_length)
            for i in range(oa.n_chunks):
                x0[i] = self.FS_target_chunks[i][0].x
                if oa.estimate_psi:
                    psi0[i] = torch.from_numpy(NFS_target_chunks[i][0].psi)
        else:
            x0 = expand_tensor(self.FS_target[0].x, self.batch_size)
            if oa.estimate_psi:
                psi0[:] = torch.from_numpy(NFS_target[0].psi)
        F0 = FrameBatchTorch(
            x=x0,
            psi=psi0,
            estimate_psi=False,  # Use the NF estimate of psi if required
            optimise=oa.optimise_F0,
            batch_size=self.batch_size
        )

        # Build control gates
        gates = {}
        gate_arg_keys = ['block', 'grad_up', 'offset_up', 'grad_down', 'offset_down']
        for k in CONTROL_KEYS:
            gate_args = {
                gak: getattr(self.simulation_args, f'{k}_gate_{gak}')
                for gak in gate_arg_keys
            }
            if any([v is not None for v in gate_args.values()]):
                gates[f'{k}_gate'] = ControlGateTorch(N=self.sim_params.worm_length, **gate_args)

        # Generate optimisable control sequence
        if oa.chunked_mode:
            n_timesteps = self.FS_target_chunks[0].n_frames
            CS_stitched = ControlSequenceBatchTorch(
                worm=self.worm_stitched.worm_solver,
                n_timesteps=self.FS_target.n_frames,
                batch_size=1,
                **gates
            )
        else:
            n_timesteps = self.FS_target.n_frames
            CS_stitched = None
        CS = ControlSequenceBatchTorch(
            worm=self.worm.worm_solver,
            n_timesteps=n_timesteps,
            optimise=oa.optimise_CS,
            batch_size=self.batch_size,
            **gates
        )

        if oa.estimate_CS:
            with torch.no_grad():
                for i in range(self.batch_size):
                    if oa.chunked_mode:
                        src = NFS_target_chunks[i]
                    else:
                        src = NFS_target
                    CS.controls['alpha'][i] = torch.tensor([F.m1 for F in src])
                    CS.controls['beta'][i] = torch.tensor([F.m2 for F in src])

        # Add some noise
        with torch.no_grad():
            for k in MP_KEYS:
                opt_mp = getattr(oa, f'optimise_MP_{k}')
                std = getattr(oa, f'init_noise_std_{k}')
                if opt_mp and std > 0:
                    v = getattr(MP, k)
                    v.data += torch.normal(torch.zeros_like(v), std=std)
            MP.clamp()

            if oa.optimise_F0 and oa.init_noise_std_psi0 > 0:
                F0.psi += torch.normal(torch.zeros_like(F0.psi), std=oa.init_noise_std_psi0)

            if oa.optimise_CS:
                for abg in CONTROL_KEYS:
                    std = getattr(oa, f'init_noise_std_{abg}')
                    if std > 0:
                        v = getattr(CS, abg)
                        v.data += torch.normal(torch.zeros_like(v), std=std)

        return MP, F0, CS, CS_stitched

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
                    prev_checkpoint = SwCheckpoint.objects.get(
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
            if runs.count() != self.batch_size:
                raise RuntimeError(f'Number of runs matching checkpoint "{prev_checkpoint.id}" ({runs.count()}) '
                                   f'not equal to batch size ({self.batch_size}). Unable to resume.')

            self.MP = MaterialParametersBatchTorch.from_list(
                batch=[
                    run.MP.get_material_parameters()
                    for run in runs
                ],
                **self.optimiser_args.get_mp_opt_flags(prefix='optimise_')
            )

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
                    gamma=torch.from_numpy(run.CS.gamma),
                    **self.CS.get_gates('clone')
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
            for idx in range(self.batch_size)
        ]

    def configure_paths(self, renew_logs: bool = False):
        """Create the directories."""
        if renew_logs:
            logger.warning('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.logs_path + '/events', exist_ok=True)
        os.makedirs(self.logs_path + '/plots', exist_ok=True)

    def save_checkpoint(self, L: LossesTorch):
        """
        Save the checkpoint information and run output to the database.
        """
        logger.info('Saving checkpoint and run records...')
        self.checkpoint.save()

        # Create the run record of inputs and optimal outputs for the most recent runs.
        for idx in range(self.batch_size):
            run = SwRun()
            run.checkpoint = self.checkpoint
            run.sim_params = self.sim_params
            run.frame_sequence = self.FS_db
            run.loss = L[idx].total
            run.loss_data = L[idx].data
            run.loss_reg = L[idx].reg
            run.reg_losses = L[idx].reg_losses_unweighted
            run.MP = SwMaterialParameters()
            run.MP.K = float(self.MP[idx].K)
            run.MP.K_rot = float(self.MP[idx].K_rot)
            run.MP.A = float(self.MP[idx].A)
            run.MP.B = float(self.MP[idx].B)
            run.MP.C = float(self.MP[idx].C)
            run.MP.D = float(self.MP[idx].D)
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
                self.save_checkpoint(L)

    def _train_step(self) -> LossesTorch:
        """
        Run a single inverse optimisation step.
        """
        oa = self.optimiser_args

        if oa.chunked_mode:
            # Make target batches
            FS_target_batch = FrameSequenceBatchTorch.from_list(self.FS_target_chunks)
        else:
            # Make target pseudo-batch (all the same)
            FS_target_batch = FrameSequenceBatchTorch.from_list([self.FS_target] * self.batch_size)

        # Forward simulation
        logger.info('Simulating...')
        FS, L, MP_opt, F0_opt, CS_opt, FS_opt, L_opt = self.worm.forward(
            MP=self.MP,
            F0=self.F0,
            CS=self.CS,
            calculate_inverse=True,
            FS_target=FS_target_batch
        )

        # Update MP, F0 and CS to the found optimals
        for k in MP_KEYS:
            if getattr(oa, f'optimise_MP_{k}'):
                setattr(self.MP, k, getattr(MP_opt, k))
        self.MP.clamp()

        if oa.optimise_F0:
            assert not is_bad(F0_opt.psi)
            self.F0 = F0_opt
        if oa.optimise_CS:
            for k in CONTROL_KEYS:
                assert not is_bad(CS_opt.controls[k])
            self.CS = CS_opt

        # Stitch together the chunks
        if oa.chunked_mode:
            # Update overlapped regions to the average from both sides
            n_chunks = oa.n_chunks
            for i in range(n_chunks - 1):
                for k in CONTROL_KEYS:
                    controls = self.CS.controls[k]
                    cs = self.FS_target_chunk_steps

                    # From where the next one starts
                    from_idx = cs[i + 1][0] - cs[i][1]
                    prev_chunk_overlap = controls[i, from_idx:]

                    # To where the previous one ends
                    to_idx = cs[i][1] - cs[i + 1][0]
                    next_chunk_overlap = controls[i + 1, :to_idx]

                    # Set both overlapping sections to the mean
                    assert prev_chunk_overlap.shape == next_chunk_overlap.shape
                    mean_overlap = (prev_chunk_overlap + next_chunk_overlap) / 2
                    self.CS.controls[k][i, from_idx:] = mean_overlap
                    self.CS.controls[k][i + 1, :to_idx] = mean_overlap

                    # Update the stitched controls
                    self.CS_stitched.controls[k][0, cs[i][0]:cs[i][1]] = self.CS.controls[k][i]
                    if i == n_chunks - 2:
                        self.CS_stitched.controls[k][0, cs[i + 1][0]:cs[i + 1][1]] = self.CS.controls[k][i + 1]

            # Use the stitched controls and F0 from the first chunk to generate full output
            logger.info('Simulating stitched...')
            FS_stitched, L_stitched = self.worm_stitched.forward(
                MP=MaterialParametersBatchTorch.from_list([self.MP[0]]),
                F0=FrameBatchTorch.from_list([self.F0[0]]),
                CS=self.CS_stitched,
                calculate_inverse=False,
                FS_target=FrameSequenceBatchTorch.from_list([self.FS_target])
            )
            self.FS_out_stitched = FS_stitched
        else:
            L_stitched = None

        # Log the output
        self.FS_out = FS_opt
        self.MP_log[self.checkpoint.step] = self.MP.clone()

        # Increment step counter
        self.checkpoint.step += 1

        # Log losses and make plots
        self.checkpoint.loss = float(L.total.mean())
        self.checkpoint.loss_data = float(L.data.mean())
        self._log_losses(L, L_stitched)
        self._make_plots()

        return L

    def _log_losses(self, L: LossesTorch, L_stitched: LossesTorch = None):
        """
        Log the losses to tensorboard.
        The main tb_logger logs means and variances of each metric and stitched outputs.
        The other loggers log the metrics for each item in the batch.
        """
        step = self.checkpoint.step
        bk = 'chunk' if self.optimiser_args.chunked_mode else 'batch'

        # Add aggregate losses - means and variances of each key
        self.tb_logger_main.add_scalar(f'loss_{bk}/total/mean', L.total.mean(), step)
        self.tb_logger_main.add_scalar(f'loss_{bk}/total/var', L.total.var(), step)
        self.tb_logger_main.add_scalar(f'loss_{bk}/data/mean', L.data.mean(), step)
        self.tb_logger_main.add_scalar(f'loss_{bk}/data/var', L.data.var(), step)
        self.tb_logger_main.add_scalar(f'loss_{bk}/reg/mean', L.reg.mean(), step)
        self.tb_logger_main.add_scalar(f'loss_{bk}/reg/var', L.reg.var(), step)
        for loss in REG_LOSS_TYPES:
            for k in REG_LOSS_VARS:
                self.tb_logger_main.add_scalar(
                    f'reg_weighted_{bk}/{loss}/{k}/mean',
                    L.reg_losses_weighted[loss][k].mean(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_weighted_{bk}/{loss}/{k}/var',
                    L.reg_losses_weighted[loss][k].var(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_unweighted_{bk}/{loss}/{k}/mean',
                    L.reg_losses_unweighted[loss][k].mean(),
                    step
                )
                self.tb_logger_main.add_scalar(
                    f'reg_unweighted_{bk}/{loss}/{k}/var',
                    L.reg_losses_unweighted[loss][k].var(),
                    step
                )

        # Log individual losses for each item in the batch
        for idx in range(self.batch_size):
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

        # Log stitched losses
        if L_stitched is not None:
            tbl = self.tb_logger_main
            tbl.add_scalar('loss_stitched/total', L_stitched.total.mean(), step)
            tbl.add_scalar('loss_stitched/data', L_stitched.data.mean(), step)
            tbl.add_scalar('loss_stitched/reg', L_stitched.reg, step)
            for loss in REG_LOSS_TYPES:
                for k in REG_LOSS_VARS:
                    tbl.add_scalar(
                        f'reg_weighted/{loss}/{k}',
                        L_stitched.reg_losses_weighted[loss][k],
                        step
                    )
                    tbl.add_scalar(
                        f'reg_unweighted/{loss}/{k}',
                        L_stitched.reg_losses_unweighted[loss][k],
                        step
                    )

    def _make_plots(self, pre_step: bool = False, final_step: bool = False):
        """
        Generate some example plots and videos.
        """
        logger.info('Plotting.')

        # Select the idxs to plot
        bs = self.batch_size
        n_examples = min(self.runtime_args.plot_n_examples, bs)
        idxs = np.random.choice(bs, n_examples, replace=False)

        # Make initial plots for all batch elements
        if pre_step:
            for idx in range(bs):
                self._plot_F0_components(idx)
                self._plot_F0_3d(idx)
            self._plot_gates()
            self._plot_pca()
            return

        if final_step or (
                self.runtime_args.plot_every_n_steps > -1
                and (self.checkpoint.step + 1) % self.runtime_args.plot_every_n_steps == 0
        ):
            for idx in idxs:
                self._plot_X(idx)
                self._plot_F0_components(idx)
                self._plot_F0_3d(idx)
                # self._plot_CS(idx)
                self._plot_CS_vs_output(idx)
            self._plot_MPs()
            self._plot_pca()

            if self.optimiser_args.chunked_mode:
                self._plot_X_stitched()
                self._plot_CS_vs_output_stitched()

        if final_step or (
                self.runtime_args.videos_every_n_steps > -1
                and (self.checkpoint.step + 1) % self.runtime_args.videos_every_n_steps == 0
        ):
            for idx in idxs:
                # self._plot_FS_3d(idx)
                self._make_diff_vids(idx)

            if self.optimiser_args.chunked_mode:
                self._make_diff_vids_stitched()

    def _plot_gates(self):
        """
        Plot the control gates. These are the same across the batch and fixed for the duration.
        """
        fig = plot_gates(self.CS)

        # Save plot regardless of setting and don't use tensorboard
        save_dir = self.logs_path + f'/plots'
        path = save_dir + f'/gates.svg'
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def _plot_X(self, idx: int):
        """
        Plot the x,y,z midline positions over time as matrices.
        """
        if self.optimiser_args.chunked_mode:
            FS_target = self.FS_target_chunks[idx]
        else:
            FS_target = self.FS_target

        fig = plot_X_vs_target(
            FS=self.FS_out[idx].to_numpy(),
            FS_target=FS_target.to_numpy()
        )
        self._save_plot(fig, f'X/{idx:03d}')

    def _plot_X_stitched(self):
        """
        Plot the stitched x,y,z midline positions over time as matrices.
        """
        fig = plot_X_vs_target(
            FS=self.FS_out_stitched[0].to_numpy(),
            FS_target=self.FS_target.to_numpy()
        )
        self._save_plot(fig, f'X/stitched')

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
        fig = plot_CS(CS=self.CS[idx].to_numpy(), dt=self.sim_params.dt)
        self._save_plot(fig, f'CS/{idx:03d}')

    def _plot_CS_vs_output(self, idx: int):
        """
        Plot the control sequences as matrices.
        """
        fig = plot_CS_vs_output(
            CS=self.CS[idx].to_numpy(),
            FS=self.FS_out[idx].to_numpy(),
            dt=self.sim_params.dt,
            show_ungated=True
        )
        self._save_plot(fig, f'CS_vs_out/{idx:03d}')

    def _plot_CS_vs_output_stitched(self):
        """
        Plot the stitched control sequences as matrices.
        """
        fig = plot_CS_vs_output(
            CS=self.CS_stitched[0].to_numpy(),
            FS=self.FS_out_stitched[0].to_numpy(),
            dt=self.sim_params.dt,
            show_ungated=True
        )
        self._save_plot(fig, f'CS_vs_out/stitched')

    def _plot_FS_3d(self, idx: int):
        fig = plot_FS_3d(
            FSs=[self.FS_out[idx].to_numpy(), self.FS_target.to_numpy()],
            CSs=[self.CS[idx].to_numpy(), None],
            labels=['Attempt', 'Target']
        )
        self._save_plot(fig, f'3D/{idx:03d}')

    def _plot_pca(self):
        bs = self.batch_size

        # Can't do PCA when not running in batch mode
        if bs == 1:
            return

        # Convert controls to matrix form
        solutions = to_numpy(torch.cat([
            self.F0.psi,
            self.CS.alpha.reshape(bs, -1),
            self.CS.beta.reshape(bs, -1),
            self.CS.gamma.reshape(bs, -1)
        ], dim=1))

        # PCA
        pca = PCA(svd_solver='randomized', copy=False)
        embeddings = pca.fit_transform(solutions)

        # tSNE projections
        tsne = TSNE(n_components=2)
        tsne_projections = tsne.fit_transform(embeddings)

        # Make plot
        fig, axes = plt.subplots(3, figsize=(10, 10))
        cmap = plt.get_cmap('rainbow')

        # Show the overall distribution of singular values
        ind = np.arange(pca.n_components_)
        ax = axes[0]
        ax.bar(ind, pca.singular_values_, align='center')
        ax.set_xticks(ind)
        ax.set_title(f'PCA on solutions (n_components={pca.n_components_})')
        ax.set_xlabel('Singular value')

        # Show the distributions for the samples
        ax = axes[1]
        for i, embedding in enumerate(embeddings):
            ax.plot(embedding, color=cmap((i + 0.5) / bs), label=i)
        if bs < 10:
            ax.legend()
        ax.set_title('Sample distribution')
        ax.set_xlabel('Singular value')
        ax.set_ylabel('Contribution')

        # Show the tSNE scatter plot
        ax = axes[2]
        ax.set_title('tSNE embedding')
        fc = cmap((np.arange(bs) + 0.5) / bs)
        ax.scatter(tsne_projections[:, 0], tsne_projections[:, 1], c=fc)

        fig.tight_layout()
        self._save_plot(fig, 'PCA')

    def _plot_MPs(self):
        """
        Plot the material parameters values over iterations.
        """
        if not any(list(self.optimiser_args.get_mp_opt_flags().values())):
            return
        bs = self.batch_size
        fig, axes = plt.subplots(2, 3, figsize=(12, 10), sharex=True)
        row1_keys = ['K', 'A', 'B']
        row2_keys = ['K_rot', 'C', 'D']
        steps = list(self.MP_log.keys())

        for row_idx, row_keys in enumerate([row1_keys, row2_keys]):
            for col_idx, k in enumerate(row_keys):
                ax = axes[row_idx, col_idx]
                is_opt = getattr(self.optimiser_args, f'optimise_MP_{k}')
                ax.set_title(k + (' (optimising)' if is_opt else ''))
                initial_val = getattr(self.simulation_args, k)
                ax.axhline(y=initial_val, linestyle=':', alpha=0.5, color='grey')
                batch_vals = np.stack([
                    to_numpy(getattr(MP, k))
                    for _, MP in self.MP_log.items()
                ])
                for i in range(bs):
                    ax.plot(steps, batch_vals[:, i], alpha=0.7)

        fig.tight_layout()
        self._save_plot(fig, 'MPs')

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
        logger.info(f'Generating scatter diff clip for idx={idx}.')
        if self.optimiser_args.chunked_mode:
            FS_target = self.FS_target_chunks[idx]
        else:
            FS_target = self.FS_target
        generate_scatter_diff_clip(
            FS_target=FS_target.to_numpy(),
            FS_attempt=self.FS_out[idx].to_numpy(),
            save_dir=self.logs_path + f'/vid_diffs/{idx:03d}',
            save_fn=str(self.checkpoint.step),
            n_arrows=12,
        )

    def _make_diff_vids_stitched(self):
        """
        Generate a video clip showing a target sequence, an attempt and the difference.
        """
        logger.info(f'Generating stitched scatter diff clip.')
        generate_scatter_diff_clip(
            FS_target=self.FS_target.to_numpy(),
            FS_attempt=self.FS_out_stitched[0].to_numpy(),
            save_dir=self.logs_path + f'/vid_diffs/stitched',
            save_fn=str(self.checkpoint.step),
            n_arrows=12,
        )
