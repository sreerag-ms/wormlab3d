import time
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from simple_worm.controls import ControlSequenceNumpy, CONTROL_KEYS
from simple_worm.frame import FrameNumpy, FrameSequenceNumpy, FRAME_KEYS
from simple_worm.worm import Worm
from wormlab3d import logger
from wormlab3d.data.model import SwCheckpoint, SwRun, SwSimulationParameters
from wormlab3d.toolkit.util import parse_target_arguments, to_numpy

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')


def get_run() -> SwRun:
    """
    Find a run by id.
    """
    args = parse_target_arguments()
    if args.sw_run is None:
        raise RuntimeError('This script must be run with the --sw-run=ID argument defined.')
    return SwRun.objects.get(id=args.sw_run)


def timestep_verification():
    """
    Loads a simulation run and recomputes the simulation output for different timesteps.
    """
    dts = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    run = get_run()
    checkpoint: SwCheckpoint = run.checkpoint
    sim_params: SwSimulationParameters = checkpoint.sim_params
    N = sim_params.worm_length
    F0 = FrameNumpy(
        x=run.F0.x,
        psi=run.F0.psi
    )
    CS = ControlSequenceNumpy(
        alpha=run.CS.alpha,
        beta=run.CS.beta,
        gamma=run.CS.gamma
    )

    outputs: List[FrameSequenceNumpy] = []
    for dt in dts:
        logger.info(f'Running simulation with dt={dt}.')
        T = int(sim_params.duration / dt)

        # Resample the controls
        controls = {}
        for x in CONTROL_KEYS:
            c = getattr(CS, x)
            c = torch.from_numpy(c[None, None, :])
            Nc = N - 1 if x == 'gamma' else N
            c = F.interpolate(c, size=(T, Nc), mode='bilinear', align_corners=True)
            c = to_numpy(c.squeeze())
            controls[x] = c
        CSi = ControlSequenceNumpy(**controls)

        # Run simulation
        worm = Worm(N, dt)
        FS = worm.solve(
            T=sim_params.duration,
            MP=sim_params.get_material_parameters(),
            F0=F0.to_fenics(worm),
            CS=CSi.to_fenics(worm)
        )
        outputs.append(FS.to_numpy())

    # Use smallest timestep as ground truth
    FS_target = outputs.pop()
    T = int(sim_params.duration / dts.pop())

    # Calculate relative errors between outputs and target
    errors = {}
    for x in FRAME_KEYS:
        v_target = getattr(FS_target, x)
        errors[x] = {}

        for i, dt in enumerate(dts):
            # Resample the output
            FSi = outputs[i].clone()
            v = getattr(FSi, x)
            if v.ndim == 2:
                v = torch.from_numpy(v[None, None, :])
            elif v.ndim == 3:
                v = torch.from_numpy(v[None, :]).transpose(1, 2)
            else:
                raise RuntimeError(f'Unrecognised shape for {x}!')
            if x == 'gamma':
                size = (T, N - 1)
            else:
                size = (T, N)
            v = F.interpolate(v, size=size, mode='bilinear', align_corners=True)
            v = to_numpy(v.squeeze())

            # Calculate squared errors
            if v.ndim == 3:
                v = v.transpose(1, 0, 2)
                v_err = ((v - v_target)**2).sum(axis=(1, 2))
            else:
                v_err = ((v - v_target)**2).sum(axis=1)

            errors[x][dt] = v_err

    # Make plots
    # interactive()
    fig, axes = plt.subplots(len(FRAME_KEYS), 2, figsize=(16, 3 * len(FRAME_KEYS)), sharex=True)
    fig.suptitle(f'Run = {run.id}')
    for i, x in enumerate(FRAME_KEYS):
        ax = axes[i, 0]
        ax.set_title(x)
        for dt, err in errors[x].items():
            ax.plot(err, label=f'dt={dt}')
        ax.legend()

        ax = axes[i, 1]
        ax.set_title(x)
        for dt, err in errors[x].items():
            ax.plot(err, label=f'dt={dt}')
        ax.set_yscale('log')
        ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    timestep_verification()
