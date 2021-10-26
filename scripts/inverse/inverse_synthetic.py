from typing import Tuple

import numpy as np
from fenics import *

from simple_worm.controls import ControlsFenics, ControlSequenceFenics
from simple_worm.frame import FrameFenics, FrameSequenceFenics
from simple_worm.material_parameters import MaterialParametersFenics
from simple_worm.util import f2n

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass
from typing import Tuple

import numpy as np

from simple_worm.controls import ControlsNumpy, ControlSequenceNumpy
from simple_worm.controls_torch import ControlSequenceBatchTorch
from simple_worm.frame import FrameNumpy, FrameSequenceNumpy, FrameFenics
from simple_worm.frame_torch import FrameBatchTorch
from simple_worm.material_parameters import MaterialParameters
from simple_worm.material_parameters_torch import MaterialParametersBatchTorch
from simple_worm.plot3d import generate_interactive_scatter_clip, plot_CS, plot_frame_components, plot_frame_3d, \
    plot_CS_vs_output
from simple_worm.worm import Worm
from wormlab3d import logger
from wormlab3d.data.model import SwRun, SwSimulationParameters
from wormlab3d.data.model.sw_run import SwMaterialParameters, SwFrameSequence, SwControlSequence
from wormlab3d.simple_worm.args import SimulationArgs, FrameSequenceArgs
from wormlab3d.simple_worm.args.parse import parse_arguments
from wormlab3d.simple_worm.manager import Manager
from simple_worm.worm import grad


def generate_target(
        simulation_args: SimulationArgs,
        show_animations: bool = False
) -> Tuple[FrameNumpy, ControlSequenceNumpy, FrameSequenceNumpy]:
    """
    Generate a target frame sequence using fixed controls.
    Resolve the initial shape first then run again.
    """
    import matplotlib.pyplot as plt


    N = 40  #simulation_args.worm_length
    MP = MaterialParameters(**simulation_args.get_mp_dict())
    gamma_pref = np.linspace(start=-1, stop=1, num=N-1) * 5
    # gamma_pref = np.ones(N - 1) * 5
    C = ControlsNumpy(
        alpha=np.ones(N) * 5,
        # alpha=np.zeros(N),
        beta=np.linspace(start=-1, stop=1, num=N) * 5,
        # beta=np.zeros(N),
        # gamma=np.ones(N - 1) * 5,
        gamma=gamma_pref,
    )

    fig, axes = plt.subplots(4)

    # Run once to get initial configuration
    T_setup = 0.5
    dt_setup = 0.0001
    n_timesteps_setup = int(T_setup / dt_setup)
    CS = ControlSequenceNumpy([C] * n_timesteps_setup)
    worm = Worm(N, dt_setup)

    worm.initialise()
    axes[0].set_title('Initial vs preferred')
    axes[0].plot(f2n(project(worm.F.gamma, worm.Q)))
    axes[0].plot(gamma_pref)

    FS = worm.solve(T_setup, MP=MP.to_fenics(), CS=CS.to_fenics(worm))
    if show_animations:
        generate_interactive_scatter_clip(FS.to_numpy(), fps=25)

    axes[1].set_title('After solve')
    axes[1].plot(f2n(project(FS[-1].gamma_expr, worm.Q)))

    x0 = project(worm.F.x, worm.V3)
    mu0 = sqrt(dot(grad(x0), grad(x0)))
    gamma = TrialFunction(worm.Q)
    v = TestFunction(worm.Q)
    F_gamma0 = (gamma - dot(grad(worm.F.e1), worm.F.e2)) / mu0 * v * dx
    a_gamma0, L_gamma0 = lhs(F_gamma0), rhs(F_gamma0)
    gamma0 = Function(worm.Q)
    solve(a_gamma0 == L_gamma0, gamma0)

    axes[2].set_title('After solve - recalculation')
    axes[2].plot(f2n(project(gamma0, worm.Q)))

    axes[3].set_title('Residual over time')
    res = [F.gamma_res for F in FS]
    axes[3].plot(res)

    fig.tight_layout()
    plt.show()

    # exit()


    # plot_CS_vs_output(CS, FS.to_numpy())
    # plt.show()

    # Run again, starting from the shape to be held
    T = simulation_args.duration
    dt = simulation_args.dt
    n_timesteps = int(T / dt)
    Fn = FS[-1].to_numpy()
    F0 = FrameNumpy(x=Fn.x, psi=Fn.psi, calculate_components=True)
    worm = Worm(N, dt, plot_it=True)
    # Fn = FS[-1].clone()
    # F0 = FrameFenics(
    #     x=project(Fn.x, worm.V3),
    #     psi=project(Fn.psi, worm.V),
    #     e0=project(Fn.e0, worm.V3),
    #     e1=project(Fn.e1, worm.V3),
    #     e2=project(Fn.e2, worm.V3),
    #     kappa_expr=Fn.kappa_expr,
    #     gamma_expr=Fn.gamma_expr,
    #     worm=worm
    # )
    # F0 = Fn
    CS = ControlSequenceNumpy([C] * n_timesteps)
    FS = worm.solve(T, MP=MP.to_fenics(), F0=F0.to_fenics(worm), CS=CS.to_fenics(worm))
    # FS = worm.solve(T, MP=MP.to_fenics(), F0=F0, CS=CS.to_fenics(worm))
    if show_animations:
        generate_interactive_scatter_clip(FS.to_numpy(), fps=25)

    # plot_CS(CS)
    # plt.show()
    plot_frame_components(F0)
    plt.show()
    # plot_frame_3d(F0)
    # plt.show()
    plot_CS_vs_output(CS, FS.to_numpy())
    plt.show()

    return F0, CS, FS.to_numpy()


def generate_or_load_sim_params(
        simulation_args: SimulationArgs
) -> SwSimulationParameters:
    sim_config = simulation_args.get_config_dict()
    sim_params = SwSimulationParameters.objects(**sim_config)
    if sim_params.count() > 0:
        sim_params = sim_params[0]
        logger.info(f'Loaded simulation parameters id={sim_params.id}.')
    else:
        logger.info('No suitable simulation parameter records found in database, creating new.')
        sim_params = SwSimulationParameters(**sim_config)
        sim_params.save()
    return sim_params


def generate_or_load_target(
        frame_sequence_args: FrameSequenceArgs,
        simulation_args: SimulationArgs,
) -> Tuple['MaterialParametersBatchTorch', 'FrameBatchTorch', 'ControlSequenceBatchTorch', 'FrameSequenceBatchTorch']:
    # if frame_sequence_args.sw_run_id is not None:
    #     return

    logger.info('Generating test target.')
    F0, CS, FS = generate_target(simulation_args, show_animations=False)
    exit()

    run = SwRun()
    run.sim_params = generate_or_load_sim_params(simulation_args)
    run.MP = SwMaterialParameters(**simulation_args.get_mp_dict())
    run.F0 = SwFrameSequence()
    run.F0.x = F0.x
    run.F0.psi = F0.psi
    run.CS = SwControlSequence()
    run.CS.alpha = CS.controls['alpha']
    run.CS.beta = CS.controls['beta']
    run.CS.gamma = CS.controls['gamma']
    run.FS = SwFrameSequence()
    run.FS.x = FS.x
    run.FS.psi = FS.psi
    run.save()

    logger.info(f'Saved simulation run id={run.id}.')

    # Set the target run id argument to the newly saved entry.
    frame_sequence_args.sw_run_id = run.id


def optimise():
    """
    Run the inverse optimisation.
    """
    runtime_args, frame_sequence_args, simulation_args, optimiser_args, regularisation_args = parse_arguments()
    generate_or_load_target(frame_sequence_args, simulation_args)

    # Construct manager
    manager = Manager(
        runtime_args=runtime_args,
        frame_sequence_args=frame_sequence_args,
        simulation_args=simulation_args,
        optimiser_args=optimiser_args,
        regularisation_args=regularisation_args
    )

    # Do some training
    manager.train(
        n_steps=runtime_args.n_steps
    )


if __name__ == '__main__':
    optimise()
