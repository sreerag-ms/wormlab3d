import os
from argparse import Namespace
from typing import Dict, Any, Union, List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.particles.sdbn_explorer import SDBNExplorer
from wormlab3d.particles.sdbn_modelling import calculate_pe_parameters_for_trajectory
from wormlab3d.particles.util import plot_3d_trajectories, plot_states, plot_2d_trajectory, plot_3d_trajectory, \
    plot_trajectory_with_frame, plot_displacements_and_states, plot_state_parameters, plot_msd
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args

show_plots = True
save_plots = True
img_extension = 'png'

# dists = {
#     'speeds': 'lognorm',
#     'planar_angles': 'cauchy',
#     'nonplanar_angles': 'cauchy',
# }


dists = {
    'speeds': 'lognorm',
    'planar_angles': 'norm',
    'nonplanar_angles': 'norm',
}


def _get_delta_ts_str(args: Namespace) -> str:
    deltas = sorted(list(args.deltas), reverse=True)
    delta_ts = np.array(deltas) / 25
    return ', '.join([f'{delta:.1f}s' for delta in delta_ts])


def _make_data_plots(
        args: Namespace,
        state_parameters: Dict[str, Dict[str, Dict[str, Union[str, float]]]],
        additional: Dict[str, Any],
        real_or_sim: str,
        states_plot: bool = False,
        trajectory_plot: bool = False,
        displacements_plot: bool = False,
        parameters_plot: bool = False,
):
    """
    Make some data plots.
    """
    if not show_plots and not save_plots:
        return

    if real_or_sim == 'real':
        title = f'Trial {args.trial}. Deltas: {{{_get_delta_ts_str(args)}}}.'
        fn_suffix = f'_trial={args.trial}'
    else:
        title = f'Simulation. Deltas: {{{_get_delta_ts_str(args)}}}.'
        fn_suffix = '_sim'

    if states_plot:
        ts = np.arange(len(additional['frame']['X']))
        plot_states(ts, additional['states'], additional['speeds'], additional['planar_angles'],
                    additional['nonplanar_angles'], title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_states{fn_suffix}.{img_extension}')

    if trajectory_plot:
        plot_trajectory_with_frame(**additional['frame'], T=200, arrow_scale=0.05, title=title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_frame{fn_suffix}.{img_extension}')

    if displacements_plot:
        plot_displacements_and_states(additional['displacements'], additional['states'], title=title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_displacements_states{fn_suffix}.{img_extension}')

    if parameters_plot:
        plot_state_parameters(state_parameters, additional['state_values'], title=title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_state_parameters{fn_suffix}.{img_extension}')

    if show_plots:
        plt.show()


def _make_simulation_plots(
        Xs_sim: np.ndarray,
        Xs_real: List[np.ndarray],
        ts: np.ndarray,
        states: np.ndarray,
        speeds: np.ndarray,
        planar_angles: np.ndarray,
        nonplanar_angles: np.ndarray,
        plot_n_examples: int = 1,
        states_plot: bool = False,
        trajectory_2d_plot: bool = False,
        trajectory_3d_plot: bool = False,
        trajectory_3d_comparison_plot: bool = False,
):
    """
    Make some plots of the simulation output.
    """
    if not show_plots and not save_plots:
        return

    for i in range(plot_n_examples):
        title = f'Simulation run {i}.'
        if states_plot:
            plot_states(ts, states[i], speeds[i], planar_angles[i], nonplanar_angles[i], title)
            if save_plots:
                plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_states_sim_{i}.{img_extension}')

        if trajectory_2d_plot:
            plot_2d_trajectory(ts, Xs_sim[i], title)
            if save_plots:
                plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_2d_sim_{i}.{img_extension}')

        if trajectory_3d_plot:
            plot_3d_trajectory(Xs_sim[i], title)
            if save_plots:
                plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_3d_sim_{i}.{img_extension}')

    if trajectory_3d_comparison_plot:
        plot_3d_trajectories(Xs_sim=Xs_sim[0][None, ...], Xs_real=Xs_real, title='Real vs Simulations')
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectories_real_vs_sim.{img_extension}')

    if show_plots:
        plt.show()


def _make_msd_plot(
        args: Namespace,
        X_real: np.ndarray,
        Xs_sim: np.ndarray,
):
    """
    Make an MSD plot of the simulated results against the real trajectory.
    """
    if not show_plots and not save_plots:
        return
    plot_msd(args, X_real, Xs_sim)
    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_msd.{img_extension}')
    if show_plots:
        plt.show()


def create_particle_model_for_trial():
    """
    Calculate the particle model parameters from a trial, generate some simulations and validate.
    """
    args = get_args()
    assert args.trajectory_point is None  # Use the full postures
    assert args.planarity_window is not None
    deltas = sorted(list(args.deltas), reverse=True)
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)

    transition_rates, state_parameters, additional = calculate_pe_parameters_for_trajectory(
        X=X,
        deltas=deltas,
        pcas=pcas,
        pca_window=args.planarity_window,
        displacement_aggregation=args.aggregation,
        distributions=dists,
        return_additional=True
    )

    # Make plots
    _make_data_plots(
        args,
        state_parameters,
        additional,
        real_or_sim='real',
        states_plot=False,
        trajectory_plot=False,
        displacements_plot=False,
        parameters_plot=False,
    )

    # Create the model
    pe = SDBNExplorer(
        batch_size=20,
        transition_rates=transition_rates,
        state_parameters=state_parameters
    )

    # Run some simulations based on these parameters
    T = len(X)
    dt = 1
    ts, Xs_sim, states, speeds, planar_angles, nonplanar_angles = pe.forward(T, dt)

    # Make plots
    _make_simulation_plots(
        Xs_sim=Xs_sim,
        Xs_real=[X.mean(axis=1), ],
        ts=ts,
        states=states,
        speeds=speeds,
        planar_angles=planar_angles,
        nonplanar_angles=nonplanar_angles,
        plot_n_examples=3,
        states_plot=False,
        trajectory_2d_plot=False,
        trajectory_3d_plot=False,
        trajectory_3d_comparison_plot=True,
    )

    if 0:
        # Recalculate parameters from simulated trajectory
        transition_rates_sim, state_parameters_sim, additional_sim = calculate_pe_parameters_for_trajectory(
            X=Xs_sim[0][:, None, :],
            deltas=deltas,
            pca_window=args.planarity_window,
            displacement_aggregation=args.aggregation,
            distributions=dists,
            return_additional=True
        )

        # Make plots
        _make_data_plots(
            args,
            state_parameters_sim,
            additional_sim,
            real_or_sim='sim',
            states_plot=True,
            trajectory_plot=True,
            displacements_plot=True,
            parameters_plot=True,
        )

    # MSD plot
    _make_msd_plot(
        args,
        X_real=X,
        Xs_sim=Xs_sim
    )


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    create_particle_model_for_trial()
