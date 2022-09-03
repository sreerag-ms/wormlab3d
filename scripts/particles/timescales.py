import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.cache import get_sim_state_from_args
from wormlab3d.particles.simulation_state import SimulationState
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.displacement import calculate_displacements, \
    calculate_transitions_and_dwells_multiple_deltas, DISPLACEMENT_AGGREGATION_L2
from wormlab3d.trajectories.pca import calculate_pcas, PCACache
from wormlab3d.trajectories.util import get_deltas_from_args, calculate_speeds, smooth_trajectory

show_plots = False
save_plots = True
img_extension = 'png'


# tex-mode needed for svg images otherwise brackets are broken
# from wormlab3d.toolkit.plot_utils import tex_mode
# tex_mode()


def make_filename(method: str, SS: SimulationState, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['sim', 'sim_idx', 'aggregation', 'deltas', 'delta_range', 'delta_step']:
        if k in excludes:
            continue
        if k == 'sim':
            fn += f'_sim={SS.parameters.id}'
        elif k == 'sim_idx' and hasattr(args, 'sim_idx') and args.sim_idx is not None:
            fn += f'_idx={args.sim_idx}'
        elif k == 'aggregation':
            fn += f'_agg-{args.aggregation}'
        elif k == 'deltas':
            fn += f'_d={",".join([str(d) for d in args.deltas])}'
        elif k == 'delta_range':
            fn += f'_dr={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            if args.delta_step < 0:
                fn += f'_ds={args.delta_step:.2f}'
            else:
                fn += f'_ds={int(args.delta_step)}'

    return LOGS_PATH / (fn + '.' + img_extension)


def _calculate_transition_matrix(states: np.ndarray) -> np.ndarray:
    """
    Given a 1D array of binary states, return a 2x2 matrix of state transition probabilities.
    """
    states = states.astype(np.int32)
    M = np.zeros((2, 2))
    for (from_state, to_state) in zip(states[:-1], states[1:]):
        M[from_state, to_state] += 1
    M /= M.sum(axis=1, keepdims=True)
    return M


def displacement_over_time(
        sim_idx: int = 0
):
    """
    Plot traces of the displacement values along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    trajectory = SS.X[sim_idx]
    args.sim_idx = sim_idx
    displacements = calculate_displacements(trajectory, args.deltas, args.aggregation)
    dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)

    deltas = list(displacements.keys())
    delta_ts = np.array(deltas) / 25
    fig, axes = plt.subplots(len(deltas), 2, figsize=(12, 4 + 2 * len(deltas)))
    for i, delta in enumerate(deltas):
        d = displacements[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(d, alpha=0.75)
        ax.set_title(f'$\Delta={delta_ts[i]:.2f}s$')
        if args.aggregation == DISPLACEMENT_AGGREGATION_L2:
            ax.set_ylabel('$d=|x(t)-x(t+\Delta)|$')
        else:
            ax.set_ylabel('$d=(x(t)-x(t+\Delta))^2$')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = d.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(d), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(d), color='orange', alpha=0.3, zorder=-1,
                            linewidth=0)

        # Plot histogram of dwell times
        ax = axes[i, 1]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.hist(dwells[delta]['on_durations'], bins=50, density=True, alpha=0.5, color='green')
        ax.hist(dwells[delta]['off_durations'], bins=50, density=True, alpha=0.5, color='orange')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('traces', SS, args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def transition_rates_trajectory(
        sim_idx: int = 0
):
    """
    Show the transition rates between above/below average mobility for a single trajectory.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    X = SS.X[sim_idx]
    args.sim_idx = sim_idx
    deltas, delta_ts = get_deltas_from_args(args)

    # Calculate displacements and states
    displacements = calculate_displacements(X, deltas, args.aggregation)
    dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)

    # Get telegraph signals and calculate transition probabilities
    lambdas = np.zeros((len(deltas), 2))
    for i, delta in enumerate(deltas):
        d = displacements[delta]
        dwd = dwells[delta]
        t = np.zeros_like(d)
        for r in dwd['on']:
            t[r[0]:r[1]] = 1
        M = _calculate_transition_matrix(t)
        lambdas[i] = [M[0, 1], M[1, 0]]
    l0 = lambdas[:, 0]
    l1 = lambdas[:, 1]

    # Fit exponential distributions
    exp_params = np.zeros((len(deltas), 2))
    for i, delta in enumerate(deltas):
        dwd = dwells[delta]
        for j, onoff in enumerate(['off', 'on']):
            durations = dwd[f'{onoff}_durations']
            exp_params[i, j] = durations.mean() / durations.var()

    # Setup plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'Simulation {SS.parameters.id}, Run #{sim_idx}.')

    # Plot lambdas
    ax = axes[0, 0]
    ax.plot(delta_ts, l0, label='$\lambda_0 (s_0 \\rightarrow s_1)$')
    ax.plot(delta_ts, l1, label='$\lambda_1 (s_1 \\rightarrow s_0)$')
    ax.legend()
    ax.set_title(f'Transition rates')
    ax.set_ylabel('P')
    ax.set_xlabel('$\Delta s$')

    # Plot lambdas on a log-plot
    ax = axes[0, 1]
    ax.plot(delta_ts, l0, label='$\lambda_0$')
    ax.plot(delta_ts, l1, label='$\lambda_1$')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(f'Transition rates (log-y)')
    ax.set_ylabel('P')
    ax.set_xlabel('$\Delta s$')

    # Plot lambda ratios
    ax = axes[1, 0]
    ax.axhline(y=1, linestyle='--', color='grey')
    ax.plot(delta_ts, l0 / l1, label='$\lambda_0/\lambda_1$')
    ax.plot(delta_ts, l1 / l0, label='$\lambda_1/\lambda_0$')
    ax.legend()
    ax.set_title(f'Ratios')
    ax.set_xlabel('$\Delta s$')

    # Plot means
    ax = axes[1, 1]
    ax.set_title('State mean')
    ax.axhline(y=0, linestyle='--', color='grey')
    mean = l0 / (l0 + l1) * 2 - 1
    ax.plot(delta_ts, mean, label='$<S>$')
    ax.legend()
    ax.set_xlabel('$\Delta s$')

    # Plot variances
    ax = axes[2, 0]
    ax.set_title('State variance')
    variances = l0 * l1 / (l0 + l1)**2
    ax.plot(delta_ts, variances, label='$var\{S\}$')
    ax.legend()
    ax.set_xlabel('$\Delta s$')

    # Plot exponential distribution parameter
    ax = axes[2, 1]
    ax.set_title('Dwell duration exponential rate')
    ax.plot(delta_ts, exp_params[:, 0], label='$r(s_0)$')
    ax.plot(delta_ts, exp_params[:, 1], label='$r(s_1)$')
    ax.legend()
    ax.set_xlabel('$\Delta s$')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_traj', SS, args)
        )
    if show_plots:
        plt.show()


def transition_rates_simulation_batch():
    """
    Show the transition rates between above/below average mobility across a batch of simulation runs.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    deltas, delta_ts = get_deltas_from_args(args)
    fps = 1 / SS.parameters.dt

    # Calculate displacements and states for all runs
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    exp0_all = {delta: [] for delta in deltas}
    exp1_all = {delta: [] for delta in deltas}
    dd0_all = {delta: [] for delta in deltas}
    dd1_all = {delta: [] for delta in deltas}
    for i, X in enumerate(SS.X):
        logger.info(f'Calculating data for sim {i + 1}/{len(SS)}.')

        # Get trajectory and calculate displacements and dwells
        displacements = calculate_displacements(X, deltas, args.aggregation)
        dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)

        # Loop over deltas
        for delta in deltas:
            if delta not in displacements:
                continue
            d = displacements[delta]
            dwd = dwells[delta]

            # Get telegraph signals and calculate transition probabilities
            t = np.zeros_like(d)
            for r in dwd['on']:
                t[r[0]:r[1]] = 1
            M = _calculate_transition_matrix(t)
            l0cd = M[0, 1] * fps
            l1cd = M[1, 0] * fps
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

            # Dwell durations
            dd0cd = dwd['off_durations'] / fps
            dd1cd = dwd['on_durations'] / fps
            dd0_all[delta].extend(dd0cd)
            dd1_all[delta].extend(dd1cd)

            # Fit exponential distributions
            e0cd = dd0cd.mean() / dd0cd.var()
            e1cd = dd1cd.mean() / dd1cd.var()
            exp0_all[delta].append(e0cd)
            exp1_all[delta].append(e1cd)

    # Get means and standard deviations across all trajectories
    l0_mean = np.zeros(len(deltas))
    l0_std = np.zeros(len(deltas))
    l1_mean = np.zeros(len(deltas))
    l1_std = np.zeros(len(deltas))
    s_mean = np.zeros(len(deltas))
    s_std = np.zeros(len(deltas))
    dd0_mean = np.zeros(len(deltas))
    dd0_std = np.zeros(len(deltas))
    dd1_mean = np.zeros(len(deltas))
    dd1_std = np.zeros(len(deltas))
    exp0_mean = np.zeros(len(deltas))
    exp0_std = np.zeros(len(deltas))
    exp1_mean = np.zeros(len(deltas))
    exp1_std = np.zeros(len(deltas))
    for j, delta in enumerate(deltas):
        n = len(lambda0_all[delta])
        if n == 0:
            continue
        l0s = np.array(lambda0_all[delta])
        l1s = np.array(lambda1_all[delta])
        l0_mean[j] = np.mean(l0s)
        l0_std[j] = np.std(l0s)
        l1_mean[j] = np.mean(l1s)
        l1_std[j] = np.std(l1s)
        s_mean_j = l0s / (l0s + l1s) * 2 - 1
        s_mean[j] = np.mean(s_mean_j)
        s_std[j] = np.std(s_mean_j)
        dd0_mean[j] = np.mean(dd0_all[delta])
        dd0_std[j] = np.std(dd0_all[delta])
        dd1_mean[j] = np.mean(dd1_all[delta])
        dd1_std[j] = np.std(dd1_all[delta])
        exp0j = np.array(exp0_all[delta])
        exp1j = np.array(exp1_all[delta])
        exp0_mean[j] = np.mean(exp0j)
        exp0_std[j] = np.std(exp0j)
        exp1_mean[j] = np.mean(exp1j)
        exp1_std[j] = np.std(exp1j)

    # Set up plots
    fig, axes = plt.subplots(5, figsize=(12, 16))

    def _plot_lambdas(ax_, l0_mean_, l0_std_, l1_mean_, l1_std_):
        ax_.fill_between(delta_ts, l0_mean_ - l0_std_, l0_mean_ + l0_std_, color='blue', alpha=0.2)
        ax_.fill_between(delta_ts, l1_mean_ - l1_std_, l1_mean_ + l1_std_, color='orange', alpha=0.2)
        ax_.plot(delta_ts, l0_mean_, color='blue', label='$\lambda_0 (s_0 \\rightarrow s_1)$')
        ax_.plot(delta_ts, l1_mean_, color='orange', label='$\lambda_1 (s_1 \\rightarrow s_0)$')
        ax_.set_yscale('log')
        ax_.legend()
        ax_.set_ylabel(f'Transition probability')
        ax_.set_xlabel('$\Delta s$')

    def _plot_lambda_ratios(ax_, l0_, l1_):
        ax_.axhline(y=1, linestyle='--', color='grey')
        ax_.plot(delta_ts, l0_ / l1_, label='$\\bar\lambda_0/\\bar\lambda_1$')
        ax_.plot(delta_ts, l1_ / l0_, label='$\\bar\lambda_1/\\bar\lambda_0$')
        ax_.legend()
        ax_.set_ylabel(f'Ratios')
        ax_.set_xlabel('$\Delta s$')

    def _plot_state_means(ax_, s_mean_, s_std_):
        ax_.axhline(y=0, linestyle='--', color='grey')
        ax_.plot(delta_ts, s_mean_, label='$<S>$', color='blue')
        ax_.fill_between(delta_ts, s_mean_ - s_std_, s_mean_ + s_std_, color='blue', alpha=0.2)
        ax_.legend()
        ax_.set_ylabel('State mean')
        ax_.set_xlabel('$\Delta s$')

    def _plot_dwell_durations(ax_, dd0_mean_, dd0_std_, dd1_mean_, dd1_std_):
        ax_.fill_between(delta_ts, dd0_mean_ - dd0_std_, dd0_mean_ + dd0_std_, color='blue', alpha=0.2)
        ax_.fill_between(delta_ts, dd1_mean_ - dd1_std_, dd1_mean_ + dd1_std_, color='orange', alpha=0.2)
        ax_.plot(delta_ts, dd0_mean_, color='blue', label='$<d(s_0)>$')
        ax_.plot(delta_ts, dd1_mean_, color='orange', label='$<d(s_1)>$')
        ax_.set_yscale('log')
        ax_.legend()
        ax_.set_ylabel('Duration')
        ax_.set_xlabel('$\Delta s$')

    def _plot_rates(ax_, exp0_mean_, exp0_std_, exp1_mean_, exp1_std_):
        ax_.fill_between(delta_ts, exp0_mean_ - exp0_std_, exp0_mean_ + exp0_std_, color='blue', alpha=0.2)
        ax_.fill_between(delta_ts, exp1_mean_ - exp1_std_, exp1_mean_ + exp1_std_, color='orange', alpha=0.2)
        ax_.plot(delta_ts, exp0_mean_, color='blue', label='$r(s_0)$')
        ax_.plot(delta_ts, exp1_mean_, color='orange', label='$r(s_1)$')
        ax_.set_yscale('log')
        ax_.legend()
        ax_.set_ylabel('Exp. Rate')
        ax_.set_xlabel('$\Delta s$')

    # Plot averages across all trajectories
    axes[0].set_title(f'Simulation {SS.parameters.id}, {len(SS)} trajectories.')
    _plot_lambdas(axes[0], l0_mean, l0_std, l1_mean, l1_std)
    _plot_lambda_ratios(axes[1], l0_mean, l1_mean)
    _plot_state_means(axes[2], s_mean, s_std)
    _plot_dwell_durations(axes[3], dd0_mean, dd0_std, dd1_mean, dd1_std)
    _plot_rates(axes[4], exp0_mean, exp0_std, exp1_mean, exp1_std)

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_batch', SS, args)
        )
    if show_plots:
        plt.show()


def transition_rates_sim_batch_simple():
    """
    Show the transition rates between above/below average mobility across a batch of simulation runs.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    fps = 1 / SS.parameters.dt
    deltas, delta_ts = get_deltas_from_args(args)

    # Calculate displacements and states for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    for i, X in enumerate(SS.X):
        logger.info(f'Calculating data for sim {i + 1}/{len(SS)}.')

        # Get trajectory and calculate displacements and dwells
        displacements = calculate_displacements(X, deltas, args.aggregation)
        dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)

        # Loop over deltas
        for delta in deltas:
            if delta not in displacements:
                continue
            d = displacements[delta]
            dwd = dwells[delta]

            # Skip if not enough data
            if len(dwd['on_durations']) < 3 or len(dwd['off_durations']) < 3:
                continue

            # Get telegraph signals and calculate transition probabilities
            t = np.zeros_like(d)
            for r in dwd['on']:
                t[r[0]:r[1]] = 1
            M = _calculate_transition_matrix(t)
            l0cd = M[0, 1] * fps
            l1cd = M[1, 0] * fps
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Get means and standard deviations across all trajectories
    l0_mean = {}
    l1_mean = {}
    l0_std = {}
    l1_std = {}
    dts = []
    for j, delta in enumerate(deltas):
        n = len(lambda0_all[delta])
        if n == 0:
            continue
        l0s = np.array(lambda0_all[delta])
        l1s = np.array(lambda1_all[delta])
        l0_mean[delta] = np.mean(l0s)
        l1_mean[delta] = np.mean(l1s)
        l0_std[delta] = np.std(l0s)
        l1_std[delta] = np.std(l1s)
        dts.append(delta_ts[j])

    # Set up plots
    fig, axes = plt.subplots(2, figsize=(20, 12))
    axes[0].set_title(f'Simulation {SS.parameters.id}, {len(SS)} trajectories.')

    ax1 = axes[0]
    ax1.set_title('All trajectories averaged')
    ax1.set_ylabel(f'Transition probability')
    ax1.set_xlabel('$\Delta s$')
    ax1l = axes[1]
    ax1l.set_title('All trajectories averaged (log-scale)')
    ax1l.set_ylabel(f'Transition probability')
    ax1l.set_xlabel('$\Delta s$')
    ax1l.set_yscale('log')

    def _plot_data(ax_, dts_, l0_mean_, l1_mean_, l0_std_, l1_std_):
        l0_mean_ = np.array(list(l0_mean_.values()))
        l0_std_ = np.array(list(l0_std_.values()))
        l0_lb = l0_mean_ - l0_std_
        l0_ub = l0_mean_ + l0_std_
        l1_mean_ = np.array(list(l1_mean_.values()))
        l1_std_ = np.array(list(l1_std_.values()))
        l1_lb = l1_mean_ - l1_std_
        l1_ub = l1_mean_ + l1_std_
        ax_.fill_between(dts_, l0_lb, l0_ub, color='blue', alpha=0.2)
        ax_.fill_between(dts_, l1_lb, l1_ub, color='orange', alpha=0.2)
        ax_.plot(dts_, l0_mean_, color='blue', label='$p_0 (s_0 \\rightarrow s_1)$')
        ax_.plot(dts_, l1_mean_, color='orange', label='$p_1 (s_1 \\rightarrow s_0)$')

    # Plot averages across all trajectories
    _plot_data(ax1, dts, l0_mean, l1_mean, l0_std, l1_std)
    _plot_data(ax1l, dts, l0_mean, l1_mean, l0_std, l1_std)
    ax1.legend()
    ax1l.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_batch_simple', SS, args)
        )
    if show_plots:
        plt.show()


def displacement_transition_rates_sim_batch_averages(
        ax: Axes = None
):
    """
    Show the average transition rates between above/below average mobility across a dataset.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    fps = 1 / SS.parameters.dt
    deltas, delta_ts = get_deltas_from_args(args)
    return_ax = ax is not None

    # Calculate displacements and states for all runs
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    for i, X in enumerate(SS.X):
        logger.info(f'Calculating data for sim {i + 1}/{len(SS)}.')

        # Calculate displacements and dwells
        displacements = calculate_displacements(X, deltas, args.aggregation)
        dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)

        # Loop over deltas
        for delta in deltas:
            if delta not in displacements:
                continue
            d = displacements[delta]
            dwd = dwells[delta]

            # Skip if not enough data
            if len(dwd['on_durations']) < 3 or len(dwd['off_durations']) < 3:
                continue

            # Get telegraph signals and calculate transition probabilities
            t = np.zeros_like(d)
            for r in dwd['on']:
                t[r[0]:r[1]] = 1
            M = _calculate_transition_matrix(t)
            l0cd = M[0, 1] * fps
            l1cd = M[1, 0] * fps
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Set up plots
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.set_title(f'Simulation {SS.parameters.id}.\n{len(SS)} trajectories.')
    ax.set_ylabel(f'Transition rate')
    ax.set_xlabel('$\Delta\ (s)$')
    ax.set_yscale('log')

    # Get means and standard deviations across all trajectories
    l0_mean = {}
    l1_mean = {}
    l0_std = {}
    l1_std = {}
    dts = []
    for j, delta in enumerate(deltas):
        n = len(lambda0_all[delta])
        if n == 0:
            continue
        l0s = np.array(lambda0_all[delta])
        l1s = np.array(lambda1_all[delta])
        l0_mean[delta] = np.mean(l0s)
        l1_mean[delta] = np.mean(l1s)
        l0_std[delta] = np.std(l0s)
        l1_std[delta] = np.std(l1s)
        dts.append(delta_ts[j])

    def _plot_data(ax_, dts_, l0_mean_, l1_mean_, l0_std_, l1_std_):
        l0_mean_ = np.array(list(l0_mean_.values()))
        l0_std_ = np.array(list(l0_std_.values()))
        l0_lb = l0_mean_ - l0_std_
        l0_ub = l0_mean_ + l0_std_
        l1_mean_ = np.array(list(l1_mean_.values()))
        l1_std_ = np.array(list(l1_std_.values()))
        l1_lb = l1_mean_ - l1_std_
        l1_ub = l1_mean_ + l1_std_
        ax_.fill_between(dts_, l0_lb, l0_ub, color='blue', alpha=0.2)
        ax_.fill_between(dts_, l1_lb, l1_ub, color='orange', alpha=0.2)
        ax_.plot(dts_, l0_mean_, color='blue', label='$p_0(s_0\\rightarrow\ s_1)$')
        ax_.plot(dts_, l1_mean_, color='orange', label='$p_1(s_1\\rightarrow\ s_0)$')

    # Plot averages across all trajectories
    _plot_data(ax, dts, l0_mean, l1_mean, l0_std, l1_std)
    ax.legend()
    if return_ax:
        return ax

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'displacement_transition_rates_batch_averages',
                SS, args,
            )
        )
    if show_plots:
        plt.show()


def nonplanarity_over_time(
        sim_idx: int = 0
):
    """
    Plot traces of the nonplanarity along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    X = SS.X[sim_idx]
    args.sim_idx = sim_idx

    # Calculate planarities across different time windows
    nonp = {}
    for delta in args.deltas:
        if delta < 3:
            delta = 3
        logger.info(f'Fetching PCA data for delta = {int(delta)}.')
        pcas = calculate_pcas(X, window_size=delta, parallel=True)
        pcas_cache = PCACache(pcas)
        nonp[delta] = pcas_cache.nonp
    dwells = calculate_transitions_and_dwells_multiple_deltas(nonp)

    deltas = list(nonp.keys())
    delta_ts = np.array(deltas) * SS.parameters.dt

    fig, axes = plt.subplots(len(deltas), 2, figsize=(12, 4 + 2 * len(deltas)))
    for i, delta in enumerate(deltas):
        p = nonp[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(p, alpha=0.75)
        ax.set_title(f'$\Delta={delta_ts[i]:.2f}s$')
        ax.set_ylabel('Non-planarity')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = p.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(p), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(p), color='orange', alpha=0.3, zorder=-1,
                            linewidth=0)

        # Plot histogram of dwell times
        ax = axes[i, 1]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.hist(dwells[delta]['on_durations'], bins=50, density=True, alpha=0.5, color='green')
        ax.hist(dwells[delta]['off_durations'], bins=50, density=True, alpha=0.5, color='orange')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(f'traces_nonplanarity', SS, args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def speed_over_time(
        sim_idx: int = 0
):
    """
    Plot traces of the speed along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    X = SS.X[sim_idx]
    args.sim_idx = sim_idx
    speeds = {}
    for delta in args.deltas:
        if delta < 2:
            delta = 3
        if delta % 2 == 0:
            delta += 1
        logger.info(f'Calculating speeds with smoothing window = {int(delta)}.')
        Xs = smooth_trajectory(X, window_len=delta)
        speeds[delta] = calculate_speeds(Xs, signed=False) / SS.parameters.dt

    dwells = calculate_transitions_and_dwells_multiple_deltas(speeds)
    deltas = list(speeds.keys())
    delta_ts = np.array(deltas) / 25

    fig, axes = plt.subplots(len(deltas), 2, figsize=(12, 4 + 2 * len(deltas)))
    for i, delta in enumerate(deltas):
        s = speeds[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(s, alpha=0.75)
        ax.set_title(f'$\Delta={delta_ts[i]:.2f}s$')
        ax.set_ylabel('Speed')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = s.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(s), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(s), color='orange', alpha=0.3, zorder=-1,
                            linewidth=0)

        # Plot histogram of dwell times
        ax = axes[i, 1]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.hist(dwells[delta]['on_durations'], bins=50, density=True, alpha=0.5, color='green')
        ax.hist(dwells[delta]['off_durations'], bins=50, density=True, alpha=0.5, color='orange')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(f'traces_speed', SS, args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def nonplanarity_and_displacement_over_time(
        sim_idx: int = 0,
        x_label: str = 'time'
):
    """
    Plot traces of the nonplanarity/displacement changes along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    X = SS.X[sim_idx]
    args.sim_idx = sim_idx
    deltas = list(args.deltas)

    # Calculate planarities across different time windows
    nonp = {}
    for delta in deltas:
        if delta < 3:
            delta = 3
        logger.info(f'Fetching PCA data for delta = {int(delta)}.')
        pcas = calculate_pcas(X, window_size=delta, parallel=True)
        pcas_cache = PCACache(pcas)
        nonp[delta] = pcas_cache.nonp
    dwells_nonp = calculate_transitions_and_dwells_multiple_deltas(nonp)

    deltas = list(nonp.keys())
    delta_ts = np.array(deltas)
    if x_label == 'time':
        delta_ts = delta_ts * SS.parameters.dt

    # Calculate displacements
    displacements = calculate_displacements(X, deltas, args.aggregation)
    dwells_displacements = calculate_transitions_and_dwells_multiple_deltas(displacements)

    def _add_dwell_regions(ax_, dwells_, ub_):
        for on_dwell in dwells_[delta]['on']:
            fill_range = np.arange(on_dwell[0] - 1, on_dwell[1])
            if x_label == 'time':
                fill_range = fill_range / 25
            ax_.fill_between(fill_range, ub_, color='green', alpha=0.2, zorder=-1, linewidth=0)
        for off_dwell in dwells_[delta]['off']:
            fill_range = np.arange(off_dwell[0] - 1, off_dwell[1])
            if x_label == 'time':
                fill_range = fill_range / 25
            ax_.fill_between(fill_range, ub_, color='orange', alpha=0.2, zorder=-1, linewidth=0)

    max_i = min([len(d) for _, d in displacements.items()])
    ts = np.arange(max_i)
    if x_label == 'time':
        ts = ts / 25

    fig, axes = plt.subplots(2, len(deltas), figsize=(4 + 4 * len(deltas), 8))
    for i, delta in enumerate(deltas):
        d = displacements[delta][:max_i]
        p = nonp[delta][:max_i]

        # Displacement trace over time
        ax = axes[0, i]
        ax.plot(ts, d, alpha=0.75, zorder=100)
        if x_label == 'time':
            ax.set_title(f'$\Delta={delta_ts[i]}s$')
        else:
            ax.set_title(f'$\Delta={delta_ts[i]}\ frames$')
        if args.aggregation == DISPLACEMENT_AGGREGATION_L2:
            ax.set_ylabel('$d=|x(t)-x(t+\Delta)|$')
        else:
            ax.set_ylabel('$d=(x(t)-x(t+\Delta))^2$')
        avg = d.mean()
        ax.axhline(y=avg, color='red')
        ax.set_xlim(left=0, right=ts[-1])
        ax.set_ylim(bottom=0, top=max(d))
        _add_dwell_regions(ax, dwells_displacements, max(d))
        if x_label == 'time':
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('Frame')
        ax.locator_params(axis='y', nbins=6)

        # Nonplanarity trace over time
        ax = axes[1, i]
        ax.plot(ts, p, alpha=0.75, zorder=100)
        ax.set_ylabel('Non-planarity')
        avg = p.mean()
        ax.axhline(y=avg, color='red')
        ax.set_xlim(left=0, right=ts[-1])
        # ax.set_ylim(bottom=0, top=max(p))
        ax.set_ylim(bottom=0, top=0.65)
        _add_dwell_regions(ax, dwells_nonp, 0.65)
        if x_label == 'time':
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('Frame')
        ax.set_yticks([0, 0.2, 0.4, 0.6])

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(f'traces_nonp_and_disp', SS, args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def nonplanarity_transition_rates_sim_batch_averages(
        ax: Axes = None
):
    """
    Show the average transition rates between above/below average nonplanarity across a dataset.
    """
    args = get_args(validate_source=False)
    SS = get_sim_state_from_args(args)
    fps = 1 / SS.parameters.dt
    deltas, delta_ts = get_deltas_from_args(args, min_delta=3)
    return_ax = ax is not None

    # Calculate nonplanarity for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    for i, X in enumerate(SS.X):
        logger.info(f'Calculating data for sim {i + 1}/{len(SS)}.')

        # Calculate planarities across different time windows
        nonp = {}
        for delta in deltas:
            logger.info(f'Calculating non-planarity for delta = {int(delta)}.')
            nonp[delta] = SS.get_nonp_windowed(int(delta))[i]
        if SS.needs_save:
            SS.save()
        dwells = calculate_transitions_and_dwells_multiple_deltas(nonp)

        # Loop over deltas
        for delta in deltas:
            if delta not in nonp:
                continue
            p = nonp[delta]
            dwd = dwells[delta]

            # Skip if not enough data
            if len(dwd['on_durations']) < 3 or len(dwd['off_durations']) < 3:
                continue

            # Get telegraph signals and calculate transition probabilities
            t = np.zeros_like(p)
            for r in dwd['on']:
                t[r[0]:r[1]] = 1
            M = _calculate_transition_matrix(t)
            l0cd = M[0, 1] * fps
            l1cd = M[1, 0] * fps
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Set up plots
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        ax.set_title(f'Simulation {SS.parameters.id}, {len(SS)} trajectories.')
    ax.set_ylabel(f'Transition rate')
    ax.set_xlabel('$\Delta\ (s)$')
    ax.set_yscale('log')

    # Get means and standard deviations across all trajectories
    l0_mean = {}
    l1_mean = {}
    l0_std = {}
    l1_std = {}
    dts = []
    for j, delta in enumerate(deltas):
        n = len(lambda0_all[delta])
        if n == 0:
            continue
        l0s = np.array(lambda0_all[delta])
        l1s = np.array(lambda1_all[delta])
        l0_mean[delta] = np.mean(l0s)
        l1_mean[delta] = np.mean(l1s)
        l0_std[delta] = np.std(l0s)
        l1_std[delta] = np.std(l1s)
        dts.append(delta_ts[j])

    def _plot_data(ax_, dts_, l0_mean_, l1_mean_, l0_std_, l1_std_):
        l0_mean_ = np.array(list(l0_mean_.values()))
        l0_std_ = np.array(list(l0_std_.values()))
        l0_lb = l0_mean_ - l0_std_
        l0_ub = l0_mean_ + l0_std_
        l1_mean_ = np.array(list(l1_mean_.values()))
        l1_std_ = np.array(list(l1_std_.values()))
        l1_lb = l1_mean_ - l1_std_
        l1_ub = l1_mean_ + l1_std_
        ax_.fill_between(dts_, l0_lb, l0_ub, color='blue', alpha=0.2)
        ax_.fill_between(dts_, l1_lb, l1_ub, color='orange', alpha=0.2)
        ax_.plot(dts_, l0_mean_, color='blue', label='$p_0 (s_0 \\rightarrow s_1)$')
        ax_.plot(dts_, l1_mean_, color='orange', label='$p_1 (s_1 \\rightarrow s_0)$')

    # Plot averages across all trajectories
    _plot_data(ax, dts, l0_mean, l1_mean, l0_std, l1_std)
    ax.legend()

    if return_ax:
        return ax

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'nonplanarity_transition_rates_batch_averages',
                SS, args
            )
        )
    if show_plots:
        plt.show()


def transition_rates_sim_batch_averages():
    """
    Combine the displacements and non-planarity dataset averages plots.
    """
    plateaux_line_denom = 40

    def add_plateaux_indicator(ax_):
        ax_.axhline(y=1 / plateaux_line_denom, color='red', linestyle='--', alpha=0.8, linewidth=2)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax_.text(
            -0.01,
            1 / plateaux_line_denom,
            f'$\\frac{{1}}{{{plateaux_line_denom}s}}$',
            color='red',
            fontsize=14,
            verticalalignment='center',
            horizontalalignment='right',
            transform=trans
        )

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

    ax = axes[0]
    displacement_transition_rates_sim_batch_averages(ax)
    ax.set_title('Displacements')
    add_plateaux_indicator(ax)
    ax.set_ylim(bottom=5e-3)
    ax.set_yticks([1e-2, 1e-1, 1])

    ax = axes[1]
    nonplanarity_transition_rates_sim_batch_averages(ax)
    ax.set_title('Non-planarity')
    add_plateaux_indicator(ax)
    ax.yaxis.set_tick_params(labelbottom=True)

    fig.tight_layout()

    if save_plots:
        args = get_args(validate_source=False)
        SS = get_sim_state_from_args(args)
        plt.savefig(
            make_filename(
                'transition_rates_batch_averages',
                SS, args
            )
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    args_ = get_args(validate_source=False)

    idx = 0

    # displacement_over_time(sim_idx=idx)
    # transition_rates_trajectory(sim_idx=idx)
    # transition_rates_simulation_batch()
    # transition_rates_sim_batch_simple()
    # displacement_transition_rates_sim_batch_averages()
    #
    # nonplanarity_over_time(sim_idx=idx)
    # speed_over_time(sim_idx=idx)
    #
    # nonplanarity_and_displacement_over_time(sim_idx=idx, x_label='time')
    # nonplanarity_transition_rates_sim_batch_averages()
    transition_rates_sim_batch_averages()
