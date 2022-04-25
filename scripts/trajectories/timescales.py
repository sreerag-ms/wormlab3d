import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Reconstruction, Trial, Dataset
from wormlab3d.trajectories.angles import calculate_trajectory_angles_parallel, calculate_planar_angles_parallel
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.displacement import calculate_displacements, \
    calculate_transitions_and_dwells_multiple_deltas, DISPLACEMENT_AGGREGATION_L2
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import get_deltas_from_args

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'svg'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['dataset', 'trial', 'frames', 'src', 'directionality', 'aggregation', 'deltas', 'delta_range',
              'delta_step', 'u']:
        if k in excludes:
            continue
        if k == 'dataset' and args.dataset is not None:
            fn += f'_dataset={args.dataset}'
        elif k == 'trial' and args.trial is not None:
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            fn += f'_{args.midline3d_source}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'aggregation':
            fn += f'_{args.aggregation}'
        elif k == 'deltas':
            fn += f'_d={",".join([str(d) for d in args.deltas])}'
        elif k == 'delta_range':
            fn += f'_dr={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            if args.delta_step < 0:
                fn += f'_ds={args.delta_step:.2f}'
            else:
                fn += f'_ds={int(args.delta_step)}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'projection':
            fn += f'_p={args.projection}'

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


def displacement_over_time():
    """
    Plot traces of the displacement values along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args()
    trajectory = get_trajectory_from_args(args)
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
            make_filename('traces', args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def transition_rates_trajectory():
    """
    Show the transition rates between above/below average mobility for a single trajectory.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Calculate displacements and states
    X = get_trajectory_from_args(args)
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
    if args.reconstruction is not None:
        reconstruction = Reconstruction.objects.get(id=args.reconstruction)
        trial = reconstruction.trial
    else:
        trial = Trial.objects.get(id=args.trial)
    fig.suptitle(f'Trial {trial.id}.')

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
            make_filename('transition_rates', args)
        )
    if show_plots:
        plt.show()


def transition_rates_dataset():
    """
    Show the transition rates between above/below average mobility across a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate displacements and states for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    exp0_all = {delta: [] for delta in deltas}
    exp1_all = {delta: [] for delta in deltas}
    dd0_all = {delta: [] for delta in deltas}
    dd1_all = {delta: [] for delta in deltas}
    lambda0 = {}
    lambda1 = {}
    dd0 = {}
    dd1 = {}
    exp0 = {}
    exp1 = {}
    n_trials = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')
        args.trial = trial.id

        # Group results by concentration
        c = trial.experiment.concentration
        if c not in lambda0:
            lambda0[c] = {delta: [] for delta in deltas}
            lambda1[c] = {delta: [] for delta in deltas}
            exp0[c] = {delta: [] for delta in deltas}
            exp1[c] = {delta: [] for delta in deltas}
            dd0[c] = {delta: [] for delta in deltas}
            dd1[c] = {delta: [] for delta in deltas}
            n_trials[c] = 0
        n_trials[c] += 1

        # Get trajectory and calculate displacements and dwells
        X = get_trajectory_from_args(args)
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
            l0cd = M[0, 1]
            l1cd = M[1, 0]
            lambda0[c][delta].append(l0cd)
            lambda1[c][delta].append(l1cd)
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

            # Dwell durations
            dd0cd = dwd['off_durations'] / 25
            dd1cd = dwd['on_durations'] / 25
            dd0[c][delta].extend(dd0cd)
            dd1[c][delta].extend(dd1cd)
            dd0_all[delta].extend(dd0cd)
            dd1_all[delta].extend(dd1cd)

            # Fit exponential distributions
            e0cd = dd0cd.mean() / dd0cd.var()
            e1cd = dd1cd.mean() / dd1cd.var()
            exp0[c][delta].append(e0cd)
            exp1[c][delta].append(e1cd)
            exp0_all[delta].append(e0cd)
            exp1_all[delta].append(e1cd)

    # Sort by concentration
    lambda0 = {k: v for k, v in sorted(list(lambda0.items()))}
    lambda1 = {k: v for k, v in sorted(list(lambda1.items()))}
    dd0 = {k: v for k, v in sorted(list(dd0.items()))}
    dd1 = {k: v for k, v in sorted(list(dd1.items()))}
    exp0 = {k: v for k, v in sorted(list(exp0.items()))}
    exp1 = {k: v for k, v in sorted(list(exp1.items()))}
    concs = list(lambda0.keys())

    # Set up plots
    n_rows = 2 + len(n_trials)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 3))
    fig.suptitle(f'Dataset {ds.id}.')

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

    # Aggregate results at each concentration
    n_trials_total = 0
    l0_means = []
    l1_means = []
    s_means = []
    dd0_means = []
    dd1_means = []
    exp0_means = []
    exp1_means = []
    for i, c in enumerate(concs):
        n_trials_total += n_trials[c]

        # Get means and standard deviations for parameters
        l0_mean = np.zeros(len(deltas))
        l0_std = np.zeros(len(deltas))
        l1_mean = np.zeros(len(deltas))
        l1_std = np.zeros(len(deltas))
        s_mean = np.zeros(len(deltas))
        s_std = np.zeros(len(deltas))
        exp0_mean = np.zeros(len(deltas))
        dd0_mean = np.zeros(len(deltas))
        dd0_std = np.zeros(len(deltas))
        dd1_mean = np.zeros(len(deltas))
        dd1_std = np.zeros(len(deltas))
        exp0_std = np.zeros(len(deltas))
        exp1_mean = np.zeros(len(deltas))
        exp1_std = np.zeros(len(deltas))
        for j, delta in enumerate(deltas):
            n = len(lambda0[c][delta])
            if n == 0:
                continue
            l0s = np.array(lambda0[c][delta])
            l1s = np.array(lambda1[c][delta])
            l0_mean[j] = np.mean(l0s)
            l0_std[j] = np.std(l0s)
            l1_mean[j] = np.mean(l1s)
            l1_std[j] = np.std(l1s)
            s_mean_j = l0s / (l0s + l1s) * 2 - 1
            s_mean[j] = np.mean(s_mean_j)
            s_std[j] = np.std(s_mean_j)

            dd0j = np.array(dd0[c][delta])
            dd1j = np.array(dd1[c][delta])
            dd0_mean[j] = np.mean(dd0j)
            dd0_std[j] = np.std(dd0j)
            dd1_mean[j] = np.mean(dd1j)
            dd1_std[j] = np.std(dd1j)

            exp0j = np.array(exp0[c][delta])
            exp1j = np.array(exp1[c][delta])
            exp0_mean[j] = np.mean(exp0j)
            exp0_std[j] = np.std(exp0j)
            exp1_mean[j] = np.mean(exp1j)
            exp1_std[j] = np.std(exp1j)
        l0_means.append(l0_mean)
        l1_means.append(l1_mean)
        s_means.append(s_mean)
        dd0_means.append(dd0_mean)
        dd1_means.append(dd1_mean)
        exp0_means.append(exp0_mean)
        exp1_means.append(exp1_mean)

        # Plots
        axes[i + 2, 2].set_title(f'Concentration = {c:.2f}% ({n_trials[c]} trials)')
        _plot_lambdas(axes[i + 2, 0], l0_mean, l0_std, l1_mean, l1_std)
        _plot_lambda_ratios(axes[i + 2, 1], l0_mean, l1_mean)
        _plot_state_means(axes[i + 2, 2], s_mean, s_std)
        _plot_dwell_durations(axes[i + 2, 3], dd0_mean, dd0_std, dd1_mean, dd1_std)
        _plot_rates(axes[i + 2, 4], exp0_mean, exp0_std, exp1_mean, exp1_std)

    # Get means and standard deviations across all trajectories
    l0_mean = np.zeros(len(deltas))
    l0_std = np.zeros(len(deltas))
    l1_mean = np.zeros(len(deltas))
    l1_std = np.zeros(len(deltas))
    s_mean = np.zeros(len(deltas))
    s_std = np.zeros(len(deltas))
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

        exp0j = np.array(exp0_all[delta])
        exp1j = np.array(exp1_all[delta])
        exp0_mean[j] = np.mean(exp0j)
        exp0_std[j] = np.std(exp0j)
        exp1_mean[j] = np.mean(exp1j)
        exp1_std[j] = np.std(exp1j)

    # Plot averages across all trajectories
    axes[0, 2].set_title(f'All trajectories ({n_trials_total} trials)')
    _plot_lambdas(axes[0, 0], l0_mean, l0_std, l1_mean, l1_std)
    _plot_lambda_ratios(axes[0, 1], l0_mean, l1_mean)
    _plot_state_means(axes[0, 2], s_mean, s_std)
    _plot_dwell_durations(axes[0, 3], dd0_mean, dd0_std, dd1_mean, dd1_std)
    _plot_rates(axes[0, 4], exp0_mean, exp0_std, exp1_mean, exp1_std)

    # Get means and stds of the concentration-means
    l0_mean = np.mean(l0_means, axis=0)
    l0_std = np.std(l0_means, axis=0)
    l1_mean = np.mean(l1_means, axis=0)
    l1_std = np.std(l1_means, axis=0)
    s_mean = np.mean(s_means, axis=0)
    s_std = np.std(s_means, axis=0)
    dd0_mean = np.mean(dd0_means, axis=0)
    dd0_std = np.std(dd0_means, axis=0)
    dd1_mean = np.mean(dd1_means, axis=0)
    dd1_std = np.std(dd1_means, axis=0)
    exp0_mean = np.mean(exp0_means, axis=0)
    exp0_std = np.std(exp0_means, axis=0)
    exp1_mean = np.mean(exp1_means, axis=0)
    exp1_std = np.std(exp1_means, axis=0)

    # Plot concentration averages
    axes[1, 2].set_title(f'Equally weighted concentrations ({n_trials_total} trials)')
    _plot_lambdas(axes[1, 0], l0_mean, l0_std, l1_mean, l1_std)
    _plot_lambda_ratios(axes[1, 1], l0_mean, l1_mean)
    _plot_state_means(axes[1, 2], s_mean, s_std)
    _plot_dwell_durations(axes[1, 3], dd0_mean, dd0_std, dd1_mean, dd1_std)
    _plot_rates(axes[1, 4], exp0_mean, exp0_std, exp1_mean, exp1_std)

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_dataset', args)
        )
    if show_plots:
        plt.show()


def transition_rates_dataset_simple():
    """
    Show the transition rates between above/below average mobility across a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate displacements and states for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    lambda0 = {}
    lambda1 = {}
    n_trials = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')
        args.trial = trial.id

        # Group results by concentration
        c = trial.experiment.concentration
        if c == 0.75:
            continue
        if c not in lambda0:
            lambda0[c] = {delta: [] for delta in deltas}
            lambda1[c] = {delta: [] for delta in deltas}
            n_trials[c] = 0
        n_trials[c] += 1

        # Get trajectory and calculate displacements and dwells
        X = get_trajectory_from_args(args)
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
            l0cd = M[0, 1]
            l1cd = M[1, 0]
            lambda0[c][delta].append(l0cd)
            lambda1[c][delta].append(l1cd)
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Sort by concentration
    lambda0 = {k: v for k, v in sorted(list(lambda0.items()))}
    lambda1 = {k: v for k, v in sorted(list(lambda1.items()))}
    concs = list(lambda0.keys())

    # Set up plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Dataset {ds.id}.')

    ax1 = axes[0, 0]
    ax1.set_title('All trajectories averaged')
    ax1.set_ylabel(f'Transition probability')
    ax1.set_xlabel('$\Delta s$')
    ax1l = axes[0, 1]
    ax1l.set_title('All trajectories averaged (log-scale)')
    ax1l.set_ylabel(f'Transition probability')
    ax1l.set_xlabel('$\Delta s$')
    ax1l.set_yscale('log')

    ax2 = axes[1, 0]
    ax2.set_title('Equally weighted concentrations average')
    ax2.set_ylabel(f'Transition probability')
    ax2.set_xlabel('$\Delta s$')
    ax2l = axes[1, 1]
    ax2l.set_title('Equally weighted concentrations average (log-scale)')
    ax2l.set_ylabel(f'Transition probability')
    ax2l.set_xlabel('$\Delta s$')
    ax2l.set_yscale('log')

    # Aggregate results at each concentration
    n_trials_total = 0
    l0_means = {delta: [] for delta in deltas}
    l1_means = {delta: [] for delta in deltas}
    for i, c in enumerate(concs):
        n_trials_total += n_trials[c]

        # Get means and standard deviations for parameters
        l0_mean = {}
        l1_mean = {}
        for j, delta in enumerate(deltas):
            n = len(lambda0[c][delta])
            if n == 0:
                continue
            l0s = np.array(lambda0[c][delta])
            l1s = np.array(lambda1[c][delta])
            l0_mean[delta] = np.mean(l0s)
            l1_mean[delta] = np.mean(l1s)
            l0_means[delta].append(l0_mean[delta])
            l1_means[delta].append(l1_mean[delta])

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
    _plot_data(ax1, dts, l0_mean, l1_mean, l0_std, l1_std)
    _plot_data(ax1l, dts, l0_mean, l1_mean, l0_std, l1_std)

    # Get means and stds of the concentration-means
    l0_mean = {}
    l1_mean = {}
    l0_std = {}
    l1_std = {}
    dts = []
    for j, delta in enumerate(deltas):
        n = len(l0_means[delta])
        if n == 0:
            continue
        l0s = np.array(l0_means[delta])
        l1s = np.array(l1_means[delta])
        l0_mean[delta] = np.mean(l0s)
        l1_mean[delta] = np.mean(l1s)
        l0_std[delta] = np.std(l0s)
        l1_std[delta] = np.std(l1s)
        dts.append(delta_ts[j])

    # Plot concentration averages
    _plot_data(ax2, dts, l0_mean, l1_mean, l0_std, l1_std)
    _plot_data(ax2l, dts, l0_mean, l1_mean, l0_std, l1_std)

    ax1.legend()
    ax2.legend()
    ax1l.legend()
    ax2l.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_dataset_simple', args)
        )
    if show_plots:
        plt.show()


def displacement_transition_rates_dataset_averages():
    """
    Show the average transition rates between above/below average mobility across a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate displacements and states for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')

        # Get trajectory and calculate displacements and dwells
        args.trial = trial.id
        X = get_trajectory_from_args(args)
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
            l0cd = M[0, 1]
            l1cd = M[1, 0]
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Set up plots
    fig, ax = plt.subplots(1, figsize=(4, 3))
    # ax.set_title(f'Dataset {ds.id}.')
    ax.set_ylabel(f'Transition probability')
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
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('transition_rates_dataset_averages', args)
        )
    if show_plots:
        plt.show()


def angle_diffs_over_time(which: str = 'trajectory'):
    """
    Plot traces of the angle changes along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args()
    if which == 'trajectory':
        args.trajectory_point = -1
    trajectory = get_trajectory_from_args(args)

    if which == 'trajectory':
        angles = calculate_trajectory_angles_parallel(trajectory, args.deltas)
    else:
        angles = calculate_planar_angles_parallel(trajectory, args.deltas)
    dwells = calculate_transitions_and_dwells_multiple_deltas(angles)

    deltas = list(angles.keys())
    delta_ts = np.array(deltas) / 25
    fig, axes = plt.subplots(len(deltas), 2, figsize=(12, 4 + 2 * len(deltas)))
    for i, delta in enumerate(deltas):
        a = angles[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(a, alpha=0.75)
        ax.set_title(f'$\Delta={delta_ts[i]:.2f}s$')
        if which == 'trajectory':
            ax.set_ylabel('$\\theta=\measuredangle\left(x_t-x_{t+\Delta}, x_{t+\Delta}-x_{t+2\Delta})\\right)$')
        else:
            ax.set_ylabel('$\\theta=\measuredangle\left(v_3[t:t+\Delta], v_3[t+\Delta:t+2\Delta])\\right)$')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = a.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(a), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(a), color='orange', alpha=0.3, zorder=-1,
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
            make_filename(f'traces_angles_{which}', args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def transition_rates_angle_diffs_trajectory(which: str = 'trajectory'):
    """
    Show the transition rates between above/below average angles for a single trajectory.
    """
    args = get_args()
    if which == 'trajectory':
        args.trajectory_point = -1
    deltas, delta_ts = get_deltas_from_args(args)
    trajectory = get_trajectory_from_args(args)

    # Calculate angles and states
    if which == 'trajectory':
        angles = calculate_trajectory_angles_parallel(trajectory, deltas)
    else:
        angles = calculate_planar_angles_parallel(trajectory, deltas)
    dwells = calculate_transitions_and_dwells_multiple_deltas(angles)

    # Get telegraph signals and calculate transition probabilities
    lambdas = np.zeros((len(deltas), 2))
    for i, delta in enumerate(deltas):
        d = angles[delta]
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
    duration_means = {'on': {}, 'off': {}}
    for i, delta in enumerate(deltas):
        dwd = dwells[delta]
        for j, onoff in enumerate(['off', 'on']):
            durations = dwd[f'{onoff}_durations']
            exp_params[i, j] = durations.mean() / durations.var()
            duration_means[onoff][delta] = durations.mean()

    # Setup plots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    if args.reconstruction is not None:
        reconstruction = Reconstruction.objects.get(id=args.reconstruction)
        trial = reconstruction.trial
    else:
        trial = Trial.objects.get(id=args.trial)
    fig.suptitle(f'Trial {trial.id}.')

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

    # Plot dwell duration means
    ax = axes[1, 0]
    ax.plot(delta_ts, duration_means['off'].values(), label='$s_0$')
    ax.plot(delta_ts, duration_means['on'].values(), label='$s_1$')
    # ax.axhline(y=1, linestyle='--', color='grey')
    # ax.plot(delta_ts, l0 / l1, label='$\lambda_0/\lambda_1$')
    # ax.plot(delta_ts, l1 / l0, label='$\lambda_1/\lambda_0$')
    ax.legend()
    # ax.set_title(f'Ratios')
    ax.set_title(f'Dwell duration means')
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
            make_filename(f'transition_rates_{which}_angles', args)
        )
    if show_plots:
        plt.show()


def nonplanarity_over_time():
    """
    Plot traces of the nonplanarity along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args()

    # Calculate planarities across different time windows
    nonp = {}
    for delta in args.deltas:
        logger.info(f'Fetching PCA data for delta = {int(delta)}.')
        args.planarity_window = int(delta)
        pca_cache = get_pca_cache_from_args(args)
        r = pca_cache.explained_variance_ratio.T
        nonp_delta = r[2] / np.sqrt(r[1] * r[0])
        nonp[delta] = nonp_delta
    dwells = calculate_transitions_and_dwells_multiple_deltas(nonp)

    deltas = list(nonp.keys())
    delta_ts = np.array(deltas) / 25

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
            make_filename(f'traces_nonplanarity', args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def nonplanarity_and_displacement_over_time(x_label: str = 'time'):
    """
    Plot traces of the nonplanarity/displacement changes along a trajectory highlighting regions above and below the average.
    Show histograms of the dwell times spent in each state.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    deltas = list(args.deltas)
    delta_ts = np.array(deltas)
    if x_label == 'time':
        delta_ts = delta_ts / 25
    # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trial={args.trial}', trajectory)

    # Calculate displacements
    displacements = calculate_displacements(X, deltas, args.aggregation)
    dwells_displacements = calculate_transitions_and_dwells_multiple_deltas(displacements)

    # Calculate planarities across different time windows
    nonp = {}
    for delta in deltas:
        logger.info(f'Fetching PCA data for delta = {int(delta)}.')
        args.planarity_window = int(delta)
        pca_cache = get_pca_cache_from_args(args)
        r = pca_cache.explained_variance_ratio.T
        nonp[delta] = r[2] / np.sqrt(r[1] * r[0])
    dwells_nonp = calculate_transitions_and_dwells_multiple_deltas(nonp)

    # # Calculate displacements across all time windows for transitions
    # deltas_all, delta_ts_all = get_deltas_from_args(args)
    # displacements_all = calculate_displacements(X, deltas_all, args.aggregation)
    # dwells_displacements_all = calculate_transitions_and_dwells_multiple_deltas(displacements_all)
    #
    # # Calculate planarities across all time windows
    # nonp_all = {}
    # for delta in deltas_all:
    #     logger.info(f'Fetching PCA data for delta = {int(delta)}.')
    #     args.planarity_window = int(delta)
    #     pca_cache = get_pca_cache_from_args(args)
    #     r = pca_cache.explained_variance_ratio.T
    #     nonp_all[delta] = r[2] / np.sqrt(r[1] * r[0])
    # dwells_nonp_all = calculate_transitions_and_dwells_multiple_deltas(nonp_all)

    # # Get telegraph signals and calculate transition probabilities
    # lambdas_displacements = np.zeros((len(deltas_all), 2))
    # lambdas_nonp = np.zeros((len(deltas_all), 2))
    # for i, delta in enumerate(deltas_all):
    #     for data, dwells, lambdas in [
    #         [displacements_all, dwells_displacements_all, lambdas_displacements],
    #         [nonp_all, dwells_nonp_all, lambdas_nonp]
    #     ]:
    #         d = data[delta]
    #         dwd = dwells[delta]
    #         t = np.zeros_like(d)
    #         for r in dwd['on']:
    #             t[r[0]:r[1]] = 1
    #         M = _calculate_transition_matrix(t)
    #         lambdas[i] = [M[0, 1], M[1, 0]]

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
    # fig, axes = plt.subplots(2, 1 + len(deltas), figsize=(4 + 5 * len(deltas), 8))
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

    # # Plot lambdas
    # for i, lambdas in enumerate([lambdas_displacements, lambdas_nonp]):
    #     l0 = lambdas[:, 0]
    #     l1 = lambdas[:, 1]
    #     ax = axes[i, len(deltas)]
    #     ax.plot(delta_ts_all, l0, label='$p_0 (s_0 \\rightarrow s_1)$')
    #     ax.plot(delta_ts_all, l1, label='$p_1 (s_1 \\rightarrow s_0)$')
    #     ax.set_yscale('log')
    #     ax.legend()
    #     ax.set_title(f'Transition probabilities')
    #     ax.set_ylabel('Probability')
    #     ax.set_xlabel('$\Delta s$')

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(f'traces_nonp_and_disp', args, excludes=['delta_range', 'delta_step'])
        )
    if show_plots:
        plt.show()


def nonplanarity_transition_rates_dataset_averages():
    """
    Show the average transition rates between above/below average nonplanarity across a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate nonplanarity for all trials
    lambda0_all = {delta: [] for delta in deltas}
    lambda1_all = {delta: [] for delta in deltas}
    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')

        # Calculate planarities across different time windows
        args.trial = trial.id
        nonp = {}
        for delta in deltas:
            logger.info(f'Fetching PCA data for delta = {int(delta)}.')
            args.planarity_window = int(delta)
            pca_cache = get_pca_cache_from_args(args)
            r = pca_cache.explained_variance_ratio.T
            nonp[delta] = r[2] / np.sqrt(r[1] * r[0])
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
            l0cd = M[0, 1]
            l1cd = M[1, 0]
            lambda0_all[delta].append(l0cd)
            lambda1_all[delta].append(l1cd)

    # Set up plots
    fig, ax = plt.subplots(1, figsize=(6, 4))
    # ax.set_title(f'Dataset {ds.id}.')
    ax.set_ylabel(f'Transition probability')
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
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('nonplanarity_transition_rates_dataset_averages', args)
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    args_ = get_args()
    # displacement_over_time()
    # angle_diffs_over_time()
    # nonplanarity_over_time()
    nonplanarity_and_displacement_over_time(x_label='time')
    # if args_.dataset is not None:
    #     transition_rates_dataset()
    # else:
    #     transition_rates_trajectory()
    # transition_rates_angle_diffs_trajectory()

    # if args_.dataset is not None:
    # transition_rates_dataset()
    # transition_rates_dataset_simple()
    # displacement_transition_rates_dataset_averages()
    # nonplanarity_transition_rates_dataset_averages()
