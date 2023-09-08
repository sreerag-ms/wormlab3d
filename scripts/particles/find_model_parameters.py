import os
from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import NonlinearConstraint, differential_evolution
from scipy.stats import kstest

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, logger
from wormlab3d.data.model import Dataset, PEParameters
from wormlab3d.particles.cache import get_sim_state_from_args
from wormlab3d.particles.simulation_state import SimulationState
from wormlab3d.particles.tumble_run import find_approximation, find_approximation2, generate_or_load_ds_statistics
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import calculate_speeds, smooth_trajectory

show_plots = True
save_plots = False
# show_plots = False
# save_plots = True
interactive_plots = False
img_extension = 'svg'

DATA_CACHE_PATH = LOGS_PATH / 'cache'
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def _identifiers(args: Namespace) -> str:
    id_str = (f'_ds={args.dataset}'
              f'_e={args.approx_error_limit}'
              f'_pw={args.planarity_window_vertices}'
              f'_df={args.approx_distance}'
              f'_hf={args.approx_curvature_height}'
              f'_sm={args.smoothing_window_K}')
    return id_str


def _calculate_derived_parameters(
        args: Namespace,
        ds: Dataset,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Calculate the derived parameters.
    """
    logger.info('Calculating derived parameters.')
    # error_limits = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    error_limit = 0.05
    avg_op = np.mean
    # avg_op = np.median

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Log the speeds
    low_speeds = np.zeros(len(ds.include_trials))
    high_speeds = np.zeros(len(ds.include_trials))

    # Transition matrix
    T = np.zeros((3, 3), dtype=int)

    # Values
    min_run_speed_duration: Tuple[float, float] = (0.01, 60.)
    durations = []
    speeds = []
    planar_angles = []
    nonplanar_angles = []

    # Calculate the approximation for all trials
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Computing tumble-run model for trial={trial.id}.')
        args.trial = trial.id
        dt = 1 / trial.fps

        X = get_trajectory_from_args(args)
        pcas = get_pca_cache_from_args(args)
        e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

        # Take centre of mass
        if X.ndim == 3:
            X = X.mean(axis=1)
        X -= X.mean(axis=0)

        # Defaults
        # distance_first: int = 500,
        # distance_min: int = 3,
        # height_first: int = 50,
        # smooth_e0_first: int = 101,
        # smooth_K_first: int = 101,

        # Paper values:
        # distance = 500
        # distance_min = 3
        # height = 100
        # smooth_e0 = 201
        # smooth_K = 201

        # Find the approximation
        approx, distance, height, smooth_e0, smooth_K = find_approximation(
            X=X,
            e0=e0,
            error_limit=error_limit,
            planarity_window_vertices=args.planarity_window_vertices,
            distance_first=100,
            distance_min=3,
            height_first=80,
            smooth_e0_first=251,
            smooth_K_first=251,
            max_attempts=50,
            quiet=False
        )
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, _, _, _ = approx

        # Put in time units
        run_durations *= dt
        run_speeds /= dt

        # Discard long runs where the distance travelled is too small  (0.01, 60.)
        include_idxs = np.unique(
            np.concatenate([
                np.argwhere(run_speeds > min_run_speed_duration[0]),
                np.argwhere(run_durations < min_run_speed_duration[1])
            ])
        )
        run_durations = run_durations[include_idxs]
        run_speeds = run_speeds[include_idxs]

        durations.extend(run_durations.tolist())
        speeds.extend(run_speeds.tolist())
        planar_angles.extend(planar_angles_j.tolist())
        nonplanar_angles.extend(nonplanar_angles_j.tolist())

        # Split up the trajectory into the run sections to calculate scaled speeds per run in approximation
        logger.info('Calculating trajectory speeds.')
        speed = calculate_speeds(X)  # / dt
        # speed = smooth_trajectory(speed, 25)  #  smooth the speed signal again
        speed_ap = np.zeros_like(speed)
        v0 = 0
        for j in range(len(tumble_idxs) + 1):
            v1 = tumble_idxs[j] if j < len(tumble_idxs) else -1
            run_speed = speed[v0:v1]

            # Distance is reduced using the straight-line runs, so scale speeds accordingly
            assert np.allclose(vertices[j], X[v0])
            assert np.allclose(vertices[j + 1], X[v1])
            run_dist_og = np.linalg.norm(X[v0:v1 - 1] - X[v0 + 1:v1], axis=-1).sum()
            run_dist_ap = np.linalg.norm(vertices[j] - vertices[j + 1], axis=-1).sum()
            sf = run_dist_ap / run_dist_og  # always < 1
            speed_ap[v0:v1] = run_speed * sf

            v0 = v1

        # Calculate the high and low speeds
        avg_speed = avg_op(speed_ap)
        low_speeds[i] = avg_op(speed_ap[speed_ap <= avg_speed])
        high_speeds[i] = avg_op(speed_ap[speed_ap > avg_speed])

        # Split up the trajectory into the run sections to calculate scaled speeds per run
        logger.info('Calculating transitions.')
        v0 = 0
        for j in range(len(tumble_idxs) + 1):
            v1 = tumble_idxs[j] if j < len(tumble_idxs) else -1
            run_speed = speed_ap[v0:v1]

            # Get telegraph signal of above and below the mean trajectory speed
            states = (run_speed > avg_speed).astype(int)

            # Previously, started with a turn (unless first)
            if v0 > 0:
                T[2, states[0]] += 1

            # Log speed-state transitions
            for (from_state, to_state) in zip(states[:-1], states[1:]):
                T[from_state, to_state] += 1

            # End with a turn (unless last)
            if j < len(tumble_idxs):
                # T[states[-1], 2] += 1
                # Restrict turns from the slow state
                if states[-1] == 1:
                    T[1, 0] += 1
                T[0, 2] += 1

            v0 = v1

    # Calculate the final transition matrix to get the rates
    M = T / T.sum(axis=1, keepdims=True)

    if show_plots:
        plot_histograms(durations, speeds, planar_angles, nonplanar_angles)

    return low_speeds, high_speeds, M


def _generate_or_load_directly_derived_params(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / ('derived' + _identifiers(args))
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    T = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            low_speeds = data['low_speeds']
            high_speeds = data['high_speeds']
            T = data['T']
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            T = None
            logger.warning(f'Could not load cache: {e}')

    if T is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        low_speeds, high_speeds, T = _calculate_derived_parameters(args, ds)
        data = {
            'low_speeds': low_speeds,
            'high_speeds': high_speeds,
            'T': T,
        }
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, low_speeds, high_speeds, T


def plot_histograms(durations, speeds, planar_angles, nonplanar_angles):
    # Plot histograms
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    colour_real = default_colours[0]

    for i, (param_name, values) in enumerate({
                                                 'Run durations': durations,
                                                 'Run speeds': speeds,
                                                 'Planar angles': planar_angles,
                                                 'Non-planar angles': nonplanar_angles
                                             }.items()):
        ax = axes[i]
        ax.set_title(param_name)

        if param_name not in ['Planar angles', 'Non-planar angles']:
            ax.set_yscale('log')
            bin_range = None
        if param_name == 'Planar angles':
            bin_range = (-np.pi, np.pi)
        elif param_name == 'Non-planar angles':
            bin_range = (-np.pi / 2, np.pi / 2)
        if param_name == 'Speeds':
            weights = np.array(durations)
        else:
            weights = np.ones_like(values)

        ax.hist(
            values,
            weights=weights,
            color=colour_real,
            bins=11,
            density=True,
            alpha=0.75,
            range=bin_range
        )
        ax.set_title(param_name)
        ax.set_ylabel('Density')
        if param_name == 'Run durations':
            ax.set_xticks([0, 50, 100])
            ax.set_xticklabels(['0', '50', '100'])
            ax.set_xlabel('s')
            ax.set_yticks([1e-5, 1e-2])
        if param_name == 'Run speeds':
            ax.set_xticks([0, 0.1, 0.2])
            ax.set_xticklabels(['0', '0.1', '0.2'])
            ax.set_xlabel('mm/s')
            ax.set_yticks([1e-1, 1e1])
        if param_name == 'Planar angles':
            ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
            ax.set_xticks([-np.pi, np.pi])
            ax.set_xticklabels(['$-\pi$', '$\pi$'])
            ax.set_xlabel('$\\theta$')
            ax.set_yticks([0, 0.2])
        if param_name == 'Non-planar angles':
            ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
            ax.set_xticks([-np.pi / 2, np.pi / 2])
            ax.set_xticklabels(['$-\\frac{\pi}{2}$', '$\\frac{\pi}{2}$'])
            ax.set_xlabel('$\\phi$')
            ax.set_yticks([0, 0.6])

    plt.tight_layout()
    plt.show()


def directly_derive_parameters():
    """
    Directly derive model parameters from the data.
    """
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=False,
        include_K_options=False,
        include_planarity_options=True,
        include_helicity_options=False,
        include_manoeuvre_options=True,
        include_approximation_options=True,
        include_pe_options=False,
        include_fractal_dim_options=False,
        include_video_options=False,
        validate_source=True,
    )

    # Generate or load data
    ds, low_speeds, high_speeds, T = _generate_or_load_directly_derived_params(args, rebuild_cache=True,
                                                                               cache_only=False)

    M = T
    # M[:2] *= 25  # 25 frames per second
    rates = {
        '01': M[0, 1],
        '10': M[1, 0],
        '02': M[0, 2],
        '12': M[1, 2],
        '20': M[2, 0],
        '21': M[2, 1],
    }

    # Get the low and high speed statistics
    speed_stats = {
        'low_mu': np.mean(low_speeds),
        'low_std': np.std(low_speeds),
        'high_mu': np.mean(high_speeds),
        'high_std': np.std(high_speeds),
    }


def _evaluate_pe(x, shared_pe_args, approx_args, data_values):
    # Build the parameters
    params = PEParameters(
        **shared_pe_args,
        rate_01=x[0],
        rate_10=x[1],
        rate_02=x[2],
        rate_20=x[3],
        speeds_0_mu=x[4],
        speeds_0_sig=x[5],
        speeds_1_mu=x[6],
        speeds_1_sig=x[7],
        theta_dist_params=(x[8], 0, x[9], x[10], np.pi, x[11]),
        phi_dist_params=(0, x[12]),
        delta_max=x[13]
    )

    # Generate the trajectories
    SS = SimulationState(params, no_cache=True, quiet=True)

    # Use the same approximation parameters for the simulation runs as for real data
    stats = SS.get_approximation_statistics(**approx_args)

    # Calculate the statistical tests between the distributions
    test_results = np.zeros(len(data_values))
    for i, k in enumerate(data_values.keys()):
        vals_real = data_values[k][0]
        vals_sim = stats[k][0]
        test_stat, p_value = kstest(vals_real, vals_sim)

        # Take the inverse log of the p-value (so smaller is better and very small p_values don't ruin the score)
        test_results[i] = np.log(1 / max(1e-20, p_value))

    # The score for this parameter set is the product of the test results (smaller is better)
    score = np.prod(test_results)

    return score

def _calculate_evolved_parameters(
        args: Namespace,
        ds: Dataset,
):
    """
    Calculate the evolved parameters.
    """
    logger.info('Calculating evolved parameters.')
    dist_keys = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Set the approximation parameters
    approx_args = dict(
        error_limits=[args.approx_error_limit],
        planarity_window=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.smoothing_window_K,
        smooth_K_first=args.smoothing_window_K,
    )

    # Generate or load tumble/run values
    stats = generate_or_load_ds_statistics(
        ds=ds,
        rebuild_cache=args.regenerate,
        **approx_args
    )
    # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles
    data_values = {k: stats[i + 1] for i, k in enumerate(dist_keys)}

    # Define the initial model parameter values
    p0 = np.array([
        0.1 * args.sim_dt,  # r01
        0.1 * args.sim_dt,  # r10
        0.1 * args.sim_dt,  # r02
        0.8,  # r20
        0.001,  # speed0_mu
        0.0005,  # speed0_sig
        0.007,  # speed1_mu
        0.001,  # speed1_sig
        1.,  # theta_w1
        1.25,  # theta_sig1
        0.35,  # theta_w2
        0.8,  # theta_sig2
        0.3,  # phi_sig
        5,  # delta_max
    ])

    # Define the bounds
    eps = 1e-3
    bounds = [
        (eps, 1 - eps),  # r01
        (eps, 1 - eps),  # r10
        (eps, 1 - eps),  # r02
        (eps, 1 - eps),  # r20
        (0, 0.1),  # speed0_mu
        (0, 0.01),  # speed0_sig
        (0, 0.1),  # speed1_mu
        (0, 0.01),  # speed1_sig
        (0, 2),  # theta_w1
        (0, 10),  # theta_sig1
        (0, 2),  # theta_w2
        (0, 10),  # theta_sig2
        (0, 10),  # phi_sig
        (0, 10),  # delta_max
    ]

    # Define the shared PE parameters
    shared_pe_args = dict(
        batch_size=args.batch_size,
        duration=args.sim_duration,
        dt=args.sim_dt,
        n_steps=int(args.sim_duration / args.sim_dt),
        theta_dist_type='2norm',
        phi_dist_type='norm',
        delta_type='quadratic',
    )

    # Update the approximation args to include noise and smoothing
    approx_args['noise_scale'] = args.approx_noise
    approx_args['smoothing_window'] = args.smoothing_window

    # Minimize the objective function
    logger.info('Running differential evolution.')
    result = differential_evolution(
        _evaluate_pe,
        bounds,
        args=(shared_pe_args, approx_args, data_values),
        strategy='best1bin',
        x0=p0,
        workers=1,
        disp=True,
        popsize=2,
        maxiter=2,
        # tol=0.1,
        # atol=0,
        constraints=[
            NonlinearConstraint(lambda x: x[6] - x[4], 0, np.inf),  # slow speed < fast speed
        ]
    )

    logger.info('Optimisation complete.')
    print(result)


def _generate_or_load_evolved_params(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / ('evolved' + _identifiers(args))
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    T = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            low_speeds = data['low_speeds']
            high_speeds = data['high_speeds']
            T = data['T']
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            T = None
            logger.warning(f'Could not load cache: {e}')

    if T is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        low_speeds, high_speeds, T = _calculate_evolved_parameters(args, ds)
        data = {
            'low_speeds': low_speeds,
            'high_speeds': high_speeds,
            'T': T,
        }
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, low_speeds, high_speeds, T


def evolve_parameters():
    """
    Evolve model parameters to match data distributions.
    """
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=False,
        include_K_options=False,
        include_planarity_options=True,
        include_helicity_options=False,
        include_manoeuvre_options=True,
        include_approximation_options=True,
        include_pe_options=True,
        include_fractal_dim_options=False,
        include_video_options=False,
        validate_source=True,
    )

    assert args.batch_size is not None, 'Must provide a batch size!'
    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Generate or load data
    res = _generate_or_load_evolved_params(args, rebuild_cache=True, cache_only=False)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    # directly_derive_parameters()
    evolve_parameters()
