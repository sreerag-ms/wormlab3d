import os
import shutil
from argparse import Namespace
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import dill
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.display import Display
from scipy.stats import kstest

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, N_WORKERS, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, PEParameters
from wormlab3d.particles.simulation_state import SimulationState
from wormlab3d.particles.tumble_run import find_approximation, generate_or_load_ds_statistics
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import calculate_speeds

show_plots = True
save_plots = False
# show_plots = False
# save_plots = True
interactive_plots = False
img_extension = 'svg'

DATA_CACHE_PATH = LOGS_PATH / 'cache'
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)

DIST_KEYS = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']


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
    avg_op = lambda x: np.quantile(x, 0.03)

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
        include_pe_options=True,
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

    print('rates', rates)
    print('speed_stats', speed_stats)


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
    test_stats = np.zeros(len(data_values))
    p_values = np.zeros(len(data_values))
    test_results = np.zeros(len(data_values))
    for i, k in enumerate(data_values.keys()):
        vals_real = data_values[k][0]
        vals_sim = stats[k][0]
        test_stat, p_value = kstest(vals_real, vals_sim)
        test_stats[i] = test_stat
        p_values[i] = p_value

        # Prioritise the run statistics
        if k in ['durations', 'speeds']:
            test_results[i] = test_stat**2
        else:
            test_results[i] = 0.5 * test_stat**2

    # The score for this parameter set is the sum of the test results (smaller is better)
    score = np.sum(test_results)

    return {
        'score': score,
        'test_stats': test_stats,
        'p_values': p_values,
        'test_results': test_results,
        'vals': stats,
    }


class PEProblem(ElementwiseProblem):
    def __init__(self, bounds, shared_pe_args, approx_args, data_values, **kwargs):
        bounds = np.array(bounds)
        self.shared_pe_args = shared_pe_args
        self.approx_args = approx_args
        self.data_values = data_values

        super().__init__(
            n_var=len(bounds),
            n_obj=1,
            n_ieq_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = _evaluate_pe(
            x,
            self.shared_pe_args,
            self.approx_args,
            self.data_values
        )

        out['F'] = res['score']
        out['n_runs'] = len(res['vals']['durations'][0])
        out['n_tumbles'] = len(res['vals']['planar_angles'][0])
        for k, v in res.items():
            if k == 'score':
                continue
            if k in ['test_stats', 'p_values', 'test_results']:
                out[k] = v
            else:
                for k2, v2 in v.items():
                    out[f'{k}_{k2}'] = v2[0]

        # Add constraint for speed0 < speed1
        out['G'] = x[4] - x[6]

    def _format_dict(self, out, N, return_values_of):
        ret = super()._format_dict(out, N, return_values_of)

        # Make the value arrays the same length
        max_n_runs = int(ret['n_runs'].max())
        max_n_tumbles = int(ret['n_tumbles'].max())
        for k in DIST_KEYS:
            if k in ['durations', 'speeds']:
                max_length = max_n_runs
            else:
                max_length = max_n_tumbles
            ret[f'vals_{k}'] = np.stack([
                np.pad(ret[f'vals_{k}'][i], (0, max_length - len(ret[f'vals_{k}'][i])))
                for i in range(N)
            ])

        return ret


def _plot_population(
        data_values: Dict[str, List[np.ndarray]],
        algorithm: Algorithm,
        save_dir: Path
):
    """
    Plot histogram comparisons of the best scoring individuals against the real distributions.
    """

    # Select the best n individuals
    pop = algorithm.pop.copy()
    n_examples = min(5, len(pop))
    scores = np.array([ind.F[0] for ind in pop])
    positions = np.argsort(scores)
    pop = pop[positions[:n_examples]]

    # Plot histograms
    fig, axes = plt.subplots(n_examples, 5, figsize=(12, 2 + 2 * n_examples))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    colour_real = default_colours[0]
    colour_sim = default_colours[1]

    for i in range(n_examples):
        D_vals = pop[i].get('test_stats')
        for j, k in enumerate(DIST_KEYS):
            ax = axes[i, j]
            vals_real = data_values[k][0]
            vals_sim = pop[i].get(f'vals_{k}')
            if k in ['durations', 'speeds']:
                vals_sim = vals_sim[:int(pop[i].get('n_runs'))]
            else:
                vals_sim = vals_sim[:int(pop[i].get('n_tumbles'))]

            # Set titles
            if i == 0:
                if k == 'durations':
                    ax.set_title('Run durations')
                elif k == 'speeds':
                    ax.set_title('Run speeds')
                elif k == 'planar_angles':
                    ax.set_title('Planar angles')
                elif k == 'nonplanar_angles':
                    ax.set_title('Non-planar angles')
                elif k == 'twist_angles':
                    ax.set_title('Twist angles')

            # Label individuals' scores
            if j == 0:
                ax.set_ylabel(f'Score={scores[positions[i]]:.3E}')
            ax.set_xlabel(f'D={D_vals[j]:.3E}')

            # Set weights for speeds
            weights = [np.ones_like(vals_real), np.ones_like(vals_sim)]
            if k == 'speeds':
                durations = pop[i].get(f'vals_durations')[:int(pop[i].get('n_runs'))]
                weights = [data_values['durations'][0], durations]

            # Set bin range
            bin_range = None
            if k == 'planar_angles':
                bin_range = (-np.pi, np.pi)
            elif k == 'nonplanar_angles':
                bin_range = (-np.pi / 2, np.pi / 2)
            elif k == 'twist_angles':
                bin_range = (-np.pi, np.pi)

            # Set log scale for durations and speeds
            if k in ['durations', 'speeds']:
                ax.set_yscale('log')

            ax.hist(
                [vals_real, vals_sim],
                weights=weights,
                color=[colour_real, colour_sim],
                bins=21,
                density=True,
                alpha=0.75,
                range=bin_range
            )
            if k in ['planar_angles', 'twist_angles']:
                ax.set_xlim(left=-np.pi - 0.1, right=np.pi + 0.1)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
            if k == 'nonplanar_angles':
                ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
                ax.set_xticks([-np.pi / 2, np.pi / 2])
                ax.set_xticklabels(['$-\pi/2$', '$\pi/2$'])

    fig.tight_layout()
    plt.savefig(save_dir / f'{algorithm.n_gen:05d}.png')
    plt.close(fig)


def _calculate_evolved_parameters(
        args: Namespace,
        save_dir: Path,
        ds: Dataset,
):
    """
    Calculate the evolved parameters.
    """
    logger.info('Calculating evolved parameters.')

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
    data_values = {k: stats[i + 1] for i, k in enumerate(DIST_KEYS)}

    # Define the initial model parameter values
    p0 = np.array([
        args.rate_01 * args.sim_dt,
        args.rate_10 * args.sim_dt,
        args.rate_02 * args.sim_dt,
        args.rate_20,
        args.speeds_0_mu,
        args.speeds_0_sig,
        args.speeds_1_mu,
        args.speeds_1_sig,
        args.theta_dist_params[0],  # theta_w1
        args.theta_dist_params[2],  # theta_sig1
        args.theta_dist_params[3],  # theta_w2
        args.theta_dist_params[5],  # theta_sig2
        args.phi_dist_params[1],  # phi_sig
        args.nonp_pause_max,  # delta_max
    ])

    # Define the bounds
    eps = 1e-5
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

    # Initialize the thread pool and create the runner
    pool = ThreadPool(N_WORKERS)
    runner = StarmapParallelization(pool.starmap)

    # Initialise problem
    problem = PEProblem(
        bounds,
        shared_pe_args,
        approx_args,
        data_values,
        elementwise_runner=runner
    )

    # Algorithm setup
    algorithm_args = dict(
        pop_size=args.pop_size,
        variant=args.de_variant,
        CR=args.de_cr,
        dither='vector',
        jitter=False,
    )

    # Check for a checkpoint
    extra_args = {
        'p0': p0,
        'bounds': bounds,
        'shared_pe_args': shared_pe_args,
        'approx_args': approx_args,
        'data_values': data_values,
        'algorithm_args': algorithm_args,
    }
    checkpoint_path = (DATA_CACHE_PATH
                       / ('evolved_checkpoint_' + hash_data(to_dict(args)) + '_' + hash_data(extra_args) + '.cp'))
    algorithm = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                algorithm = dill.load(f)
                algorithm.display = Display(algorithm.output, verbose=algorithm.verbose, progress=algorithm.progress)
            algorithm.problem.elementwise_runner = runner
            logger.info(f'Restored checkpoint from {checkpoint_path} at generation {algorithm.n_gen}.')
        except Exception as e:
            logger.warning(f'Could not load checkpoint: {e}')
    if algorithm is None:
        # Set up optimisation algorithm
        algorithm = DE(**algorithm_args, sampling=LHS())
        algorithm.setup(
            problem,
            seed=1,
            termination=('n_gen', args.n_generations),
            verbose=True,
            progress=True
        )

    # Set up the plot directories
    hist_plots_dir = save_dir / 'histograms'
    hist_plots_dir.mkdir(parents=True, exist_ok=True)
    params_dir = save_dir / 'parameters'
    params_dir.mkdir(parents=True, exist_ok=True)

    # Minimize the objective function
    logger.info('Running optimisation.')
    display = algorithm.display
    while algorithm.has_next():
        algorithm.next()

        # Make plots
        _plot_population(data_values, algorithm, hist_plots_dir)

        # Save the checkpoint
        with open(checkpoint_path, 'wb') as f:
            algorithm.display = None
            dill.dump(algorithm, f)
            algorithm.display = display

        # Save the best parameters
        opt = algorithm.opt[0]
        opd = to_dict(opt)['data']
        opd['rate_01'] = opt.x[0]
        opd['rate_10'] = opt.x[1]
        opd['rate_02'] = opt.x[2]
        opd['rate_20'] = opt.x[3]
        opd['speeds_0_mu'] = opt.x[4]
        opd['speeds_0_sig'] = opt.x[5]
        opd['speeds_1_mu'] = opt.x[6]
        opd['speeds_1_sig'] = opt.x[7]
        opd['theta_w1'] = opt.x[8]
        opd['theta_sig1'] = opt.x[9]
        opd['theta_w2'] = opt.x[10]
        opd['theta_sig2'] = opt.x[11]
        opd['phi_sig'] = opt.x[12]
        opd['delta_max'] = opt.x[13]
        for key, value in opd.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.generic):
                opd[key] = value.tolist()
        with open(params_dir / f'{algorithm.n_gen:05d}.yml', 'w') as f:
            yaml.dump(opd, f)

    logger.info('Optimisation complete.')


def _generate_or_load_evolved_params(
        args: Namespace,
        save_dir: Path,
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
    cache_path = DATA_CACHE_PATH / ('evolved' + hash_data(to_dict(args)))
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
        low_speeds, high_speeds, T = _calculate_evolved_parameters(args, save_dir, ds)
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
        include_evolution_options=True,
        validate_source=False,
    )

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            setattr(args, k, v)
    print_args(args)

    assert args.batch_size is not None, 'Must provide a batch size!'
    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    # Generate or load data
    res = _generate_or_load_evolved_params(args, save_dir, rebuild_cache=True, cache_only=False)


def _calculate_approximation_speeds_and_tumbles(
        args: Namespace,
        ds: Dataset,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
]:
    """
    Calculate the approximation speeds and tumble idxs.
    """
    logger.info('Calculating speeds and tumbles.')
    # error_limits = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
    error_limit = 0.05

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Values
    speeds_ap = []
    tumble_idxs = []

    # Calculate the approximation for all trials
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Computing tumble-run model for trial={trial.id}.')
        args.trial = trial.id

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
            distance_first=500,
            distance_min=3,
            height_first=80,
            smooth_e0_first=251,
            smooth_K_first=251,
            max_attempts=50,
            quiet=False
        )
        X_approx, vertices, tumble_idxs_i, _, _, _, _, _, _, _, _ = approx

        # Split up the trajectory into the run sections to calculate scaled speeds per run in approximation
        logger.info('Calculating approximation speeds.')
        speed = calculate_speeds(X)  # / dt
        speed_ap = np.zeros_like(speed)
        v0 = 0
        for j in range(len(tumble_idxs_i) + 1):
            v1 = tumble_idxs_i[j] if j < len(tumble_idxs_i) else -1
            run_speed = speed[v0:v1]

            # Distance is reduced using the straight-line runs, so scale speeds accordingly
            assert np.allclose(vertices[j], X[v0])
            assert np.allclose(vertices[j + 1], X[v1])
            run_dist_og = np.linalg.norm(X[v0:v1 - 1] - X[v0 + 1:v1], axis=-1).sum()
            run_dist_ap = np.linalg.norm(vertices[j] - vertices[j + 1], axis=-1).sum()
            sf = run_dist_ap / run_dist_og  # always < 1
            speed_ap[v0:v1] = run_speed * sf

            v0 = v1

        tumble_idxs.append(tumble_idxs_i)
        speeds_ap.append(speed_ap)

    return speeds_ap, tumble_idxs


def _calculate_rates_and_speeds(
        speed_threshold: float,
        speeds_ap: List[np.ndarray],
        tumble_idxs: List[np.ndarray],
        avg_op: Callable = np.mean
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
]:
    """
    Calculate the transition rates for a given speed threshold between slow and fast.
    """
    low_speeds = np.zeros(len(speeds_ap))
    high_speeds = np.zeros(len(speeds_ap))

    # Transition matrix
    T = np.zeros((3, 3), dtype=int)

    # Calculate the rates for all trials
    for i, (speed_ap_i, tumble_idxs_i) in enumerate(zip(speeds_ap, tumble_idxs)):
        # Calculate the high and low speeds
        thresh_speed = np.quantile(speed_ap_i, speed_threshold)
        low_speeds[i] = avg_op(speed_ap_i[speed_ap_i <= thresh_speed])
        high_speeds[i] = avg_op(speed_ap_i[speed_ap_i > thresh_speed])

        # Split up the trajectory into the run sections to calculate transition rates
        v0 = 0
        for j in range(len(tumble_idxs_i) + 1):
            v1 = tumble_idxs_i[j] if j < len(tumble_idxs_i) else -1
            run_speed = speed_ap_i[v0:v1]

            # Get telegraph signal of above and below the mean trajectory speed
            states = (run_speed > thresh_speed).astype(int)

            # Previously, started with a turn (unless first)
            if v0 > 0:
                T[2, states[0]] += 1

            # Log speed-state transitions
            for (from_state, to_state) in zip(states[:-1], states[1:]):
                T[from_state, to_state] += 1

            # End with a turn (unless last)
            if j < len(tumble_idxs_i):
                # T[states[-1], 2] += 1
                # Restrict turns from the slow state
                if states[-1] == 1:
                    T[1, 0] += 1
                T[0, 2] += 1

            v0 = v1

    # Calculate the final transition matrix to get the rates
    M = T / T.sum(axis=1, keepdims=True)
    rates = {
        '01': M[0, 1],
        '10': M[1, 0],
        '02': M[0, 2],
        '12': M[1, 2],
        '20': M[2, 0],
        '21': M[2, 1],
    }

    # Get the low and high speed statistics
    speeds = {
        '0_mu': np.mean(low_speeds),
        '0_sig': np.std(low_speeds),
        '1_mu': np.mean(high_speeds),
        '1_sig': np.std(high_speeds),
    }

    return rates, speeds


def _evaluate_hybrid_pe(x, shared_pe_args, approx_args, data_values, trajectory_speeds, trajectory_tumble_idxs):
    # Calculate the rates and speeds using the new speed threshold
    rates, speeds = _calculate_rates_and_speeds(
        speed_threshold=x[0],
        speeds_ap=trajectory_speeds,
        tumble_idxs=trajectory_tumble_idxs,
        avg_op=np.mean
    )

    # Build the parameters
    params = PEParameters(
        **shared_pe_args,
        rate_01=rates['01'],
        rate_10=rates['10'],
        rate_02=rates['02'],
        rate_20=rates['20'],
        speeds_0_mu=speeds['0_mu'],
        speeds_0_sig=speeds['0_sig'],
        speeds_1_mu=speeds['1_mu'],
        speeds_1_sig=speeds['1_sig'],
        theta_dist_params=(x[1], 0, x[2], x[3], np.pi, x[4]),
        phi_dist_params=(0, x[5]),
        delta_max=x[6]
    )

    # Generate the trajectories
    SS = SimulationState(params, no_cache=True, quiet=True)

    # Use the same approximation parameters for the simulation runs as for real data
    stats = SS.get_approximation_statistics(**approx_args)

    # Calculate the statistical tests between the distributions
    test_stats = np.zeros(len(data_values))
    p_values = np.zeros(len(data_values))
    test_results = np.zeros(len(data_values))
    for i, k in enumerate(data_values.keys()):
        vals_real = data_values[k][0]
        vals_sim = stats[k][0]
        test_stat, p_value = kstest(vals_real, vals_sim)
        test_stats[i] = test_stat
        p_values[i] = p_value

        # Prioritise the run statistics
        if k in ['durations', 'speeds']:
            test_results[i] = test_stat**2
        else:
            test_results[i] = 0.5 * test_stat**2

    # The score for this parameter set is the sum of the test results (smaller is better)
    score = np.sum(test_results)

    return {
        'score': score,
        'test_stats': test_stats,
        'p_values': p_values,
        'test_results': test_results,
        'vals': stats,
        'rates': rates,
        'speeds': speeds
    }


class PEHybridProblem(ElementwiseProblem):
    def __init__(
            self,
            bounds,
            shared_pe_args,
            approx_args,
            data_values,
            trajectory_speeds,
            trajectory_tumble_idxs,
            **kwargs
    ):
        bounds = np.array(bounds)
        self.shared_pe_args = shared_pe_args
        self.approx_args = approx_args
        self.data_values = data_values
        self.trajectory_speeds = trajectory_speeds
        self.trajectory_tumble_idxs = trajectory_tumble_idxs

        super().__init__(
            n_var=len(bounds),
            n_obj=1,
            n_ieq_constr=0,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = _evaluate_hybrid_pe(
            x,
            self.shared_pe_args,
            self.approx_args,
            self.data_values,
            self.trajectory_speeds,
            self.trajectory_tumble_idxs,
        )

        out['F'] = res['score']
        out['n_runs'] = len(res['vals']['durations'][0])
        out['n_tumbles'] = len(res['vals']['planar_angles'][0])
        for k, v in res.items():
            if k == 'score':
                continue
            if k in ['test_stats', 'p_values', 'test_results']:
                out[k] = v
            elif k in ['rates', 'speeds']:
                for k2, v2 in v.items():
                    out[f'{k[:-1]}_{k2}'] = v2
            else:
                for k2, v2 in v.items():
                    out[f'{k}_{k2}'] = v2[0]

    def _format_dict(self, out, N, return_values_of):
        ret = super()._format_dict(out, N, return_values_of)

        # Make the value arrays the same length
        max_n_runs = int(ret['n_runs'].max())
        max_n_tumbles = int(ret['n_tumbles'].max())
        for k in DIST_KEYS:
            if k in ['durations', 'speeds']:
                max_length = max_n_runs
            else:
                max_length = max_n_tumbles
            ret[f'vals_{k}'] = np.stack([
                np.pad(ret[f'vals_{k}'][i], (0, max_length - len(ret[f'vals_{k}'][i])))
                for i in range(N)
            ])

        return ret


def _calculate_hybrid_parameters(
        args: Namespace,
        save_dir: Path,
        ds: Dataset,
):
    """
    Calculate the hybrid-evolved parameters.
    """
    logger.info('Calculating hybrid-evolved parameters.')

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
    data_values = {k: stats[i + 1] for i, k in enumerate(DIST_KEYS)}

    # Calculate the approximation speeds and tumble idxs
    speeds_ap, tumble_idxs = _calculate_approximation_speeds_and_tumbles(args, ds)

    # Define the initial model parameter values
    p0 = np.array([
        0.5,  # speed_threshold
        args.theta_dist_params[0],  # theta_w1
        args.theta_dist_params[2],  # theta_sig1
        args.theta_dist_params[3],  # theta_w2
        args.theta_dist_params[5],  # theta_sig2
        args.phi_dist_params[1],  # phi_sig
        args.nonp_pause_max,  # delta_max
    ])

    # Define the bounds
    eps = 1e-4
    bounds = [
        (eps, 1 - eps),  # speed_threshold
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

    # Initialize the thread pool and create the runner
    pool = ThreadPool(N_WORKERS)
    runner = StarmapParallelization(pool.starmap)

    # Initialise problem
    problem = PEHybridProblem(
        bounds,
        shared_pe_args,
        approx_args,
        data_values,
        speeds_ap,
        tumble_idxs,
        elementwise_runner=runner
    )

    # Algorithm setup
    algorithm_args = dict(
        pop_size=args.pop_size,
        variant=args.de_variant,
        CR=args.de_cr,
        dither='vector',
        jitter=False,
    )

    # Check for a checkpoint
    extra_args = {
        'p0': p0,
        'bounds': bounds,
        'shared_pe_args': shared_pe_args,
        'approx_args': approx_args,
        'data_values': data_values,
        'algorithm_args': algorithm_args,
    }
    checkpoint_path = (DATA_CACHE_PATH
                       / ('hybrid_checkpoint_' + hash_data(to_dict(args)) + '_' + hash_data(extra_args) + '.cp'))
    algorithm = None
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                algorithm = dill.load(f)
                algorithm.display = Display(algorithm.output, verbose=algorithm.verbose, progress=algorithm.progress)
            algorithm.problem.elementwise_runner = runner
            logger.info(f'Restored checkpoint from {checkpoint_path} at generation {algorithm.n_gen}.')
        except Exception as e:
            logger.warning(f'Could not load checkpoint: {e}')
    if algorithm is None:
        # Set up optimisation algorithm
        algorithm = DE(**algorithm_args, sampling=LHS())
        algorithm.setup(
            problem,
            seed=1,
            termination=('n_gen', args.n_generations),
            verbose=True,
            progress=True
        )

    # Set up the plot directories
    hist_plots_dir = save_dir / 'histograms'
    hist_plots_dir.mkdir(parents=True, exist_ok=True)
    params_dir = save_dir / 'parameters'
    params_dir.mkdir(parents=True, exist_ok=True)

    # Minimize the objective function
    logger.info('Running optimisation.')
    display = algorithm.display
    while algorithm.has_next():
        algorithm.next()

        # Make plots
        _plot_population(data_values, algorithm, hist_plots_dir)

        # Save the checkpoint
        with open(checkpoint_path, 'wb') as f:
            algorithm.display = None
            dill.dump(algorithm, f)
            algorithm.display = display

        # Save the best parameters
        opt = algorithm.opt[0]
        opd = to_dict(opt)['data']
        opd['speed_threshold'] = opt.x[0]
        opd['theta_w1'] = opt.x[1]
        opd['theta_sig1'] = opt.x[2]
        opd['theta_w2'] = opt.x[3]
        opd['theta_sig2'] = opt.x[4]
        opd['phi_sig'] = opt.x[5]
        opd['delta_max'] = opt.x[6]
        for key, value in opd.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.generic):
                opd[key] = value.tolist()
        with open(params_dir / f'{algorithm.n_gen:05d}.yml', 'w') as f:
            yaml.dump(opd, f)

    logger.info('Optimisation complete.')


def _generate_or_load_hybrid_params(
        args: Namespace,
        save_dir: Path,
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
    cache_path = DATA_CACHE_PATH / ('hybrid' + hash_data(to_dict(args)))
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
        low_speeds, high_speeds, T = _calculate_hybrid_parameters(args, save_dir, ds)
        data = {
            'low_speeds': low_speeds,
            'high_speeds': high_speeds,
            'T': T,
        }
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, low_speeds, high_speeds, T


def hybrid_evolve_parameters():
    """
    Evolve just the derivation parameter to match data distributions.
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
        include_evolution_options=True,
        validate_source=False,
    )

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            setattr(args, k, v)
    print_args(args)

    assert args.batch_size is not None, 'Must provide a batch size!'
    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    # Generate or load data
    res = _generate_or_load_hybrid_params(args, save_dir, rebuild_cache=True, cache_only=False)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    # directly_derive_parameters()
    # evolve_parameters()
    hybrid_evolve_parameters()
