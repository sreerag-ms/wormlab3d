from argparse import Namespace
from typing import Tuple, List, Dict

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from wormlab3d import LOGS_PATH, logger
from wormlab3d.data.model import Dataset
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.util import normalise, orthogonalise
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args, calculate_pcas, PCACache
from wormlab3d.trajectories.util import smooth_trajectory, get_deltas_from_args


def calculate_curvature(
        e0: np.ndarray,
        smooth_e0: int = 101,
        smooth_K: int = 101
) -> np.ndarray:
    """
    Identify tumbles as large changes in direction over a short period of time - ie, high curvature
    e0 = normalised tangent to curve => d/dt(e0) = K(t) = curvature vector
    """
    e0 = smooth_trajectory(e0, window_len=smooth_e0)
    K = np.gradient(e0, 1 / (len(e0) - 1), axis=0, edge_order=1)

    # Smooth the curvature
    K = smooth_trajectory(K, window_len=smooth_K)

    # Curvature magnitude
    k = np.linalg.norm(K, axis=-1)

    return k


def get_approximate(
        X: np.ndarray,
        k: np.ndarray,
        distance: int,
        height: int = 50,
        planarity_window: int = 3,
        quiet: bool = False
):
    """
    Calculate a tumble-and-run approximation to a trajectory X with curvature k.
    The distance parameter sets a minimum distance that subsequent curvature peaks must be.
    """
    if not quiet:
        logger.info(f'Calculating approximation to X at distance {distance}, height={height}.')
    T = len(X)

    # Take centre of mass
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Find peaks in curvature
    tumble_idxs, section_props = find_peaks(k, distance=distance, height=height)
    N = len(tumble_idxs)
    if N <= 1:
        raise RuntimeError('Too few peaks found! Try decreasing distance / height.')

    # Build the approximation and calculate run durations
    run_durations = np.zeros(N - 1)
    X_approx = np.zeros_like(X)
    X_approx[0] = X[0]
    x = X[0]
    start_idx = 0
    for i, tumble_idx in enumerate(tumble_idxs):
        end_idx = tumble_idx
        run_start = x[None, :]
        run_end = X[tumble_idx]
        run_steps = end_idx - start_idx
        if i > 0:
            run_durations[i - 1] = run_steps
        y = np.linspace(0, 1, run_steps)[:, None]
        X_approx[start_idx:end_idx] = (1 - y) * run_start + y * run_end
        x = run_end
        start_idx = tumble_idx

    # Add final run
    end_idx = T
    run_start = x[None, :]
    run_end = X[-1]
    run_steps = end_idx - start_idx
    y = np.linspace(0, 1, run_steps)[:, None]
    X_approx[start_idx:end_idx] = (1 - y) * run_start + y * run_end

    # Add vertices for start and end points
    vertices = np.concatenate([
        X[0][None, :],
        X[tumble_idxs],
        X[-1][None, :],
    ], axis=0)

    # Calculate run speeds
    run_distances = np.linalg.norm(vertices[2:-1] - vertices[1:-2], axis=-1)
    run_speeds = run_distances / run_durations

    # Calculate PCA along the vertices
    pcas = calculate_pcas(vertices, window_size=min(vertices.shape[0] - 1, planarity_window), parallel=False)
    pcas = PCACache(pcas)

    # Pad the PCAs to match the number of tumbles/vertices
    components = pcas.components.copy()
    diff = N - components.shape[0] + 1
    components = np.concatenate([
        np.repeat(components[0][None, ...], repeats=np.ceil(diff / 2), axis=0),
        components,
        np.repeat(components[-1][None, ...], repeats=np.floor(diff / 2), axis=0)
    ], axis=0)

    # Calculate e0 as normalised line segments between tumbles
    e0 = normalise(vertices[1:] - vertices[:-1])

    # e1 is the frame vector pointing out into the principal plane of the curve
    v1 = components[:, 1].copy()

    # Orthogonalise the pca planar direction vector against the trajectory to get e1
    e1 = normalise(orthogonalise(v1, e0))

    # e2 is the remaining cross product
    e2 = normalise(np.cross(e0, e1))

    # Duplicate final frame to line things up
    e0 = np.r_[e0, e0[-1][None, ...]]
    e1 = np.r_[e1, e1[-1][None, ...]]
    e2 = np.r_[e2, e2[-1][None, ...]]

    # Calculate the angles
    planar_angles = np.zeros(N)
    nonplanar_angles = np.zeros(N)
    twist_angles = np.zeros(N)
    for i in range(N):
        prev_frame = np.stack([e0[i], e1[i], e2[i]])
        next_frame = np.stack([e0[i + 1], e1[i + 1], e2[i + 1]])
        R, rmsd = Rotation.align_vectors(prev_frame, next_frame)
        R = R.as_matrix()

        # Decompose rotation matrix R into the axes of A
        A = prev_frame
        rp = Rotation.from_matrix(A.T @ R @ A)
        a2, a1, a0 = rp.as_euler('zyx')
        # xp = A @ Rotation.from_rotvec(a0 * np.array([1, 0, 0])).as_matrix() @ A.T
        # yp = A @ Rotation.from_rotvec(a1 * np.array([0, 1, 0])).as_matrix() @ A.T
        # zp = A @ Rotation.from_rotvec(a2 * np.array([0, 0, 1])).as_matrix() @ A.T
        # assert np.allclose(xp @ yp @ zp, R)
        planar_angles[i] = a2  # Rotation about e2
        nonplanar_angles[i] = a1  # Rotation about e1
        twist_angles[i] = a0  # Rotation about e0

    return X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2


def find_approximation(
        X: np.ndarray,
        e0: np.ndarray,
        error_limit: float,
        planarity_window_vertices: int = 3,
        distance_first: int = 500,
        distance_min: int = 3,
        height_first: int = 50,
        smooth_e0_first: int = 101,
        smooth_K_first: int = 101,
        max_attempts: int = 10,
        quiet: bool = False
):
    """
    Find an approximation to the trajectory at a given error limit.
    """
    if not quiet:
        logger.info(f'Finding approximation at error limit={error_limit:.3f}.')
    mse = np.inf
    attempts = 0

    if X.ndim == 3:
        X = X.mean(axis=1)

    approx = None
    distance = distance_first
    height = height_first
    smooth_e0 = smooth_e0_first
    smooth_K = smooth_K_first

    while mse > error_limit and attempts < max_attempts:
        k = calculate_curvature(e0, smooth_e0=smooth_e0, smooth_K=smooth_K)

        # Calculate the approximation, tumbles and runs
        try:
            approx = get_approximate(X, k, distance=distance, height=height, planarity_window=planarity_window_vertices,
                                     quiet=quiet)
            mse = np.mean(np.sum((X - approx[0])**2, axis=-1))
        except RuntimeError as e:
            if not quiet:
                logger.warning(e)
            mse = np.inf

        attempts += 1

        if mse > error_limit:
            distance = max(distance_min, distance - 10)
            if attempts > 5:
                height = max(10, height - 1)
            smooth_e0 = max(11, smooth_e0 - 6)
            smooth_K = max(11, smooth_K - 6)

    if approx is None:
        raise RuntimeError('Failed to find approximation!')

    return approx, distance, height, smooth_e0, smooth_K


def generate_or_load_ds_statistics(
        ds: Dataset,
        error_limits: List[float],
        min_run_speed_duration: Tuple[float, float] = (0.01, 60.),
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate or load tumble/run values
    """
    args = get_args(validate_source=False)
    cache_path = LOGS_PATH / f'ds={ds.id}_errors={",".join([str(err) for err in error_limits])}' \
                             f'_pw={args.planarity_window_vertices}' \
                             f'_mrsd={min_run_speed_duration[0]:.2f},{min_run_speed_duration[1]:.1f}'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    if not rebuild_cache and cache_fn.exists():
        data = np.load(cache_fn)
        trajectory_lengths = data[f'trajectory_lengths'].astype(np.uint32)
        durations = [data[f'durations_{i}'] for i in range(len(error_limits))]
        speeds = [data[f'speeds_{i}'] for i in range(len(error_limits))]
        planar_angles = [data[f'planar_angles_{i}'] for i in range(len(error_limits))]
        nonplanar_angles = [data[f'nonplanar_angles_{i}'] for i in range(len(error_limits))]
        twist_angles = [data[f'twist_angles_{i}'] for i in range(len(error_limits))]
    else:
        trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles \
            = _calculate_dataset_values(ds, error_limits, min_run_speed_duration)
        save_arrs = {'trajectory_lengths': trajectory_lengths}
        for i in range(len(error_limits)):
            save_arrs[f'durations_{i}'] = durations[i]
            save_arrs[f'speeds_{i}'] = speeds[i]
            save_arrs[f'planar_angles_{i}'] = planar_angles[i]
            save_arrs[f'nonplanar_angles_{i}'] = nonplanar_angles[i]
            save_arrs[f'twist_angles_{i}'] = twist_angles[i]
        np.savez(cache_path, **save_arrs)

    return trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles


def generate_or_load_ds_msds(
        ds: Dataset,
        args: Namespace,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate or load tumble/run values
    """

    # Generate or load MSDs
    msds_path = LOGS_PATH / f'ds={ds.id}_msds_d={args.min_delta}-{args.max_delta}_ds={args.delta_step}'
    msds_fn = msds_path.with_suffix(msds_path.suffix + '.npz')
    msds = None
    if not rebuild_cache and msds_fn.exists():
        try:
            data = np.load(msds_fn)
            msds_all = data['msds_all']
            msds = {}
            for trial in ds.include_trials:
                k = f'msd_{trial.id}'
                if k in data:
                    msds[trial.id] = data[k]
        except Exception as e:
            logger.warning(f'Could not load MSDs: {e}')
            msds = None
    if msds is None:
        msds, msds_all = _calculate_dataset_msds()
        save_arrs = {'msds_all': msds_all}
        for trial in ds.include_trials:
            if trial.id in msds:
                save_arrs[f'msd_{trial.id}'] = msds[trial.id]
        np.savez(msds_path, **save_arrs)

    return msds_all, msds


def _calculate_dataset_values(
        ds: Dataset,
        error_limits: List[float],
        min_run_speed_duration: Tuple[float, float] = (0.01, 60.)
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Calculate the tumble/run approximation values across a dataset.
    """
    args = get_args(validate_source=False)
    assert args.planarity_window is not None

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    # Values
    trajectory_lengths = np.zeros(len(ds.include_trials), dtype=np.uint32)
    durations = {i: [] for i in range(len(error_limits))}
    speeds = {i: [] for i in range(len(error_limits))}
    planar_angles = {i: [] for i in range(len(error_limits))}
    nonplanar_angles = {i: [] for i in range(len(error_limits))}
    twist_angles = {i: [] for i in range(len(error_limits))}

    # Calculate the model for all trials
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Computing tumble-run model for trial={trial.id}.')
        args.trial = trial.id
        dt = 1 / trial.fps

        X = get_trajectory_from_args(args)
        pcas = get_pca_cache_from_args(args)
        e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)
        trajectory_lengths[i] = X.shape[0]

        # Take centre of mass
        if X.ndim == 3:
            X = X.mean(axis=1)
        X -= X.mean(axis=0)

        # Calculate coefficients of variation for all params for all trials at all distances
        distance = 500
        distance_min = 3
        height = 100
        smooth_e0 = 201
        smooth_K = 201

        for j, error_limit in enumerate(error_limits):
            approx, distance, height, smooth_e0, smooth_K \
                = find_approximation(X, e0, error_limit, args.planarity_window_vertices, distance, distance_min,
                                     height, smooth_e0, smooth_K, max_attempts=50)
            X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, _, _, _ = approx

            # Put in time units
            run_durations *= dt
            run_speeds /= dt

            # Discard long runs where the distance travelled is too small
            include_idxs = np.unique(
                np.concatenate([
                    np.argwhere(run_speeds > min_run_speed_duration[0]),
                    np.argwhere(run_durations < min_run_speed_duration[1])
                ])
            )
            run_durations = run_durations[include_idxs]
            run_speeds = run_speeds[include_idxs]

            durations[j].extend(run_durations.tolist())
            speeds[j].extend(run_speeds.tolist())
            planar_angles[j].extend(planar_angles_j.tolist())
            nonplanar_angles[j].extend(nonplanar_angles_j.tolist())
            twist_angles[j].extend(twist_angles_j.tolist())

    return trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles


def _calculate_dataset_msds():
    """
    Calculate the MSDs for trials in a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.tracking_only = True

    msds_all = np.zeros(len(deltas))
    msds = {}
    d_all = {delta: [] for delta in deltas}

    logger.info(f'Calculating displacements.')
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Calculating MSD for trial={trial.id} ({i + 1}/{len(ds.include_trials)}).')
        msds_i = []
        args.trial = trial.id
        X = get_trajectory_from_args(args)
        if X.ndim == 3:
            X = X.mean(axis=1)
        X -= X.mean(axis=0)
        if X.shape[0] < 9 * 60 * 25:
            continue
        for delta in deltas:
            if delta > X.shape[0] / 3:
                continue
            d = np.sum((X[delta:] - X[:-delta])**2, axis=-1)
            d_all[delta].append(d)
            msds_i.append(d.mean())

        msds[trial.id] = np.array(msds_i)

    for i, delta in enumerate(deltas):
        msds_all[i] = np.concatenate(d_all[delta]).mean()

    return msds, msds_all
