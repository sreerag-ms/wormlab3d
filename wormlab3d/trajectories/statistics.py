from argparse import Namespace
from typing import Dict, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from wormlab3d import logger
from wormlab3d.data.model import Trial
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate, find_approximation
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.postures.helicities import calculate_helicities, calculate_trajectory_helicities
from wormlab3d.toolkit.util import orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args
from wormlab3d.trajectories.util import calculate_speeds


def _calculate_posture_nonplanarity(X: np.ndarray) -> np.ndarray:
    """
    Average planarity of the postures in a window.
    """
    assert X.ndim == 3
    nonp = np.zeros(len(X))
    for i, Xi in enumerate(X):
        pca_i = PCA(svd_solver='full', copy=True, n_components=3)
        pca_i.fit(Xi)
        r = pca_i.explained_variance_ratio_.T
        nonp[i] = r[2] / np.sqrt(r[1] * r[0])
    return nonp


def calculate_trial_turn_statistics(
        args: Namespace,
        window_size: int,
) -> Dict[str, np.ndarray]:
    """
    Calculate the angles and distances for each turn in a trial trajectory.
    """
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'Calculating trial turn statistics for trial={trial.id}.')
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)
    k = calculate_curvature(e0, smooth_e0=args.smoothing_window_K, smooth_K=args.smoothing_window_K)
    w2 = int(window_size / 2)

    # Take centre of mass
    X_full = None
    if X.ndim == 3:
        if X.shape[1] != 1:
            X_full = X - X.mean(axis=0)
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    if args.approx_error_limit is not None:
        # Find an approximation for specified error limit
        approx, distance, height, smooth_e0, smooth_K = find_approximation(
            X=X,
            e0=e0,
            error_limit=args.approx_error_limit,
            planarity_window_vertices=args.planarity_window_vertices,
            distance_first=args.approx_distance,
            distance_min=w2,
            max_iterations=args.approx_max_attempts,
            quiet=False
        )
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx
    else:
        # Calculate the approximation, tumbles and runs for specified parameters
        approx = get_approximate(X, k, distance=w2, height=args.approx_curvature_height)
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

    # Set up outputs
    start_idxs = []
    end_idxs = []
    distances = []
    window_sizes_actual = []
    pcas = []
    thetas = []
    phis = []
    psis = []
    nonp = []
    etas = []
    v_in = []
    v_out = []
    v_in_theta = []
    v_out_theta = []
    v_in_phi = []
    v_out_phi = []
    v_in_psi = []
    v_out_psi = []
    nonp_postures = []
    nonp_postures_max = []

    # Recalculate the angles using the full trajectory and a fixed PCA window
    for i, tumble_idx in enumerate(tumble_idxs):
        # Discard first and last tumbles
        if i == 0 or i == len(tumble_idxs) - 1:
            continue

        # Get a fixed time window around the turn
        start_idx = max(0, tumble_idx - w2)
        start_idxs.append(start_idx)
        end_idx = min(X.shape[0] - 1, tumble_idx + (window_size - w2))
        end_idxs.append(end_idx)
        X_window = X[start_idx:end_idx]
        window_sizes_actual.append(end_idx - start_idx)

        # Calculate distance travelled across windowed trajectory
        distances.append(np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum())

        # Calculate the PCA for the complete window
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)
        pcas.append(pca)

        # Nonplanarity of the manoeuvre window
        r = pca.explained_variance_ratio_.T
        nonp.append(r[2] / np.sqrt(r[1] * r[0]))

        # Calculate the PCA for the windows before and after the turn
        pca_in = PCA(svd_solver='full', copy=True, n_components=3)
        pca_in.fit(X[start_idx:tumble_idx])
        pca_out = PCA(svd_solver='full', copy=True, n_components=3)
        pca_out.fit(X[tumble_idx:end_idx])

        # Angle between incoming/outgoing planes
        eta = min(
            calculate_angle(pca_in.components_[2], pca_out.components_[2]),
            calculate_angle(pca_in.components_[2], -pca_out.components_[2])
        )
        etas.append(eta)

        # Incoming and outgoing trajectories
        v_in_i = X[tumble_idx] - X[start_idx]
        v_out_i = X[end_idx] - X[tumble_idx]
        v_io = v_out_i - v_in_i

        # Project onto the principal plane to find planar angle.
        v_in_theta_i = orthogonalise(v_in_i, pca.components_[2])
        v_out_theta_i = orthogonalise(v_out_i, pca.components_[2])
        if np.linalg.norm(v_io) < max(np.linalg.norm(v_in_i), np.linalg.norm(v_out_i)):
            theta = min(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))
        else:
            theta = max(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))
        thetas.append(theta)

        # Orthogonalise against 2nd principal component to find non-planar angle.
        v_in_phi_i = orthogonalise(v_in_i, pca.components_[1])
        v_out_phi_i = orthogonalise(v_out_i, pca.components_[1])
        # phis[i] = calculate_angle(v_in_phi, v_out_phi)
        phi = min(calculate_angle(v_in_phi_i, v_out_phi_i), calculate_angle(v_in_phi_i, -v_out_phi_i))
        phis.append(phi)

        # Orthogonalise against 1st principal component to find the final angle.
        v_in_psi_i = orthogonalise(v_in_i, pca.components_[0])
        v_out_psi_i = orthogonalise(v_out_i, pca.components_[0])
        psi = min(calculate_angle(v_in_psi_i, v_out_psi_i), calculate_angle(v_in_psi_i, -v_out_psi_i))
        psis.append(psi)

        v_in.append(v_in_i)
        v_out.append(v_out_i)
        v_in_theta.append(v_in_theta_i)
        v_out_theta.append(v_out_theta_i)
        v_in_phi.append(v_in_phi_i)
        v_out_phi.append(v_out_phi_i)
        v_in_psi.append(v_in_psi_i)
        v_out_psi.append(v_out_psi_i)

        # Average planarity of the postures
        if X_full is not None:
            nonp_postures_i = _calculate_posture_nonplanarity(X_full[start_idx:end_idx])
            nonp_postures.append(nonp_postures_i.mean())
            nonp_postures_max.append(nonp_postures_i.max())

    # Calculate run stats
    nonp_runs = []
    run_distances = []
    run_speeds = []
    for i, tumble_idx in enumerate(tumble_idxs):
        # Skip the first tumble
        if i == 0:
            continue
        start_idx = tumble_idxs[i - 1] + w2
        end_idx = tumble_idxs[i] - w2
        X_window = X[start_idx:end_idx]

        # Calculate distance and speed
        distance = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()
        speed = np.mean(distance / (end_idx - start_idx) * trial.fps)
        if args.approx_min_run_distance is not None and distance < args.approx_min_run_distance:
            continue
        if args.approx_min_run_speed is not None and speed < args.approx_min_run_speed:
            continue
        run_distances.append(distance)
        run_speeds.append(speed)

        # Calculate the non-planarity of the run
        pca_run = PCA(svd_solver='full', copy=True, n_components=3)
        pca_run.fit(X_window)
        r = pca_run.explained_variance_ratio_.T
        nonp_runs.append(r[2] / np.sqrt(r[1] * r[0]))

    return {
        'k': k,
        'vertices': vertices,
        'tumble_idxs': tumble_idxs,
        'start_idxs': start_idxs,
        'end_idxs': end_idxs,
        'distances': np.array(distances),
        'speeds': np.array(distances) / np.array(window_sizes_actual) * trial.fps,
        'pcas': pcas,
        'thetas': np.array(thetas),
        'phis': np.array(phis),
        'psis': np.array(psis),
        'nonp': np.array(nonp),
        'etas': np.array(etas),
        'v_in': np.array(v_in),
        'v_out': np.array(v_out),
        'v_in_theta': np.array(v_in_theta),
        'v_out_theta': np.array(v_out_theta),
        'v_in_phi': np.array(v_in_phi),
        'v_out_phi': np.array(v_out_phi),
        'v_in_psi': np.array(v_in_psi),
        'v_out_psi': np.array(v_out_psi),
        'nonp_postures': np.array(nonp_postures),
        'nonp_postures_max': np.array(nonp_postures_max),
        'run_durations': run_durations,
        'run_speeds': np.array(run_speeds),
        'run_distances': np.array(run_distances),
        'nonp_runs': np.array(nonp_runs),
    }


def calculate_trial_run_statistics(
        args: Namespace,
) -> Dict[str, np.ndarray]:
    """
    Calculate the run statistics for each turn in a trial trajectory.
    Runs are defined as trajectory sections under the curvature threshold.
    """
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'Calculating trial run statistics for trial={trial.id}.')
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

    # Take centre of mass
    if X.ndim == 3:
        X_full = X - X.mean(axis=0)
        X = X.mean(axis=1)
        if X_full.shape[1] == 1:
            X_full = None
    else:
        X_full = None
    X -= X.mean(axis=0)

    # Calculate the curvature and threshold
    k = calculate_curvature(e0, smooth_e0=args.smoothing_window_K, smooth_K=args.smoothing_window_K)
    r_state = np.r_[[0], k < args.approx_run_max_K, [0]]

    # Find straight line sections
    run_idxs, section_props = find_peaks(r_state, width=args.approx_min_run_duration, height=1)
    N = len(run_idxs)
    if N <= 1:
        raise RuntimeError('Too few peaks found! Try decreasing duration / increasing height.')

    # Set up outputs
    start_idxs = []
    end_idxs = []
    distances = []
    speeds = []
    nonp = []
    nonp_postures = []
    nonp_postures_max = []

    for i, run_idx in enumerate(run_idxs):
        start_idx = section_props['left_bases'][i]
        start_idxs.append(start_idx)
        end_idx = section_props['right_bases'][i] - 1
        end_idxs.append(end_idx)
        X_window = X[start_idx:end_idx]

        # Calculate distance and average speed travelled across windowed trajectory
        distance = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()
        speed = np.mean(distance / (end_idx - start_idx) * trial.fps)
        if args.approx_min_run_distance is not None and distance < args.approx_min_run_distance:
            continue
        if args.approx_min_run_speed is not None and speed < args.approx_min_run_speed:
            continue
        distances.append(distance)
        speeds.append(speed)

        # Nonplanarity of the run window
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)
        r = pca.explained_variance_ratio_.T
        nonp.append(r[2] / np.sqrt(r[1] * r[0]))

        # Average planarity of the postures
        if X_full is not None:
            nonp_postures_i = _calculate_posture_nonplanarity(X_full[start_idx:end_idx])
            nonp_postures.append(nonp_postures_i.mean())
            nonp_postures_max.append(nonp_postures_i.max())

    distances = np.array(distances)
    speeds = np.array(speeds)
    nonp = np.array(nonp)
    nonp_postures = np.array(nonp_postures)

    return {
        'start_idxs': start_idxs,
        'end_idxs': end_idxs,
        'distances': distances,
        'speeds': speeds,
        'nonp': nonp,
        'nonp_postures': nonp_postures,
        'nonp_postures_max': nonp_postures_max,
    }


def calculate_windowed_statistics(
        args: Namespace,
        window_sizes: List[int],
        smooth_postures: bool = False
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Calculate statistics for a trajectory window.
    """
    assert args.trajectory_point is not None, '--trajectory-point needs to be defined!'
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'Calculating trajectory window statistics for trial={trial.id}.')

    # Get the full postures for signed speed calculation and posture planarities
    u = args.trajectory_point
    args.trajectory_point = None
    X = get_trajectory_from_args(args)
    speeds_all = calculate_speeds(X, signed=True)

    # Posture planarities
    logger.info('Fetching posture planarities.')
    args.planarity_window = 1
    sw = args.smoothing_window
    if not smooth_postures:
        args.smoothing_window = None
    pcas = get_pca_cache_from_args(args)
    args.smoothing_window = sw
    nonp_postures_all = pcas.nonp

    speeds = {}
    nonp_trajectories = {}
    nonp_postures_mean = {}
    nonp_postures_max = {}

    logger.info('Fetching trajectory planarities.')
    args.trajectory_point = u
    for ws in window_sizes:
        args.planarity_window = int(ws)
        pcas = get_pca_cache_from_args(args)
        speeds_window = sliding_window_view(speeds_all, ws, axis=0)
        nonp_postures_window = sliding_window_view(nonp_postures_all, ws, axis=0)

        speeds[ws] = speeds_window.mean(axis=-1)
        nonp_trajectories[ws] = pcas.nonp
        nonp_postures_mean[ws] = nonp_postures_window.mean(axis=1)
        nonp_postures_max[ws] = nonp_postures_window.max(axis=1)

    return {
        'speeds': speeds,
        'nonp_trajectories': nonp_trajectories,
        'nonp_postures_mean': nonp_postures_mean,
        'nonp_postures_max': nonp_postures_max,
    }


def calculate_windowed_helicity(
        args: Namespace,
        window_sizes: List[int],
        smooth_postures: bool = False
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Calculate helicity statistics for a trajectory window.
    """
    assert args.trajectory_point is not None, '--trajectory-point needs to be defined!'
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'Calculating trajectory window statistics for trial={trial.id}.')

    # Get the full postures for signed speed calculation and posture helicities
    u = args.trajectory_point
    args.trajectory_point = None
    X = get_trajectory_from_args(args)
    speeds_all = calculate_speeds(X, signed=True)

    # Posture planarities
    logger.info('Calculating posture helicities.')
    if not smooth_postures and args.smoothing_window is not None:
        args.smoothing_window = None
        X = get_trajectory_from_args(args)
    Hp = calculate_helicities(X)

    speeds = {}
    Ht = {}
    Hp_mean = {}
    Hp_max = {}

    logger.info('Calculating trajectory helicities.')
    args.trajectory_point = u
    X = get_trajectory_from_args(args)
    for ws in window_sizes:
        Ht_window = calculate_trajectory_helicities(X, ws, match_length=False)
        speeds_window = sliding_window_view(speeds_all, ws, axis=0)
        Hp_window = sliding_window_view(Hp, ws, axis=0)

        speeds[ws] = speeds_window.mean(axis=-1)
        Ht[ws] = Ht_window
        Hp_mean[ws] = Hp_window.mean(axis=1)
        Hp_max[ws] = Hp_window.max(axis=1)

    return {
        'speeds': speeds,
        'Ht': Ht,
        'Hp_mean': Hp_mean,
        'Hp_max': Hp_max,
    }
