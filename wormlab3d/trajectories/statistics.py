from argparse import Namespace
from typing import Dict

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from wormlab3d import logger
from wormlab3d.data.model import Trial
from wormlab3d.particles.tumble_run import calculate_curvature, get_approximate
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.util import orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_pca_cache_from_args


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
        smooth_K: int,
        window_size: int,
        curvature_height: int
) -> Dict[str, np.ndarray]:
    """
    Calculate the angles and distances for each turn in a trial trajectory.
    """
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'Calculating trial turn statistics for trial={trial.id}.')
    X = get_trajectory_from_args(args)
    pcas = get_pca_cache_from_args(args)
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

    # Take centre of mass
    X_full = None
    if X.ndim == 3:
        if X.shape[1] != 1:
            X_full = X - X.mean(axis=0)
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Calculate the approximation, tumbles and runs
    k = calculate_curvature(e0, smooth_e0=smooth_K, smooth_K=smooth_K)
    approx = get_approximate(X, k, distance=window_size, height=curvature_height)
    X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2 = approx

    # Set up outputs
    start_idxs = []
    end_idxs = []
    distances = np.zeros(len(tumble_idxs))
    pcas = []
    thetas = np.zeros(len(tumble_idxs))
    phis = np.zeros(len(tumble_idxs))
    psis = np.zeros(len(tumble_idxs))
    nonp = np.zeros(len(tumble_idxs))
    etas = np.zeros(len(tumble_idxs))
    v_in = np.zeros((len(tumble_idxs), 3))
    v_out = np.zeros((len(tumble_idxs), 3))
    v_in_theta = np.zeros((len(tumble_idxs), 3))
    v_out_theta = np.zeros((len(tumble_idxs), 3))
    v_in_phi = np.zeros((len(tumble_idxs), 3))
    v_out_phi = np.zeros((len(tumble_idxs), 3))
    v_in_psi = np.zeros((len(tumble_idxs), 3))
    v_out_psi = np.zeros((len(tumble_idxs), 3))
    nonp_postures = np.zeros(len(tumble_idxs))
    nonp_postures_max = np.zeros(len(tumble_idxs))

    # Recalculate the angles using the full trajectory and a fixed PCA window
    for i, tumble_idx in enumerate(tumble_idxs):
        # Discard first and last tumbles
        if i == 0 or i == len(tumble_idxs) - 1:
            continue

        # Get a fixed time window around the turn
        start_idx = max(0, tumble_idx - window_size)
        start_idxs.append(start_idx)
        end_idx = min(X.shape[0] - 1, tumble_idx + window_size)
        end_idxs.append(end_idx)
        X_window = X[start_idx:end_idx]

        # Calculate distance travelled across windowed trajectory
        distances[i] = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()

        # Calculate the PCA for the complete window
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)
        pcas.append(pca)

        # Nonplanarity of the manoeuvre window
        r = pca.explained_variance_ratio_.T
        nonp[i] = r[2] / np.sqrt(r[1] * r[0])

        # Calculate the PCA for the windows before and after the turn
        pca_in = PCA(svd_solver='full', copy=True, n_components=3)
        pca_in.fit(X[start_idx:tumble_idx])
        pca_out = PCA(svd_solver='full', copy=True, n_components=3)
        pca_out.fit(X[tumble_idx:end_idx])

        # Angle between incoming/outgoing planes
        etas[i] = min(
            calculate_angle(pca_in.components_[2], pca_out.components_[2]),
            calculate_angle(pca_in.components_[2], -pca_out.components_[2])
        )

        # Incoming and outgoing trajectories
        v_in_i = X[tumble_idx] - X[start_idx]
        v_out_i = X[end_idx] - X[tumble_idx]
        v_io = v_out_i - v_in_i

        # Project onto the principal plane to find planar angle.
        v_in_theta_i = orthogonalise(v_in_i, pca.components_[2])
        v_out_theta_i = orthogonalise(v_out_i, pca.components_[2])
        if np.linalg.norm(v_io) < max(np.linalg.norm(v_in_i), np.linalg.norm(v_out_i)):
            thetas[i] = min(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))
        else:
            thetas[i] = max(calculate_angle(v_in_theta_i, v_out_theta_i), calculate_angle(v_in_theta_i, -v_out_theta_i))

        # Orthogonalise against 2nd principal component to find non-planar angle.
        v_in_phi_i = orthogonalise(v_in_i, pca.components_[1])
        v_out_phi_i = orthogonalise(v_out_i, pca.components_[1])
        # phis[i] = calculate_angle(v_in_phi, v_out_phi)
        phis[i] = min(calculate_angle(v_in_phi_i, v_out_phi_i), calculate_angle(v_in_phi_i, -v_out_phi_i))

        # Orthogonalise against 1st principal component to find the final angle.
        v_in_psi_i = orthogonalise(v_in_i, pca.components_[0])
        v_out_psi_i = orthogonalise(v_out_i, pca.components_[0])
        # psis[i] = calculate_angle(v_in_psi, v_out_psi)
        psis[i] = min(calculate_angle(v_in_psi_i, v_out_psi_i), calculate_angle(v_in_psi_i, -v_out_psi_i))

        v_in[i] = v_in_i
        v_out[i] = v_out_i
        v_in_theta[i] = v_in_theta_i
        v_out_theta[i] = v_out_theta_i
        v_in_phi[i] = v_in_phi_i
        v_out_phi[i] = v_out_phi_i
        v_in_psi[i] = v_in_psi_i
        v_out_psi[i] = v_out_psi_i

        # Average planarity of the postures
        if X_full is not None:
            nonp_postures_i = _calculate_posture_nonplanarity(X_full[start_idx:end_idx])
            nonp_postures[i] = nonp_postures_i.mean()
            nonp_postures_max[i] = nonp_postures_i.max()

    return {
        'k': k,
        'vertices': vertices,
        'tumble_idxs': tumble_idxs,
        'start_idxs': start_idxs,
        'end_idxs': end_idxs,
        'distances': distances,
        'speeds': distances / window_size * trial.fps,
        'pcas': pcas,
        'thetas': thetas,
        'phis': phis,
        'psis': psis,
        'nonp': nonp,
        'etas': etas,
        'v_in': v_in,
        'v_out': v_out,
        'v_in_theta': v_in_theta,
        'v_out_theta': v_out_theta,
        'v_in_phi': v_in_phi,
        'v_out_phi': v_out_phi,
        'v_in_psi': v_in_psi,
        'v_out_psi': v_out_psi,
        'nonp_postures': nonp_postures,
        'nonp_postures_max': nonp_postures_max,
    }


def calculate_trial_run_statistics(
        args: Namespace,
        smooth_K: int,
        k_max: int,
        min_run_duration: int,
        min_run_distance: int = None,
        min_run_speed: int = None,
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
    k = calculate_curvature(e0, smooth_e0=smooth_K, smooth_K=smooth_K)
    r_state = np.r_[[0], k < k_max, [0]]

    # Find straight line sections
    run_idxs, section_props = find_peaks(r_state, width=min_run_duration, height=1)
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
        if min_run_distance is not None and distance < min_run_distance:
            continue
        if min_run_speed is not None and speed < min_run_speed:
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
