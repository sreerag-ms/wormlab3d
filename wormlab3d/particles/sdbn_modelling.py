import itertools
from typing import List, Dict

import numpy as np
import scipy.stats
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import rv_continuous

from wormlab3d import logger
from wormlab3d.particles.sdbn_explorer import PARTICLE_PARAMETER_KEYS
from wormlab3d.toolkit.util import orthogonalise, normalise
from wormlab3d.trajectories.displacement import calculate_displacements, DISPLACEMENT_AGGREGATION_L2
from wormlab3d.trajectories.pca import PCACache, calculate_pcas
from wormlab3d.trajectories.util import calculate_speeds


def _calculate_transition_rates(states_list_: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the transition rates from a list of telegraph signals.
    """
    M = np.zeros((2, 2))
    for states_ in states_list_:
        states_ = states_.astype(np.int32)
        for (from_state, to_state) in zip(states_[:-1], states_[1:]):
            M[from_state, to_state] += 1
    M /= M.sum(axis=1, keepdims=True)
    return np.array([M[0, 1], M[1, 0]])


def _centre_select(X: np.ndarray, T: int) -> np.ndarray:
    """
    Take the central T portion of an array X.
    """
    assert len(X) >= T
    Xc = X[int(np.ceil((len(X) - T) / 2)):-int(np.floor((len(X) - T) / 2))]
    assert len(Xc) == T
    return Xc


def calculate_pe_parameters_for_trajectory(
        X: np.ndarray,
        deltas: List[int],
        pcas: PCACache = None,
        pca_window: int = 25,
        displacement_aggregation: str = DISPLACEMENT_AGGREGATION_L2,
        distributions: Dict[str, str] = None,
        return_additional: bool = False,
):
    """
    Calculate the particle model parameters from a trajectory.
    """
    deltas = sorted(deltas, reverse=True)
    X_com = X.mean(axis=1)
    X_com_centred = X_com - X_com.mean(axis=0, keepdims=True)
    T = len(X) - max(deltas)

    # Calculate speeds
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=False)
    speeds = _centre_select(speeds, T)

    # If the pcas are not provided then calculate them
    if pcas is None:
        pcas = calculate_pcas(X, window_size=pca_window)
        pcas = PCACache(pcas)

    # Calculate e0 (the tangent to the trajectory curve)
    logger.info('Calculating frame components.')
    e0 = normalise(np.gradient(X_com_centred, axis=0))
    e0 = _centre_select(e0, T)

    # e1 is the frame vector pointing out into the principal plane of the curve
    v1 = pcas.components[:, 1].copy()

    # Correct sign flips
    v1dotv1p = np.einsum('bi,bi->b', v1[:-1], v1[1:])
    flip_idxs = (v1dotv1p < 0).nonzero()[0]
    sign = 1
    for i in range(len(v1)):
        if i - 1 in flip_idxs:
            sign = -sign
        v1[i] = sign * v1[i]
    v1 = _centre_select(v1, T)

    # Orthogonalise the pca planar direction vector against the trajectory to get e1
    e1 = normalise(orthogonalise(v1, e0))

    # e2 is the remaining cross product
    e2 = normalise(np.cross(e0, e1))

    # Calculate the rotation angles
    logger.info('Calculating rotation angles.')
    planar_angles = np.zeros(T)
    nonplanar_angles = np.zeros(T)
    for t in range(T - 1):
        prev_frame = np.stack([e0[t], e1[t], e2[t]])
        next_frame = np.stack([e0[t + 1], e1[t + 1], e2[t + 1]])
        R, rmsd = Rotation.align_vectors(next_frame, prev_frame)
        R = R.as_matrix()

        # Decompose rotation matrix R into the axes of A
        A = prev_frame
        rp = Rotation.from_matrix(A.T @ R @ A)
        a2, a1, a0 = rp.as_euler('zyx')
        # xp = A @ Rotation.from_rotvec(a0 * np.array([1, 0, 0])).as_matrix() @ A.T
        # yp = A @ Rotation.from_rotvec(a1 * np.array([0, 1, 0])).as_matrix() @ A.T
        # zp = A @ Rotation.from_rotvec(a2 * np.array([0, 0, 1])).as_matrix() @ A.T
        # assert np.allclose(xp @ yp @ zp, R)
        planar_angles[t] = a1
        nonplanar_angles[t] = a2

    # Calculate the aligned displacements
    displacements = calculate_displacements(X, deltas, displacement_aggregation)
    max_T = len(displacements[min(deltas)])
    D = []
    for delta in deltas:
        offset = int((delta - min(deltas)) / 2)
        if delta == min(deltas):
            d = displacements[delta]
        else:
            d = np.zeros(max_T)
            d[offset:-offset - 1] = displacements[delta]
        D.append(d)
    D = np.array(D).T
    D = _centre_select(D, T)

    # Convert to states
    states = D > D.mean(axis=0, keepdims=True)

    # Calculate the transition rates
    logger.info('Calculating transition rates.')
    transition_rates = []
    for i in range(len(deltas)):
        if i == 0:
            transition_rates.append(_calculate_transition_rates([states[:, 0], ]))
        else:
            rates_i = np.zeros((2,) * (i + 1))

            # Condition on previous deltas
            prev_state_possibilities = list(itertools.product([0, 1], repeat=i))
            for prev_state in prev_state_possibilities:
                cond_state = np.ones(T, dtype=np.bool)
                for j, prev_state_value in enumerate(prev_state):
                    cond_state = np.logical_and(cond_state, states[:, j] == prev_state_value)

                # Add 0's on to ends otherwise peaks are ignored
                cond_state = np.r_[[False, ], cond_state, [False, ]]
                centre_idxs, section_props = find_peaks(cond_state, width=1)

                # Find transitions for all sections
                states_list = []
                for j in range(len(centre_idxs)):
                    start = section_props['left_bases'][j]
                    end = section_props['right_bases'][j] - 1
                    w = section_props['widths'][j]
                    if w < 2:
                        continue
                    states_list.append(states[start:end, i])
                rates_i[prev_state] = _calculate_transition_rates(states_list)
            transition_rates.append(rates_i)

    # Collate the values in each state
    state_values = {}
    state_keys = [''.join(map(str, c)) for c in list(itertools.product([0, 1], repeat=len(deltas)))]
    for state_key in state_keys:
        state_values[state_key] = {
            'speeds': [],
            'planar_angles': [],
            'nonplanar_angles': []
        }
    for t in range(T):
        state_key = ''.join(map(str, states[t].astype(np.uint8).tolist()))
        state_values[state_key]['speeds'].append(speeds[t])
        state_values[state_key]['planar_angles'].append(planar_angles[t])
        state_values[state_key]['nonplanar_angles'].append(nonplanar_angles[t])

    # Calculate the distribution values
    if distributions is None:
        distributions = {
            'speeds': 'lognorm',
            'planar_angles': 'cauchy',
            'nonplanar_angles': 'cauchy',
        }
    state_parameters = {}
    for state_key in state_keys:
        state_parameters[state_key] = {}
        for param_key in PARTICLE_PARAMETER_KEYS:
            dist_type = distributions[param_key]
            dist_cls: rv_continuous = getattr(scipy.stats, dist_type)
            dist_params = dist_cls.fit(state_values[state_key][param_key], floc=0)

            if dist_type == 'norm':
                mu, sigma = dist_params[0], dist_params[1]
            elif dist_type == 'lognorm':
                mu, sigma = np.log(dist_params[2]), dist_params[0]
            elif dist_type == 'cauchy':
                mu, sigma = dist_params[0], dist_params[1]
            else:
                raise RuntimeError(f'Distribution type {dist_type} not supported!')

            state_parameters[state_key][param_key] = {
                'dist': dist_type,
                'mu': float(mu),
                'sigma': float(sigma),
                'params': dist_params
            }

    if return_additional:
        additional = {
            'frame': {
                'X': _centre_select(X_com_centred, T),
                'e0': e0,
                'e1': e1,
                'e2': e2
            },
            'displacements': D,
            'speeds': speeds,
            'planar_angles': planar_angles,
            'nonplanar_angles': nonplanar_angles,
            'states': states,
            'state_values': state_values,
        }

        return transition_rates, state_parameters, additional

    return transition_rates, state_parameters
