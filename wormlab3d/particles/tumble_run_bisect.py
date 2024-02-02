from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from wormlab3d import N_WORKERS, logger
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import normalise, orthogonalise
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.pca import PCACache, calculate_pcas


def make_plot(
        X: np.ndarray,
        K: np.ndarray = None,
        X_approx: np.ndarray = None,
        tumble_idxs: List[int] = None,
        show_plot: bool = True,
        azim: int = -60,
        elev: int = 30,
):
    """
    Debugging function to help visualise the trajectory approximation.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    colours = np.linspace(0, 1, len(X))

    # Plot the trajectories
    if K is None:
        cmap = 'viridis_r'
    else:
        cmap = 'OrRd'
        colours = K
    ax.scatter(*X.T, c=colours, cmap=cmap, s=10, alpha=0.3, zorder=-1)
    if X_approx is not None:
        colours = np.linspace(0, 1, len(X))
        ax.scatter(*X_approx.T, c=colours, cmap='plasma_r', s=10, alpha=0.4)

    # Scatter the tumble idxs
    if tumble_idxs is not None:
        tumble_idxs = np.array([0, ] + tumble_idxs + [len(X) - 1])
        ax.scatter(*X[tumble_idxs].T, marker='x', color='green', s=200, linewidths=5, zorder=10)

    # Setup axis
    equal_aspect_ratio(ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=azim, elev=elev)

    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        return fig


def build_approx(X_section, tumble_idxs):
    # Build the approximation
    X_approx = np.zeros_like(X_section)
    X_approx[0] = X_section[0]
    x = X_section[0]
    start_idx = 0
    for i in range(len(tumble_idxs) + 1):
        end_idx = tumble_idxs[i] if i < len(tumble_idxs) else len(X_section)
        run_start = x[None, :]
        run_end = X_section[end_idx] if i < len(tumble_idxs) else X_section[-1]
        run_steps = end_idx - start_idx
        y = np.linspace(0, 1, run_steps + 1)[:-1, None]
        X_approx[start_idx:end_idx] = (1 - y) * run_start + y * run_end
        x = run_end
        start_idx = end_idx

    return X_approx


def calculate_approximation_error(args):
    X, tumble_idxs = args
    X_approx = build_approx(X, tumble_idxs)
    return np.mean(np.sum((X - X_approx)**2, axis=-1))


def find_best_bisection_idx(X: np.ndarray, check_idxs: List[int]):
    if len(X) < 3 or len(check_idxs) == 0:
        return -1

    if current_process().name == 'MainProcess':
        with Pool(processes=N_WORKERS) as pool:
            errs = pool.map(
                calculate_approximation_error,
                [(X, [i, ]) for i in check_idxs]
            )
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            errs = list(executor.map(
                calculate_approximation_error,
                [(X, [i, ]) for i in check_idxs]
            ))

    errs = np.array(errs)

    # If all the bisections increase the error, then return the index of the biggest error
    error_baseline = calculate_approximation_error((X, []))
    if np.all(errs > error_baseline):
        best_idx = np.argmax(errs)
    else:
        best_idx = np.argmin(errs)
    return check_idxs[best_idx]


def find_best_vertex_to_add(X: np.ndarray, tumble_idxs: List[int], new_idxs: List[int]):
    if len(X) < 3 or len(tumble_idxs) == 0:
        return -1

    if current_process().name == 'MainProcess':
        with Pool(processes=N_WORKERS) as pool:
            errs = pool.map(
                calculate_approximation_error,
                [(X, sorted(tumble_idxs + [new_idx, ])) for new_idx in new_idxs]
            )
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            errs = list(executor.map(
                calculate_approximation_error,
                [(X, sorted(tumble_idxs + [new_idx, ])) for new_idx in new_idxs]
            ))
    errs = np.array(errs)

    # If any of the errors increase the current error, pick the worst one
    error_baseline = calculate_approximation_error((X, tumble_idxs))
    if np.any(errs > error_baseline):
        best_idx = np.argmax(errs)
    else:
        best_idx = np.argmin(errs)
    return new_idxs[best_idx]


def find_best_vertex_to_prune(X: np.ndarray, tumble_idxs: List[int]):
    if len(X) < 3 or len(tumble_idxs) == 0:
        return -1

    if current_process().name == 'MainProcess':
        with Pool(processes=N_WORKERS) as pool:
            errs = pool.map(
                calculate_approximation_error,
                [(X, tumble_idxs[:i] + tumble_idxs[i + 1:]) for i in range(len(tumble_idxs))]
            )
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            errs = list(executor.map(
                calculate_approximation_error,
                [(X, tumble_idxs[:i] + tumble_idxs[i + 1:]) for i in range(len(tumble_idxs))]
            ))
    errs = np.array(errs)
    return tumble_idxs[np.argmin(errs)]


def find_approximation_bisect(
        X: np.ndarray,
        e0: np.ndarray,
        error_limit: float,
        planarity_window_vertices: int = 3,
        min_curvature: int = 50,
        smooth_e0: int = 101,
        smooth_K: int = 101,
        use_euler_angles: bool = True,
        max_iterations: int = 10,
        quiet: bool = False,
        plot_dir: Path = None,
        plot_every_n_changes: int = 1
):
    """
    Find an approximation to the trajectory at a given error limit.
    """
    from wormlab3d.particles.tumble_run import calculate_curvature
    if not quiet:
        logger.info(f'Finding approximation at error limit={error_limit:.3f}.')

    if X.ndim == 3:
        X = X.mean(axis=1)
    T = len(X)
    iteration = 0
    azim = -60
    azim_incr = 10

    # Calculate the curvature
    K = calculate_curvature(e0, smooth_e0=smooth_e0, smooth_K=smooth_K)

    # Resample the trajectory so that it has a constant speed
    X_orig = X.copy()

    # Calculate cumulative distance along the trajectory
    distances = np.linalg.norm(X[1:] - X[:-1], axis=1)
    cumulative_distances = np.cumsum(np.r_[[0, ], distances])

    # Interpolate the spatial coordinates based on constant-speed cumulative distances
    interp_func = interp1d(cumulative_distances, X, axis=0, kind='cubic', fill_value='extrapolate')
    cs_cumulative_distances = np.linspace(0, cumulative_distances[-1], T)
    X = interp_func(cs_cumulative_distances)

    # Create the inverse interpolation function and check that the original trajectory can be recovered
    inverse_interp_func = interp1d(cs_cumulative_distances, X, axis=0, kind='cubic', fill_value='extrapolate')
    X_orig_recovered = inverse_interp_func(cumulative_distances)
    assert np.allclose(X_orig, X_orig_recovered, atol=1e-2)

    # Get the mapping of indices from the original to the resampled trajectory
    sf = np.r_[[1, ], cumulative_distances[1:] / cs_cumulative_distances[1:]]

    def map_idxs(idxs):
        return np.round(idxs * sf[idxs]).astype(int).tolist()

    mapping_idxs = map_idxs(np.arange(T))
    X_orig_recovered2 = X[mapping_idxs]
    assert np.allclose(X_orig, X_orig_recovered2, atol=1e-2)

    # Get the inverse mapping, from the resampled to the original trajectory
    inverse_idx_func = interp1d(mapping_idxs, np.arange(T), kind='nearest', fill_value='extrapolate')

    def inverse_map_idxs(idxs):
        return inverse_idx_func(idxs).astype(int).tolist()

    inverse_mapping_idxs = inverse_map_idxs(np.arange(T))
    X_by_idx_mapping = X_orig[inverse_mapping_idxs]
    assert np.allclose(X, X_by_idx_mapping,
                       atol=1e-1), f'X not recovered by idx mapping! Max error = {np.max(np.abs(X - X_by_idx_mapping))}'

    # Only allow tumbles where the curvature is above the height threshold
    # allowable_tumble_idxs = map_idxs((K > min_curvature).nonzero()[0])
    # allowable_tumble_idxs = np.unique(allowable_tumble_idxs)
    allowable_tumble_idxs = np.arange(T)

    # Cache the best bisection indices for each section
    best_idx_cache = np.zeros((T + 1, T + 1), dtype=int)

    # Add an initial tumble vertex
    v0 = find_best_bisection_idx(X, allowable_tumble_idxs)
    if not quiet:
        logger.debug(f'Added initial vertex at {v0}.')
    best_idx_cache[0, T] = v0
    tumble_idxs = [v0]
    X_approx = build_approx(X, tumble_idxs)
    X_approx_prev = np.zeros_like(X_approx)

    def approx_error(X_approx_=None):
        if X_approx_ is None:
            X_approx_ = X_approx
        return np.mean(np.sum((X - X_approx_)**2, axis=-1))

    # Keep adding and pruning vertices until the approximation doesn't change
    while (approx_error() > error_limit or not np.allclose(X_approx, X_approx_prev)) and iteration < max_iterations:
        X_approx_prev = X_approx.copy()

        # Keep adding vertices one at a time until the error is half the target or we run out of vertices to add
        new_idxs = []
        while approx_error() > error_limit / 2 or len(tumble_idxs) < 2:
            candidate_idxs = []
            vls = np.r_[[0, ], tumble_idxs].astype(int)
            vrs = np.r_[tumble_idxs, [T, ]].astype(int)

            # Find the best bisection indices for each section and cache results
            for i in range(len(tumble_idxs) + 1):
                vl = vls[i]
                vr = vrs[i]
                if best_idx_cache[vl, vr] == 0:
                    Xi = X[vl:vr]
                    check_idxs = allowable_tumble_idxs[(allowable_tumble_idxs > vl) & (allowable_tumble_idxs < vr)] - vl
                    new_idx = find_best_bisection_idx(Xi, check_idxs)
                    if new_idx > -1:
                        new_idx += vl
                    best_idx_cache[vl, vr] = new_idx
                new_idx = best_idx_cache[vl, vr]
                if new_idx > -1:
                    candidate_idxs.append(new_idx)

            # Add the vertex that improves the error the most
            if len(candidate_idxs) == 0:
                break
            add_idx = find_best_vertex_to_add(X, tumble_idxs, candidate_idxs)
            new_idxs.append(add_idx)
            tumble_idxs = sorted(tumble_idxs + [add_idx, ])
            X_approx = build_approx(X, tumble_idxs)
            if not quiet:
                logger.info(f'Added vertex at {add_idx}. Error = {approx_error():.4f}.')

            if plot_dir is not None and (len(new_idxs) % plot_every_n_changes) == 0:
                fig = make_plot(X_orig, K, X_approx, inverse_map_idxs(tumble_idxs), show_plot=False, azim=azim)
                fig.savefig(plot_dir / f'approx_{iteration:03d}.0.{len(new_idxs):04d}.png')
                plt.close(fig)
                azim += azim_incr

        if len(new_idxs) > 0:
            new_idxs = sorted(new_idxs)
            if not quiet:
                logger.debug(f'Added tumble idxs at {new_idxs}')

        # Prune vertices by removing ones that decrease the error the least until the error limit is reached
        removed_tumble_idxs = []
        while approx_error() < error_limit and len(tumble_idxs) > 2:
            prune_idx = find_best_vertex_to_prune(X, tumble_idxs)
            if prune_idx == -1:
                break
            pruned_idxs = tumble_idxs.copy()
            pruned_idxs.remove(prune_idx)
            X_approx_pruned = build_approx(X, pruned_idxs)
            if approx_error(X_approx_pruned) > error_limit:
                break
            removed_tumble_idxs.append(prune_idx)
            tumble_idxs = pruned_idxs
            X_approx = X_approx_pruned

            if plot_dir is not None and (len(pruned_idxs) % plot_every_n_changes) == 0:
                fig = make_plot(X_orig, K, X_approx, inverse_map_idxs(tumble_idxs), show_plot=False, azim=azim)
                fig.savefig(plot_dir / f'approx_{iteration:03d}.1.{len(removed_tumble_idxs):04d}.png')
                plt.close(fig)
                azim += azim_incr

        if len(removed_tumble_idxs) > 0:
            removed_tumble_idxs = sorted(removed_tumble_idxs)
            if not quiet:
                logger.debug(f'Removed tumble idxs at {removed_tumble_idxs}')

        iteration += 1
        if not quiet:
            logger.info(
                f'Iteration {iteration} of {max_iterations}. '
                f'Error = {approx_error():.3f}. '
                f'Num vertices = {len(tumble_idxs)}.'
            )
        if np.allclose(X_approx, X_approx_prev):
            if not quiet:
                logger.info('Converged!')
            if plot_dir is not None:
                fig = make_plot(X_orig, K, X_approx, inverse_map_idxs(tumble_idxs), show_plot=False, azim=azim)
                fig.savefig(plot_dir / f'approx_{iteration:03d}.2.png')

    # Map vertices back to the original trajectory
    tumble_idxs = np.array(inverse_map_idxs(tumble_idxs), dtype=int)
    tumble_idxs = np.unique(tumble_idxs)
    N = len(tumble_idxs)
    if N <= 1:
        raise RuntimeError('Too few vertices found! Try adjusting parameters.')
    X = X_orig

    # Add vertices for start and end points
    vertices = np.concatenate([
        X[0][None, :],
        X[tumble_idxs],
        X[-1][None, :],
    ], axis=0)

    # Calculate run durations
    run_durations = (np.array(tumble_idxs[1:]) - np.array(tumble_idxs[:-1])).astype(np.float64)

    # Calculate run speeds
    run_distances = np.linalg.norm(vertices[2:-1] - vertices[1:-2], axis=-1)
    run_speeds = run_distances / run_durations

    # Calculate PCA along the vertices
    pcas = calculate_pcas(vertices, window_size=min(vertices.shape[0] - 1, planarity_window_vertices), parallel=False)
    pcas = PCACache(pcas)
    components = pcas.components.copy()
    diff = N - components.shape[0] + 1

    # The first and last few tumbles don't change the plane so recalculate components using shrinking windows
    n_pre = int(np.ceil(diff / 2))
    n_post = int(np.floor(diff / 2))
    if n_pre > 0:
        prepend_components = []
        for i in range(n_pre):
            ws = planarity_window_vertices - i - 1
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            if ws > 2 and i < n_pre - 1:
                pca.fit(vertices[:ws])
            else:
                # Approximate the first components using the full trajectory
                pca.fit(X[:tumble_idxs[i]])
            prepend_components.append(pca.components_)
        prepend_components = np.stack(prepend_components[::-1])
        components = np.concatenate([prepend_components, components], axis=0)
    if n_post > 0:
        append_components = []
        for i in range(n_post):
            ws = planarity_window_vertices - i - 1
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            if ws > 2 and i < n_post - 1:
                pca.fit(vertices[-ws:])
            else:
                # Approximate the last components using the full trajectory
                pca.fit(X[tumble_idxs[-1 - i]:])
            append_components.append(pca.components_)
        append_components = np.stack(append_components)
        components = np.concatenate([components, append_components], axis=0)

    # Calculate e0 as normalised line segments between tumbles
    e0 = normalise(vertices[1:] - vertices[:-1])

    # e1 is the frame vector pointing out into the principal plane of the curve
    v1 = components[:, 1].copy()

    # Orthogonalise the pca planar direction vector against the trajectory to get e1
    e1 = normalise(orthogonalise(v1, e0))

    # e2 is the remaining cross product
    e2 = normalise(np.cross(e0, e1))

    # e2a = -e2
    # e2_tidied = np.zeros_like(e2)
    # e2_tidied[0] = e2[0]
    # for i in range(len(e0) - 1):
    #     a0 = calculate_angle(e2_tidied[i], e2[i + 1])
    #     a1 = calculate_angle(e2_tidied[i], e2a[i + 1])
    #     if a0 < a1:
    #         e2_tidied[i + 1] = e2[i + 1]
    #     else:
    #         e2_tidied[i + 1] = e2a[i + 1]
    # e2 = e2_tidied

    # Duplicate final frame to line things up
    e0 = np.r_[e0, e0[-1][None, ...]]
    e1 = np.r_[e1, e1[-1][None, ...]]
    e2 = np.r_[e2, e2[-1][None, ...]]

    # Calculate the angles
    planar_angles = np.zeros(N)
    nonplanar_angles = np.zeros(N)
    twist_angles = np.zeros(N)
    for i in range(N):
        if use_euler_angles:
            prev_frame = np.stack([e0[i], e1[i], e2[i]])
            next_frame = np.stack([e0[i + 1], e1[i + 1], e2[i + 1]])
            R, rmsd = Rotation.align_vectors(prev_frame, next_frame)
            R = R.as_matrix()

            # Decompose rotation matrix R into the axes of A
            A = prev_frame
            rp = Rotation.from_matrix(A.T @ R @ A)
            a2, a1, a0 = rp.as_euler('zyx')
            planar_angles[i] = a2  # + a0 # Rotation about e2
            nonplanar_angles[i] = a1  # Rotation about e1
            twist_angles[i] = a0  # Rotation about e0

            R2 = Rotation.from_euler('zyx', [a2, a1, a0])
            assert np.allclose(Rotation.from_matrix(R).magnitude(), R2.magnitude(), atol=0.1)

        else:
            # Project v onto the e0/e1 plane
            v = e0[i + 1]
            v_proj = v - np.dot(v, e2[i]) * e2[i]

            # Find angles
            alpha = calculate_angle(e0[i], v_proj)
            beta = calculate_angle(v_proj, v)

            # Check for flips
            if np.dot(e2[i], np.cross(e0[i], v)) < 0:
                alpha *= -1
            if np.dot(e1[i], np.cross(e0[i], v)) < 0:
                beta *= -1

            planar_angles[i] = alpha
            nonplanar_angles[i] = beta

    approx = X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles, nonplanar_angles, twist_angles, e0, e1, e2

    return approx


def test_rotation_angles():
    def generate_rotation_matrix(axis, angle):
        return Rotation.from_rotvec(axis * angle).as_matrix()

    # Example: Orthonormal frame e0/e1/e2
    e0 = np.array([1, 0, 0])
    e1 = np.array([0, 1, 0])
    e2 = np.array([0, 0, 1])

    # Example: Target vector v
    for i in range(100):
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        v_proj = v - np.dot(v, e2) * e2
        v_proj = v_proj / np.linalg.norm(v_proj)

        # Find angles
        alpha = np.arccos(np.dot(e0, v_proj))
        beta = np.arccos(np.dot(v_proj, v))

        # Check for flips
        if np.dot(e2, np.cross(e0, v)) < 0:
            alpha *= -1
        if np.dot(e1, np.cross(e0, v)) < 0:
            beta *= -1

        # Generate rotation matrices for alpha and beta
        R_alpha = generate_rotation_matrix(e2, alpha)
        v_proj_generated = R_alpha @ e0
        e1_rotated = R_alpha @ e1
        R_beta = generate_rotation_matrix(e1_rotated, beta)

        # Apply rotations to e0
        v_generated = R_beta @ v_proj_generated
        print("Original vector v:", v)
        print("Generated vector v:", v_generated)

        assert np.allclose(v, v_generated, atol=0.1)
