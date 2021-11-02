import numpy as np
from sklearn.decomposition import PCA
from wormlab3d import DATA_PATH

TRAJECTORY_CACHE_PATH = DATA_PATH + '/trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def smooth_trajectory(X: np.ndarray, window_len=5, window_type='flat'):
    """
    Smooth the trajectory data using a window with requested size.
    Adapted from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Extended to smooth along each of the body points and coordinate axes.
    todo: remove for loops!
    """

    assert X.ndim == 3, 'X must have 3 dimensions: [T, N, 3].'
    assert X.shape[0] > window_len, 'Time dimension needs to be bigger than window size.'
    assert window_len > 2, 'Window size must be > 2.'
    assert window_len % 2 == 1, 'Window size must be odd.'
    assert window_type in SMOOTHING_WINDOW_TYPES, f'Window type must be one of {SMOOTHING_WINDOW_TYPES}.'

    X_padded = np.r_[
        X[1:window_len // 2 + 1][::-1],
        X,
        X[-window_len // 2:-1][::-1]
    ]

    if window_type == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window_type)(window_len)

    # Normalise window
    w /= w.sum()

    # Convolve window with trajectory
    X_s = np.zeros_like(X)
    for i in range(3):
        for u in range(X.shape[1]):
            X_s[:, u, i] = np.convolve(w, X_padded[:, u, i], mode='valid')

    return X_s


def calculate_speeds(X: np.ndarray, signed: bool = False) -> np.ndarray:
    """
    Calculate the speed using the centre of mass.
    If signed speed requested, use the head-tail vector to determine direction.
    """
    assert X.shape[-1] == 3
    assert X.ndim == 3
    com = X.mean(axis=1)
    directional_gradients = np.gradient(com, axis=0)
    speeds = np.linalg.norm(directional_gradients, axis=1)

    if signed:
        ht_directions = X[:, 0] - X[:, -1]
        ht_dot_dir = np.einsum('ni,ni->n', ht_directions, directional_gradients)
        fwd_or_back = np.sign(ht_dot_dir)
        speeds = speeds * fwd_or_back

    return speeds


def calculate_htd(X: np.ndarray) -> np.ndarray:
    """
    Calculate the distances from the head to the tail.
    """
    htd = np.linalg.norm(X[:, 0] - X[:, -1], axis=1)
    return htd


def calculate_planarity(X: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the planarity as the magnitude of the 3rd PCA component in a sliding window.
    """
    X_padded = np.r_[
        np.ones((int(np.floor(window_size / 2)), *X.shape[1:])) * X[0],
        X,
        np.ones((int(np.ceil(window_size / 2)), *X.shape[1:])) * X[-1],
    ]

    planarities = np.zeros(len(X))
    for i in range(len(X) - window_size):
        pca = PCA(svd_solver='full', copy=False, n_components=3)
        shapes = X_padded[i:i + window_size].reshape((window_size * X.shape[1], 3))
        pca.fit(shapes)
        planarities[i] = np.abs(pca.singular_values_[2])
        # planarities[i] = 1 - pca.explained_variance_ratio_[2]
        # planarities[i] = pca.singular_values_[2]/pca.singular_values_.sum()

    return planarities


def prune_slowest_frames(
        X: np.ndarray,
        cut_ratio: float = 0.1,
) -> np.ndarray:
    """
    Remove slowest speed frames.
    """
    assert X.ndim == 3, 'Pruning slowest frames requires the full trajectory.'
    assert 0 < cut_ratio < 1, 'cut_ratio must be between 0 and 1.'
    speeds = calculate_speeds(X)
    com = X.mean(axis=1)
    N = len(X)
    n_frames_to_cut = int(N * cut_ratio)
    N_pruned = N - n_frames_to_cut
    frame_idxs_to_cut = np.argsort(speeds)[:n_frames_to_cut]
    frame_idxs_to_cut.sort()

    X_pruned = np.zeros((N_pruned, *X.shape[1:]))
    speeds_pruned = np.zeros(N_pruned)
    j = 0
    com_adj = np.zeros(3)

    for i in range(N):
        if len(frame_idxs_to_cut) > 0 and i == frame_idxs_to_cut[0]:
            frame_idxs_to_cut = frame_idxs_to_cut[1:]
            if (len(frame_idxs_to_cut) == 0 or i + 1 != frame_idxs_to_cut[0]) and i < N - 1:
                com_adj = com[i + 1] - com[i]
        else:
            X_pruned[j] = X[i] - com_adj
            speeds_pruned[j] = speeds[i]
            j += 1

    return X_pruned
