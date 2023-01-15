from multiprocessing import Pool

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import linregress

from wormlab3d import logger, N_WORKERS
from wormlab3d.toolkit.util import normalise


def calculate_box_dimension(
        X: np.ndarray,
        voxel_sizes: np.ndarray,
        plateau_threshold: float,
        sample_size: int = 1,
        sf_min: float = 1.,
        sf_max: float = 1.,
        parallel: bool = True
):
    """
    Calculate the fractal dimension of the trajectory using the box counting method.
    """
    if X.ndim == 3:
        X = X - X.mean(axis=(0, 1), keepdims=True)
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    else:
        X = X - X.mean(axis=0, keepdims=True)

    # Calculate rotated and scaled versions
    if sample_size > 1:
        logger.info(f'Calculating box dimension with sample size = {sample_size}.')
        scalars = np.linspace(sf_min, sf_max, sample_size)
        if parallel and N_WORKERS > 1:
            logger.info(f'Using {N_WORKERS} workers in parallel.')
            with Pool(processes=N_WORKERS) as pool:
                results = pool.map(
                    _calculate_box_dimension_wrapper,
                    [[X, voxel_sizes, plateau_threshold, scalars[i]] for i in range(sample_size)]
                )
        else:
            results = []
            for i in range(sample_size):
                R = Rotation.from_rotvec(normalise(np.random.random(3)) * np.random.rand() * np.pi).as_matrix()
                Xr = scalars[i] * (X @ R)
                res = calculate_box_dimension(Xr, voxel_sizes, plateau_threshold)
                results.append(res)

        Ds = np.array([res['D'] for res in results])
        ms = np.array([res['m'] for res in results])
        counts = np.stack([res['counts'] for res in results])
        starts = np.array([res['range_start'] for res in results])
        ends = np.array([res['range_end'] for res in results])

        return {
            'D': Ds.mean(),
            'm': ms.mean(),
            'D_std': Ds.std(),
            'm_std': ms.std(),
            'counts': counts,
            'range_start': starts.mean(),
            'range_end': ends.mean(),
            'Ds': Ds,
            'ms': ms,
        }

    # Count boxes (voxels)
    counts = np.zeros(len(voxel_sizes))
    for j, vs in enumerate(voxel_sizes):
        Xd = np.round(X / vs).astype(np.int32)
        counts[j] = np.unique(Xd, axis=0).shape[0]

    # Estimate the slope using different window sizes to find the best range
    window_sizes = np.round(np.linspace(0.1, 0.5, 5) * len(voxel_sizes)).astype(int).tolist()
    r2s = np.zeros((len(window_sizes), len(voxel_sizes)))
    ssrs = np.zeros((len(window_sizes), len(voxel_sizes)))
    for i, ws in enumerate(window_sizes):
        for j in range(len(voxel_sizes) - ws):
            x = np.log(voxel_sizes[j: j + ws])
            y = np.log(counts[j:j + ws])
            k, m, r, p, std_err = linregress(x, y)
            r2s[i, j + int(ws / 2)] = r**2
            # ssrs[i, j + int(ws / 2)] = ((y - (k * x + m))**2).sum()

    # Find the peak and plateau
    # err = (r2s - ssrs).sum(axis=0)
    err = r2s.sum(axis=0)
    good_vals = err > err[np.argmax(err)] * plateau_threshold
    midpoints, stats = find_peaks(good_vals, width=5)
    peak_idx = -1
    peak_width = -1
    for i in range(len(midpoints)):
        if stats['widths'][i] > peak_width:
            peak_idx = i
            peak_width = stats['widths'][i]
        else:
            continue
    if peak_idx == -1:
        raise RuntimeError('Could not determine peak!')

    # Use the values along the plateau range to calculate the final slope
    range_start = stats['left_bases'][peak_idx]
    range_end = stats['right_bases'][peak_idx]
    k, m = np.polyfit(
        np.log(voxel_sizes[range_start:range_end]),
        np.log(counts[range_start:range_end]),
        1
    )

    return {
        'D': -k,
        'm': m,
        'counts': counts,
        'range_start': range_start,
        'range_end': range_end,
    }


def _calculate_box_dimension_wrapper(args):
    X, voxel_sizes, plateau_threshold, scalar = args
    R = Rotation.from_rotvec(normalise(np.random.random(3)) * np.random.rand() * np.pi).as_matrix()
    Xr = scalar * (X @ R)
    return calculate_box_dimension(
        X=Xr,
        voxel_sizes=voxel_sizes,
        plateau_threshold=plateau_threshold
    )
