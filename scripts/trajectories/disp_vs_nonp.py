import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import gaussian_kde

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, logger
from wormlab3d.data.model import Dataset
from wormlab3d.particles.util import centre_select
from wormlab3d.toolkit.util import print_args, str2bool
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.displacement import DISPLACEMENT_AGGREGATION_OPTIONS, DISPLACEMENT_AGGREGATION_SQUARED_SUM, \
    calculate_displacements
from wormlab3d.trajectories.pca import generate_or_load_pca_cache, PCACache
from wormlab3d.trajectories.util import calculate_speeds

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

VAR_KEYS = ['disp', 'nonp', 'speed', 'vol']

# SURFACE_KEYS = ['d_vs_np', 'd_vs_s', 's_vs_np', 'd_vs_np', 'd_vs_s', 's_vs_np']

show_plots = True
save_plots = False
# show_plots = False
# save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot trade off between displacement and non-planarity.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')
    parser.add_argument('--min-avg-speed', type=float, default=0,
                        help='Minimum average speed in each window.')

    parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                        help='Time lag sizes.')
    parser.add_argument('--min-delta', type=int, default=1, help='Minimum time lag.')
    parser.add_argument('--max-delta', type=int, default=10000, help='Maximum time lag.')
    parser.add_argument('--delta-step', type=float, default=1, help='Step between deltas. -ve=exponential steps.')
    parser.add_argument('--aggregation', type=str, choices=DISPLACEMENT_AGGREGATION_OPTIONS,
                        default=DISPLACEMENT_AGGREGATION_SQUARED_SUM,
                        help='Displacements can be taken as L2 norms or as the squared sum of components.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    parser.add_argument('--highlights',
                        type=lambda s: [(int(item.split('-')[0]), float(item.split('-')[1])) for item in s.split(',')],
                        help='Highlight specific examples like --highlights=trial_id-onset,trial_id-onset.')

    parser.add_argument('--mesh-points', type=int, default=100, help='Number of mesh points in each direction.')

    # Stats
    parser.add_argument('--plot-nts', type=str2bool, default=True, help='Plot the number of turns.')
    parser.add_argument('--plot-vols', type=str2bool, default=True, help='Plot the volume explored.')
    parser.add_argument('--plot-nonps', type=str2bool, default=True, help='Plot the non-planarity.')
    parser.add_argument('--plot-ts', type=str2bool, default=True, help='Plot the onset times.')

    # 3D plots
    parser.add_argument('--width-3d', type=int, default=1000, help='Width of 3D plot (in pixels).')
    parser.add_argument('--height-3d', type=int, default=1000, help='Height of 3D plot (in pixels).')
    parser.add_argument('--distance', type=float, default=4., help='Camera distance (in worm lengths).')
    parser.add_argument('--azimuth', type=lambda s: [int(item) for item in s.split(',')], default=[70, ],
                        help='Azimuths.')
    parser.add_argument('--elevation', type=lambda s: [int(item) for item in s.split(',')], default=[45, ],
                        help='Elevations.')
    parser.add_argument('--roll', type=lambda s: [int(item) for item in s.split(',')], default=[45, ], help='Rolls.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace, delta: Optional[int] = None) -> str:
    id_str = f'ds={args.dataset}' \
             f'_u={args.trajectory_point}' \
             f'_sw={args.smoothing_window}' \
             f'_agg={args.aggregation}'

    if delta is not None:
        id_str += f'_d={delta}'
    else:
        id_str += f'_d={",".join([str(d) for d in args.deltas])}'

    return id_str


def _calculate_distances(X: np.ndarray, delta: int) -> np.ndarray:
    """
    Calculate the trajectory distances using a sliding window.
    """
    logger.info(f'Calculating distances for delta = {delta}.')
    if X.ndim == 3:
        X = X.mean(axis=1)
    pl = np.ones((int((delta - 1) / 2), 3)) * X[0]
    pr = np.ones((delta - len(pl) - 1, 3)) * X[-1]
    X_padded = np.r_[pl, X, pr]
    X_sections = sliding_window_view(X_padded, delta, axis=0)
    sl = np.linalg.norm(X_sections[..., 1:] - X_sections[..., :-1], axis=1)
    dists = sl.sum(axis=-1)

    return dists


def _calculate_volumes(X: np.ndarray, delta: int, pcas: PCACache) -> np.ndarray:
    """
    Calculate the volume of PCA-aligned cuboid extents using a sliding window.
    """
    logger.info(f'Calculating volumes for delta = {delta}.')
    if X.ndim == 3:
        X = X.mean(axis=1)
    X_sections = sliding_window_view(X, delta, axis=0)
    Xc = centre_select(X_sections, len(pcas))
    Xt = np.einsum('nik,nji->njk', Xc, pcas.components)
    extents = np.ptp(Xt, axis=-1)
    vols = np.product(extents, axis=-1)

    return vols


def _calculate_data(
        args: Namespace,
        ds: Dataset,
        delta: int
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    ids = {}
    displacements = {}
    nonps = {}
    speeds = {}
    distances = {}
    volumes = {}

    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')
        rec_id = ds.get_reconstruction_id_for_trial(trial)
        c = trial.experiment.concentration
        if c not in ids:
            ids[c] = []
            displacements[c] = []
            nonps[c] = []
            speeds[c] = []
            distances[c] = []
            volumes[c] = []

        # Use tracking if no reconstruction available
        if rec_id is None:
            X, _ = get_trajectory(
                trial_id=trial.id,
                smoothing_window=args.smoothing_window,
                tracking_only=True
            )
            X = X - X.mean(axis=(0, 1), keepdims=True)

            # Calculate non-planarities
            pcas, meta = generate_or_load_pca_cache(
                trial_id=trial.id,
                smoothing_window=args.smoothing_window,
                window_size=delta,
                tracking_only=True,
            )

        # Use reconstruction if available
        else:
            X, _ = get_trajectory(
                reconstruction_id=rec_id,
                smoothing_window=args.smoothing_window,
                trajectory_point=args.trajectory_point
            )
            X = X - X.mean(axis=(0, 1), keepdims=True)

            # Calculate non-planarities
            pcas, meta = generate_or_load_pca_cache(
                reconstruction_id=rec_id,
                smoothing_window=args.smoothing_window,
                trajectory_point=args.trajectory_point,
                window_size=delta,
            )

        # Get displacements, non-planarity, speeds and distances
        d = calculate_displacements(X, delta, aggregation=args.aggregation)
        nonp = pcas.nonp
        sp = calculate_speeds(X)
        dists = _calculate_distances(X, delta)
        vols = _calculate_volumes(X, delta, pcas)

        # Make sure the sizes match
        len_min = min(len(d), len(nonp), len(sp), len(dists))
        d = centre_select(d, len_min)
        nonp = centre_select(nonp, len_min)
        sp = centre_select(sp, len_min)
        dists = centre_select(dists, len_min)
        vols = centre_select(vols, len_min)
        assert nonp.shape == d.shape == sp.shape == dists.shape == vols.shape
        displacements[c].append(d)
        nonps[c].append(nonp)
        speeds[c].append(sp)
        distances[c].append(dists)
        volumes[c].append(vols)

        # Add ids
        ids[c].append(np.ones_like(d) * trial.id)

    # Sort by concentration
    ids = {k: np.concatenate(v) for k, v in sorted(list(ids.items()))}
    displacements = {k: np.concatenate(v) for k, v in sorted(list(displacements.items()))}
    nonps = {k: np.concatenate(v) for k, v in sorted(list(nonps.items()))}
    speeds = {k: np.concatenate(v) for k, v in sorted(list(speeds.items()))}
    distances = {k: np.concatenate(v) for k, v in sorted(list(distances.items()))}
    volumes = {k: np.concatenate(v) for k, v in sorted(list(volumes.items()))}

    return ids, displacements, nonps, speeds, distances, volumes


def _generate_or_load_data_for_delta(
        args: Namespace,
        ds: Dataset,
        delta: int,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Generate or load the data.
    """
    cache_path = DATA_CACHE_PATH / _identifiers(args, delta=delta)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    ids = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            ids = {}
            displacements = {}
            nonps = {}
            speeds = {}
            distances = {}
            vols = {}
            for name in data.files:
                c, k = name.split('_')
                if k != 'ids':
                    continue
                c = float(c)
                ids[c] = data[name]
                displacements[c] = data[f'{c}_displacements']
                nonps[c] = data[f'{c}_nonps']
                speeds[c] = data[f'{c}_speeds']
                distances[c] = data[f'{c}_distances']
                vols[c] = data[f'{c}_vols']
            ids = {k: v for k, v in sorted(list(ids.items()))}
            displacements = {k: v for k, v in sorted(list(displacements.items()))}
            nonps = {k: v for k, v in sorted(list(nonps.items()))}
            speeds = {k: v for k, v in sorted(list(speeds.items()))}
            distances = {k: v for k, v in sorted(list(distances.items()))}
            vols = {k: v for k, v in sorted(list(vols.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            ids = None
            logger.warning(f'Could not load cache: {e}')

    if ids is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        ids, displacements, nonps, speeds, distances, vols = _calculate_data(args, ds, delta)
        data = {}
        for c, d_vals in displacements.items():
            data[f'{c}_ids'] = ids[c]
            data[f'{c}_displacements'] = d_vals
            data[f'{c}_nonps'] = nonps[c]
            data[f'{c}_speeds'] = speeds[c]
            data[f'{c}_distances'] = distances[c]
            data[f'{c}_vols'] = vols[c]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ids, displacements, nonps, speeds, distances, vols


def _calculate_surface(
        args: Namespace,
        x: np.ndarray,
        y: np.ndarray
) -> np.ndarray:
    """
    Calculate the surface.
    """
    X, Y = np.mgrid[0:x.max():complex(args.mesh_points), 0:y.max():complex(args.mesh_points)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)  # , bw_method=0.2)
    Z = np.reshape(kernel(positions).T, X.shape)
    return Z


def _generate_or_load_surfaces_for_delta(
        args: Namespace,
        delta: int,
        displacements: np.ndarray,
        nonps: np.ndarray,
        speeds: np.ndarray,
        distances: np.ndarray,
        vols: np.ndarray,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the heatmap surface.
    """
    id_str = _identifiers(args, delta=delta) + f'_surf={args.mesh_points}'

    # Prune any too-slow windows
    if args.min_avg_speed > 0:
        id_str += f'_ms={args.min_avg_speed}'
        avg_speeds = distances / delta * 25
        idxs_to_keep = avg_speeds > args.min_avg_speed
        displacements = displacements[idxs_to_keep]
        nonps = nonps[idxs_to_keep]
        speeds = speeds[idxs_to_keep]
        vols = vols[idxs_to_keep]

    cache_path = DATA_CACHE_PATH / id_str
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    surfaces = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            surfaces = {}
            for i, k1 in enumerate(VAR_KEYS):
                for j, k2 in enumerate(VAR_KEYS):
                    if j <= i:
                        continue
                    k = f'{k1}_vs_{k2}'
                    surfaces[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            surfaces = None
            logger.warning(f'Could not load cache: {e}')

    if surfaces is None:
        if cache_only:
            raise RuntimeError(f'Surface cache "{cache_fn}" could not be loaded!')
        surfaces = {}
        # VAR_KEYS = ['disp', 'nonp', 'speed', 'vol']
        for i, k1 in enumerate(VAR_KEYS):
            x = [displacements, nonps, speeds, vols][i]
            for j, k2 in enumerate(VAR_KEYS):
                if j <= i:
                    continue
                y = [displacements, nonps, speeds, vols][j]
                k = f'{k1}_vs_{k2}'
                logger.info(f'Calculating {k} surface for delta={delta}.')
                surfaces[k] = _calculate_surface(args, x, y)
        logger.info(f'Saving surface data to {cache_fn}.')
        np.savez(cache_path, **surfaces)

    return surfaces


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    Dict[float, Dict[str, np.ndarray]],
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    # deltas, delta_ts = get_deltas_from_args(args)
    deltas = args.deltas

    ids = {}
    displacements = {}
    nonps = {}
    speeds = {}
    distances = {}
    vols = {}
    surfaces = {}
    for delta in deltas:
        ids_d, displacements_d, nonps_d, speeds_d, dists_d, vols_d = _generate_or_load_data_for_delta(
            args=args,
            ds=ds,
            delta=delta,
            rebuild_cache=rebuild_cache,
            cache_only=cache_only
        )
        ids[delta] = ids_d
        displacements[delta] = displacements_d
        nonps[delta] = nonps_d
        speeds[delta] = speeds_d
        distances[delta] = dists_d
        vols[delta] = vols_d

        d_vals = np.concatenate([d for d in displacements_d.values()])
        np_vals = np.concatenate([ad for ad in nonps_d.values()])
        s_vals = np.concatenate([s for s in speeds_d.values()])
        dist_vals = np.concatenate([d for d in dists_d.values()])
        vols_vals = np.concatenate([d for d in vols_d.values()])

        surfaces[delta] = _generate_or_load_surfaces_for_delta(
            args=args,
            delta=delta,
            displacements=d_vals,
            nonps=np_vals,
            speeds=s_vals,
            distances=dist_vals,
            vols=vols_vals
        )

    return ds, ids, displacements, nonps, speeds, vols, surfaces


def plot_correlations():
    """
    Plot the displacements against the non-planarities.
    """
    args = parse_args()
    ds, ids, displacements, nonps, speeds, vols, surfaces = _generate_or_load_data(args, rebuild_cache=True,
                                                                                   cache_only=False)

    # Axis lines
    ax_args = {
        'color': 'grey',
        'linestyle': '--',
        'linewidth': 0.5,
    }

    # pads
    style_vars = {
        'paper': {
            'tr_padx': 2,
            'tr_pady': 1,
            'heat_padx': 4,
            'heat_pady': 2,
            'heat_lw': 1,
        },
        'thesis': {
            'tr_padx': 2,
            'tr_pady': 4,
            'heat_padx': 4,
            'heat_pady': 3,
            'heat_lw': 1.2,
        }
    }['thesis']

    # VAR_KEYS = ['disp', 'nonp', 'speed', 'vol']
    data = {
        'disp': displacements,
        'nonp': nonps,
        'speed': speeds,
        'vol': vols,
    }
    labels = {
        'disp': 'Displacement',
        'nonp': 'Non-planarity',
        'speed': 'Speed',
        'vol': 'Volume',
    }

    def plot_heatmap(ax_, delta_, k1_, k2_):
        # Z = np.log(1+surfaces[delta_])
        Z = surfaces[delta_][f'{k1_}_vs_{k2_}']
        Z = np.log(1 + Z)
        x = np.concatenate([d for d in data[k1_][delta].values()])
        y = np.concatenate([d for d in data[k2_][delta].values()])
        ax_.set_xlabel(labels[k1_])
        ax_.set_ylabel(labels[k2_])

        # Add the heatmap
        ax_.axvline(x=0, **ax_args)
        ax_.axhline(y=0, **ax_args)
        ax_.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[0, x.max(), 0, y.max()], aspect='auto')
        # ax_.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
        # ax_.set_xscale('log')
        # ax_.set_yscale('log')

    # for c in concs:
    for delta in args.deltas:
        # VAR_KEYS = ['disp', 'nonp', 'speed', 'vol']
        for i, k1 in enumerate(VAR_KEYS):
            for j, k2 in enumerate(VAR_KEYS):
                if j <= i:
                    continue
                fig, axes = plt.subplots(1, figsize=(10, 10))
                plot_heatmap(axes, delta, k1, k2)
                axes.set_title(f'delta={delta}, {k1}_vs_{k2}')
        # ax = axes
        # ax.scatter(d_vals, np_vals, s=2, marker='o', alpha=0.3)

        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if show_plots:
        interactive()
    else:
        mlab.options.offscreen = True

    plot_correlations()
