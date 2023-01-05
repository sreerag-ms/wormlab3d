import os
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import linregress

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP, N_WORKERS
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.particles.cache import get_voxel_sizes_from_args
from wormlab3d.toolkit.util import print_args, normalise
from wormlab3d.trajectories.cache import get_trajectory

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

DATA_KEYS = [
    'D',  # Fractal dimension
    'm',  # Slope intercepts
    'D_std',  # Fractal dimension std
    'm_std',  # Slope intercepts std
    'counts',  # Voxel counts
    'range_start',  # Plateau range start
    'range_end',  # Plateau range end
]

show_plots = True
save_plots = False
# show_plots = False
# save_plots = True
interactive_plots = False
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to find fractal dimensions.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--vxs-min', type=float, default=1e-3,
                        help='Minimum voxel size.')
    parser.add_argument('--vxs-max', type=float, default=1e1,
                        help='Maximum voxel size.')
    parser.add_argument('--vxs-num', type=int, default=100,
                        help='Number of voxel sizes.')

    parser.add_argument('--plateau-threshold', type=float, default=0.95,
                        help='Percentage of the best fit value to include when finding the plateau range.')

    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of randomisations to average over when calculating the box dimension.')
    parser.add_argument('--sf-min', type=float, default=0.9,
                        help='Minimum trajectory scaling factor.')
    parser.add_argument('--sf-max', type=float, default=1.1,
                        help='Maximum trajectory scaling factor.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace, include_dataset: bool = True, include_reconstruction: bool = False) -> str:
    if include_dataset:
        id_str = f'ds={args.dataset}'
    elif include_reconstruction:
        rec = Reconstruction.objects.get(id=args.reconstruction)
        id_str = f'trial={rec.trial.id}_rec={rec.id}'

    id_str += f'_u={args.trajectory_point}' \
              f'_sw={args.smoothing_window}' \
              f'_vxs={args.vxs_min:.1E}-{args.vxs_max:.1E}_{args.vxs_num}' \
              f'_pt={args.plateau_threshold}' \
              f'_ss={args.sample_size}' \
              f'_sf={args.sf_min}-{args.sf_max}'

    return id_str


def _calculate_box_dimension(
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
                res = _calculate_box_dimension(Xr, voxel_sizes, plateau_threshold)
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
    window_sizes = [5, 10, 15, 20]
    r2s = np.zeros((len(window_sizes), len(voxel_sizes)))
    ssrs = np.zeros((len(window_sizes), len(voxel_sizes)))
    for i, ws in enumerate(window_sizes):
        for j in range(len(voxel_sizes) - ws):
            x = np.log(voxel_sizes[j: j + ws])
            y = np.log(counts[j:j + ws])
            k, m, r, p, std_err = linregress(x, y)
            r2s[i, j + int(ws / 2)] = r**2
            ssrs[i, j + int(ws / 2)] = ((y - (k * x + m))**2).sum()

    # Find the peak and plateau
    err = (r2s - ssrs).sum(axis=0)
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
    return _calculate_box_dimension(
        X=Xr,
        voxel_sizes=voxel_sizes,
        plateau_threshold=plateau_threshold
    )


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]

    ids = {}
    results = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating data for trial={trial.id}.')
        rec_id = ds.get_reconstruction_id_for_trial(trial)

        # Use tracking if no reconstruction available
        if rec_id is None:
            if args.trajectory_point != -1:
                logger.info(f'No reconstruction available and u={args.trajectory_point}, skipping.')
                continue
            X, _ = get_trajectory(
                trial_id=trial.id,
                smoothing_window=args.smoothing_window,
                tracking_only=True
            )

        # Use reconstruction if available
        else:
            X, _ = get_trajectory(
                reconstruction_id=rec_id,
                smoothing_window=args.smoothing_window,
                trajectory_point=args.trajectory_point
            )

        # Add concentration to results
        c = trial.experiment.concentration
        if c not in ids:
            ids[c] = []
            results[c] = {k: [] for k in DATA_KEYS}

        # Calculate dimension and save results
        res = _calculate_box_dimension(
            X,
            voxel_sizes,
            args.plateau_threshold,
            sample_size=args.sample_size,
            sf_min=args.sf_min,
            sf_max=args.sf_max,
        )
        for k in DATA_KEYS:
            results[c][k].append(res[k])

        # Add ids
        ids[c].append(trial.id)

        # if trial.id > 10:
        #     break

    # Sort by concentration
    ids = {c: np.array(v) for c, v in sorted(list(ids.items()))}
    results = {c: v for c, v in sorted(list(results.items()))}
    for c in results.keys():
        for k in DATA_KEYS:
            if k == 'counts':
                results[c][k] = np.stack(results[c][k])
            else:
                results[c][k] = np.array(results[c][k])

    return ids, results


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    ids = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            ids = {}
            results = {}
            for name in data.files:
                c, fn = name.split('_')
                if fn != 'ids':
                    continue
                c = float(c)
                ids[c] = data[name]
                results[c] = {k: data[f'{c}_{k.replace("_", "-")}'] for k in DATA_KEYS}

            ids = {k: v for k, v in sorted(list(ids.items()))}
            results = {k: v for k, v in sorted(list(results.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            ids = None
            logger.warning(f'Could not load cache: {e}')

    if ids is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        ids, results = _calculate_data(args, ds)
        data = {}
        for c, id_vals in ids.items():
            data[f'{c}_ids'] = id_vals
            for k in DATA_KEYS:
                data[f'{c}_{k.replace("_", "-")}'] = results[c][k]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, ids, results


def plot_dimension_over_time():
    args = parse_args()
    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]
    X, _ = get_trajectory(
        reconstruction_id='62b0d10d3cbd609fb58cb381',
        smoothing_window=args.smoothing_window,
        trajectory_point=args.trajectory_point
    )
    if X.ndim == 3:
        X = X.mean(axis=1)

    # deltas = [125, 750, 2500]
    deltas = [2500, ]

    dimensions = np.zeros((len(deltas), len(X)))

    for i, delta in enumerate(deltas):
        Xs = sliding_window_view(X, delta, axis=0).transpose(2, 0, 1)
        logger.info(f'Calculating box dimensions for {len(Xs)} windows.')
        for j, Xd in enumerate(Xs):
            if (j + 1) % 10 == 0:
                logger.info(f'Calculating box dimension for window {j + 1}/{len(Xs)}.')
            D = _calculate_box_dimension(Xd, voxel_sizes)
            dimensions[i, j + int(delta / 2)] = D

    for i, delta in enumerate(deltas):
        plt.plot(dimensions[i])

    plt.show()


def plot_reconstruction_box_dimension(
        plot_n_samples: int = 10
):
    """
    Plot the box dimension fit for a reconstruction.
    """
    args = parse_args()
    assert args.reconstruction is not None, '--reconstruction must be defined!'
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]

    # Fetch reconstruction
    X, _ = get_trajectory(
        reconstruction_id=reconstruction.id,
        smoothing_window=args.smoothing_window,
        trajectory_point=args.trajectory_point
    )

    # Calculate box dimension
    res = _calculate_box_dimension(
        X,
        voxel_sizes,
        args.plateau_threshold,
        sample_size=args.sample_size,
        sf_min=args.sf_min,
        sf_max=args.sf_max,
    )
    D = res['D']
    D_std = res['D_std']
    m = res['m']
    counts = res['counts']
    if counts.ndim == 2:
        counts = counts.mean(axis=0)

    # Make fit line
    xs = np.exp(-np.linspace(np.log(1 / voxel_sizes.max()), np.log(1 / voxel_sizes.min()), 1000))
    ys = np.exp(m) * xs**(-D)

    # Plot
    fig, axes = plt.subplots(1, figsize=(10, 10))
    ax = axes
    ax.plot(voxel_sizes, counts, marker='+', label=f'D_avg={D:.4f}', color='green', zorder=100, linewidth=2,
            markersize=10)
    ax.plot(xs, ys, color='green', linestyle='--', alpha=0.8, zorder=101, linewidth=4)
    ax.set_xlabel('Voxel size')
    ax.set_ylabel('Voxels visited')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Trial={reconstruction.trial.id}. Rec={reconstruction.id}.\n'
                 f'[u={args.trajectory_point}, smoothing={args.smoothing_window}, '
                 f'pt={args.plateau_threshold}, samples={args.sample_size}]\n'
                 f'D={D:.4f} $\pm$ {D_std} (std)')
    ax.grid()

    # # Use 60 second chunks
    # chunks = np.array_split(X, len(X)/1500)
    # cmap = plt.get_cmap('cool')
    # colours = cmap(np.linspace(0, 1, len(chunks)))
    # for i, chunk in enumerate(chunks):
    #     res = _calculate_box_dimension(chunk, voxel_sizes, args.plateau_threshold)
    #     D = res['D']
    #     m = res['m']
    #     counts = res['counts']
    #     ys = np.exp(m) * xs**(-D)
    #     ax.plot(voxel_sizes, counts, marker='x', label=f'D_{i}={D:.4f}', color=colours[i])
    #     ax.plot(xs, ys, color=colours[i], linestyle='--', alpha=0.7)

    # Plot samples
    N = min(plot_n_samples, args.sample_size)
    idxs = np.round(np.linspace(0, 1, N) * (args.sample_size - 1)).astype(int)
    cmap = plt.get_cmap('cool')
    colours = cmap(np.linspace(0, 1, N))
    for i, idx in enumerate(idxs):
        D = res['Ds'][idx]
        m = res['ms'][idx]
        counts = res['counts'][idx]
        ys = np.exp(m) * xs**(-D)
        ax.plot(voxel_sizes, counts, marker='x', label=f'D_{idx}={D:.4f}', color=colours[i], alpha=0.5, linewidth=1)
        ax.plot(xs, ys, color=colours[i], linestyle='--', alpha=0.7, linewidth=1)

    ax.legend()

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_{_identifiers(args, include_dataset=False, include_reconstruction=True)}'
                            f'.{img_extension}')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_reconstruction_box_dimension_variability():
    """
    Plot the box dimension fit for a reconstruction.
    """
    args = parse_args()
    assert args.reconstruction is not None, '--reconstruction must be defined!'
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]

    # Fetch reconstruction
    X, _ = get_trajectory(
        reconstruction_id=reconstruction.id,
        smoothing_window=args.smoothing_window,
        trajectory_point=args.trajectory_point
    )

    # Calculate box dimension
    res = _calculate_box_dimension(
        X,
        voxel_sizes,
        args.plateau_threshold,
        sample_size=args.sample_size,
        sf_min=args.sf_min,
        sf_max=args.sf_max,
    )
    D = res['D']
    Ds = res['Ds']

    # Plot histogram
    fig, axes = plt.subplots(1, figsize=(10, 10))
    ax = axes
    ax.hist(Ds, bins=10, density=True, rwidth=0.9)
    ax.set_xlabel('Box dimension')
    ax.set_ylabel('Density')
    ax.set_title(f'Trial={reconstruction.trial.id}. Rec={reconstruction.id}.\n'
                 f'[u={args.trajectory_point}, smoothing={args.smoothing_window}, pt={args.plateau_threshold}]\n'
                 f'D={D:.4f} $\pm$ {Ds.std()} (std)')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_variability'
                            f'_{_identifiers(args, include_dataset=False, include_reconstruction=True)}'
                            f'.{img_extension}')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def fractal_dimensions_by_conc(
        layout: str = 'paper'
):
    """
    Plot the fractal dimensions of paths across the dataset, grouped by concentration.
    """
    args = parse_args()
    assert args.dataset is not None, '--dataset must be defined!'

    # us = [-1, 0.1, 0.5]
    # us = [-1, 0.5]  #, 0.1, 0.5]
    # us = np.linspace(0, 1, 5)
    us = np.r_[[-1, ], np.linspace(0, 1, 5)]
    # exclude_concs = [0.75]
    exclude_concs = []
    breaks_at = [2, 3]
    us = [float(u) for u in us]

    # Prepare output
    out_conc = {}
    out_reconst = {}
    for u in us:
        args.trajectory_point = u
        ds, ids, results = _generate_or_load_data(args, rebuild_cache=True, cache_only=False)
        out_conc[u] = {}
        out_reconst[u] = {}
        i = 0
        for c, res in results.items():
            if c in exclude_concs:
                continue
            if c not in out_reconst:
                out_conc[u][c] = []
                out_reconst[u][c] = []
            D_c = res['D']
            out_conc[u][c] = np.array([D_c.mean(axis=0), D_c.std(axis=0)]).T
            out_reconst[u][c] = D_c
            i += 1

    # Determine positions
    concs = np.unique([float(c) for u_ in us for c in out_reconst[u_].keys()])
    concs.sort()
    ticks = np.arange(len(concs))
    ticks_u = {u: ticks[[c in out_reconst[u] for c in concs]] for u in us}

    # Make plots
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, len(us)))
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colours = prop_cycle.by_key()['color']

    # plt.rc('axes', labelsize=6)  # fontsize of the X label
    # plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    # plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    # plt.rc('legend', fontsize=5)  # fontsize of the legend
    # plt.rc('xtick.major', pad=2, size=2)
    # plt.rc('ytick.major', pad=1, size=2)

    # fig, ax = plt.subplots(1, figsize=(2.15, 1.9), gridspec_kw={
    #     'left': 0.16,
    #     'right': 0.99,
    #     'top': 0.97,
    #     'bottom': 0.16,
    # })

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels(concs)
    offsets = np.linspace(-0.1, 0.1, len(us))

    ax.set_xlabel('Concentration (% gelatin)')
    ax.set_ylabel('Fractal dimension', labelpad=1)

    for i, u in enumerate(us):
        for j, c in enumerate(concs):
            if c not in out_reconst[u]:
                continue
            ij_vals = out_reconst[u][c]
            ax.scatter(
                x=np.ones_like(ij_vals) * ticks[j] + offsets[i],
                y=ij_vals,
                color=colours[i],
                marker='o',
                facecolor='none',
                linewidths=0.5,
                s=20,
                alpha=0.8,
            )

        means, stds = np.array(list(out_conc[u].values())).T
        for k, break_at in enumerate(breaks_at + [np.inf]):
            ticks_in_range = ticks[(concs <= break_at) & (concs > (0 if k == 0 else breaks_at[k - 1]))]
            idxs = np.intersect1d(ticks_u[u], ticks_in_range)
            if len(idxs) == 0:
                continue
            offset = np.where(np.setdiff1d(ticks, ticks_u[u]) < idxs[0], 1, 0).sum()
            start_idx = idxs[0] - offset
            end_idx = idxs[-1] - offset + 1

            label = f'u={u:.2f}' if k == 0 else None
            ax.errorbar(
                idxs + offsets[i],
                means[start_idx:end_idx],
                yerr=stds[start_idx:end_idx],
                elinewidth=1,
                capsize=3,
                color=colours[i],
                label=label,
                alpha=0.7,
            )

            if break_at < concs[-1]:
                if layout == 'paper':
                    dx = .1
                    dy = .02
                    dist = 0.2
                else:
                    dx = .08
                    dy = .03
                    dist = 0.16

                trans = blended_transform_factory(ax.transData, ax.transAxes)
                break_line_args = dict(transform=trans, color='k', clip_on=False, linewidth=1)
                x_div = (concs == break_at).nonzero()[0][0] + 0.5
                ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (-dy, dy), **break_line_args)
                ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (-dy, dy), **break_line_args)
                ax.plot((x_div - dist / 2 - dx, x_div - dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)
                ax.plot((x_div + dist / 2 - dx, x_div + dist / 2 + dx), (1 - dy, 1 + dy), **break_line_args)

    if layout == 'paper':
        # Remove the errorbars from the legend handles
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        ax.legend(handles, labels, loc='upper right', markerscale=0.8, handlelength=1, handletextpad=0.6,
                  labelspacing=0, borderpad=0.5, ncol=len(us), columnspacing=0.8,
                  bbox_to_anchor=(0.99, 0.98), bbox_transform=ax.transAxes)

    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.95), bbox_transform=ax.transAxes)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_{_identifiers(args, include_dataset=True, include_reconstruction=False)}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    # plot_reconstruction_box_dimension()
    # plot_reconstruction_box_dimension_variability()
    fractal_dimensions_by_conc()
    # plot_dimension_over_time()
