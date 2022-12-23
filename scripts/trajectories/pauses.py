import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from scipy.signal import find_peaks

from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import calculate_speeds
from wormlab3d.trajectories.util import smooth_trajectory

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot pauses.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window-trajectories', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')
    parser.add_argument('--smoothing-window-postures', type=int, default=5,
                        help='Smooth the postures using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--max-speed', type=float, default=0.01,
                        help='Maximum average pause speed to include.')
    parser.add_argument('--min-duration', type=int, default=25,
                        help='Minimum pause duration to include.')

    parser.add_argument('--colouring', type=str, default='conc', choices=['conc', 'curvature'],
                        help='Colour the scatter items by concentration or curvature.')
    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_swt={args.smoothing_window_trajectories}' \
           f'_swp={args.smoothing_window_postures}' \
           f'_sp_max={args.max_speed}' \
           f'_t_min={args.min_duration}'


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    metas = ds.metas
    durations = {}
    activity = {}
    activity_dist = {}
    curvatures = {}

    # Group by concentration and then by reconstruction
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        c = reconstruction.trial.experiment.concentration
        if c not in durations:
            durations[c] = []
            activity[c] = []
            activity_dist[c] = []
            curvatures[c] = []

        # Fetch reconstruction and centre
        X, _ = get_trajectory(reconstruction_id=rid)
        X = X - X.mean(axis=(0, 1), keepdims=True)
        N = X.shape[1]

        # Pick trajectory point and smooth
        if args.trajectory_point == -1:
            Xt = X.mean(axis=1)
        else:
            u = round(args.trajectory_point * N)
            if u == N:
                u -= 1
            assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
            Xt = X[:, u]
        if args.smoothing_window_trajectories > 1:
            Xt = smooth_trajectory(Xt, window_len=args.smoothing_window_trajectories)

        # Smooth the postures
        if args.smoothing_window_postures > 1:
            Xp = smooth_trajectory(X, window_len=args.smoothing_window_postures)
        else:
            Xp = X

        # Calculate speeds
        fps = reconstruction.trial.fps
        ut = calculate_speeds(Xt) * fps
        up = np.linalg.norm(np.gradient(Xp, axis=0), axis=-1) * fps

        # Get natural frame representation for curvatures
        Z, _ = get_trajectory(reconstruction_id=rid, natural_frame=True,
                              smoothing_window=args.smoothing_window_postures)
        K = np.abs(Z)

        # Collate pauses
        durations_i = []
        activity_i = []
        activity_dist_i = []
        curvatures_i = []
        pause_centre_idxs, pause_props = find_peaks(ut < args.max_speed, width=args.min_duration)
        for j in range(len(pause_centre_idxs)):
            pause_start_idx = pause_props['left_bases'][j] + 1
            pause_end_idx = pause_props['right_bases'][j]
            durations_i.append(pause_props['widths'][j] / fps)
            activity_i.append(up[pause_start_idx:pause_end_idx].mean())
            activity_dist_i.append(up[pause_start_idx:pause_end_idx].mean(axis=0))
            curvatures_i.append(K[pause_start_idx:pause_end_idx].mean())

        durations[c].extend(durations_i)
        activity[c].extend(activity_i)
        activity_dist[c].extend(activity_dist_i)
        curvatures[c].extend(curvatures_i)

    # Sort by concentration
    durations = {k: np.array(v) for k, v in sorted(list(durations.items()))}
    activity = {k: np.array(v) for k, v in sorted(list(activity.items()))}
    activity_dist = {k: np.array(v) for k, v in sorted(list(activity_dist.items()))}
    curvatures = {k: np.array(v) for k, v in sorted(list(curvatures.items()))}

    return durations, activity, activity_dist, curvatures


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[Dataset, Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    durations = None
    activity = None
    activity_dist = None
    curvatures = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            durations = {}
            activity = {}
            activity_dist = {}
            curvatures = {}
            for name in data.files:
                c, d_or_a = name.split('_')
                if d_or_a != 'durations':
                    continue
                c = float(c)
                durations[c] = data[name]
                activity[c] = data[f'{c}_activity']
                activity_dist[c] = data[f'{c}_activity-dist']
                curvatures[c] = data[f'{c}_curvatures']
            durations = {k: v for k, v in sorted(list(durations.items()))}
            activity = {k: v for k, v in sorted(list(activity.items()))}
            activity_dist = {k: v for k, v in sorted(list(activity_dist.items()))}
            curvatures = {k: v for k, v in sorted(list(curvatures.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            durations = None
            logger.warning(f'Could not load cache: {e}')

    if durations is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        durations, activity, activity_dist, curvatures = _calculate_data(args, ds)
        data = {}
        for c, d_vals in durations.items():
            data[f'{c}_durations'] = d_vals
            data[f'{c}_activity'] = activity[c]
            data[f'{c}_activity-dist'] = activity_dist[c]
            data[f'{c}_curvatures'] = curvatures[c]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, durations, activity, activity_dist, curvatures


def plot_pauses():
    """
    Plot the pauses, durations against activity.
    """
    args = parse_args()
    ds, durations, activity, activity_dist, curvatures = _generate_or_load_data(args, rebuild_cache=False,
                                                                                cache_only=False)

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    d_vals = np.concatenate([d for c, d in durations.items() if _is_included(c)])
    a_vals = np.concatenate([a for c, a in activity.items() if _is_included(c)])
    ad_vals = np.concatenate([ad for c, ad in activity_dist.items() if ad.ndim == 2 and _is_included(c)])
    k_vals = np.concatenate([k for c, k in curvatures.items() if _is_included(c)])
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Set up plot
    plt.rc('axes', labelsize=9)  # fontsize of the X label
    plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=8)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=3)
    plt.rc('ytick.major', pad=2, size=3)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 2.5),
        height_ratios=(2.2, 3),
        wspace=0,
        hspace=0,
        top=0.97,
        bottom=0.14,
        left=0.1,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, 3))

    # Make scatter plot
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_scat.set_xlabel('Duration (s)', labelpad=5)
    ax_scat.set_ylabel('Activity (mm/s)', labelpad=8)
    legend = None
    if args.colouring == 'conc':
        cm = plt.get_cmap('jet')
        concs = [float(k) for k in durations.keys() if _is_included(k)]
        ticks = np.arange(min(concs), max(concs) + 0.25, 0.25)
        all_colours = cm(np.linspace(0, 1, len(ticks)))
        colours = {c: all_colours[i] for i, c in enumerate(ticks) if c in concs}
        for c, durations_c in durations.items():
            if not _is_included(c):
                continue
            ax_scat.scatter(durations_c, activity[c], label=f'{c:.2f}% ({len(durations_c):,d})', s=20, marker='o',
                            facecolors='none', edgecolors=colours[c], alpha=0.6)
        legend = ax_scat.legend()
    else:
        sc = ax_scat.scatter(d_vals, a_vals, s=20, marker='$\u25EF$', c=k_vals, alpha=0.6, cmap='coolwarm')
        cb = fig.colorbar(sc)
    ax_scat.spines['top'].set_visible(False)
    ax_scat.spines['right'].set_visible(False)

    # Duration histogram
    ax_hist_duration = fig.add_subplot(gs[0, 0])
    ax_hist_duration.hist(d_vals, orientation='vertical', color='green', **hist_args)
    ax_hist_duration.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_hist_duration.spines['bottom'].set(linestyle='--', color='grey')
    ax_hist_duration.set_ylabel('Density')
    ax_hist_duration.set_yticks([0.5])

    # Activity histogram
    ax_hist_activity = fig.add_subplot(gs[1, 1])
    ax_hist_activity.hist(d_vals, orientation='horizontal', color='blue', **hist_args)
    ax_hist_activity.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_activity.spines['left'].set(linestyle='--', color='grey')
    ax_hist_activity.set_xlabel('Density')
    ax_hist_activity.set_xticks([0.5])

    # Activity distribution
    ax_dist = fig.add_subplot(gs[0, 1])
    N = ad_vals.shape[1]
    x = np.arange(N)
    data = ad_vals / ad_vals.mean(axis=1, keepdims=True)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    ax_dist.plot(x, means, color='orange', linewidth=3)
    lb = np.clip(means - 2 * stds, a_min=0, a_max=np.inf)
    ub = means + 2 * stds
    ax_dist.fill_between(x, lb, ub, color='orange', alpha=0.2, linewidth=0)
    ax_dist.axhline(y=1, linestyle=':', color='grey', zorder=-1)
    ax_dist.set_xlim(left=0, right=N)
    ax_dist.set_xticks([0, N])
    ax_dist.set_xticklabels(['H', 'T'])
    ax_dist.set_ylim(bottom=0)
    ax_dist.set_ylabel('Relative activity', labelpad=1)
    ax_dist.set_yticks([0, 1, 2, 3])
    pos = ax_dist.get_position().bounds
    wpad = 0.06
    hpad = 0.07
    ax_dist.set_position(Bbox.from_bounds(
        pos[0] + wpad,  # xmin
        pos[1] + hpad,  # ymin
        pos[2] - wpad - 0.005,  # width
        pos[3] - hpad,  # height
    ))

    if legend is not None:
        handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]
        ax_hist_duration.legend(handles, labels, loc='upper right', ncol=2, columnspacing=0.7, handletextpad=0.2)
        legend.remove()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_pauses'
                            f'_{_identifiers(args)}'
                            f'_c={args.colouring}'
                            f'_n={len(d_vals)}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def plot_activity_dist():
    """
    Plot the activity along the body during pauses.
    """
    args = parse_args()
    ds, durations, activity, activity_dist, curvatures = _generate_or_load_data(args, rebuild_cache=False,
                                                                                cache_only=False)
    a_vals = np.concatenate([a for a in activity.values()])
    ad_vals = np.concatenate([ad for ad in activity_dist.values() if ad.ndim == 2])

    x = np.arange(ad_vals.shape[1])
    idxs = a_vals > 0.05
    data = ad_vals[idxs]
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    plt.plot(x, means, color='blue', linewidth=0.5)
    lb = np.clip(means - 2 * stds, a_min=1e-4, a_max=np.inf)
    ub = means + 2 * stds
    plt.fill_between(x, lb, ub, color='blue', alpha=0.3, linewidth=0)
    plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    plot_pauses()
    # plot_activity_dist()
