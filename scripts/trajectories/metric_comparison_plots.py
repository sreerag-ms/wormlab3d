import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.postures.helicities import calculate_helicities, calculate_trajectory_helicities
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds, calculate_htd

show_plots = True
save_plots = True
img_extension = 'png'


def make_filename(metric_a: str, metric_b: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'{metric_a}_vs_{metric_b}'

    for k in ['dataset', 'trial', 'frames', 'src', 'u', 'smoothing_window', 'directionality', 'projection']:
        if k in excludes:
            continue
        if k == 'dataset' and args.dataset is not None:
            fn += f'_dataset={args.dataset}'
        elif k == 'trial' and args.trial is not None:
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            fn += f'_{args.midline3d_source}'
        elif k == 'u' and args.trajectory_point is not None:
            fn += f'_u={args.trajectory_point}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'projection' and args.projection is not None:
            fn += f'_p={args.projection}'

    # Add K-estimation params
    if metric_a == 'K' or metric_b == 'K':
        fn += f'_Knf={args.K_sample_frames}'
        fn += f'_K0={args.K0}'

    # Add planarity window parameter
    if metric_a in ['planarity', 'nonp'] or metric_b in ['planarity', 'nonp']:
        fn += f'_pw={args.planarity_window}'

    # Add helicity window parameter
    if metric_a == 'helicity_t' or metric_b == 'helicity_t':
        fn += f'_hw={args.helicity_window}'

    return LOGS_PATH / (fn + '.' + img_extension)


def make_title(metric_a: str, metric_b: str, args: Namespace):
    speed_title = 'Speed'
    K_est_title = f'K estimate ({args.K_sample_frames} frames, K0={args.K0})'
    htd_title = 'HTD'
    planarity_title = f'Planarity ({args.planarity_window} frames)'
    nonp_title = f'Non-planarity ({args.planarity_window} frames)'
    helicity_p_title = f'Helicity of postures'
    helicity_t_title = f'Helicity of trajectory (u={args.trajectory_point}, {args.helicity_window} frames)'

    t = []
    for metric in [metric_a, metric_b]:
        if metric == 'speed':
            t.append(speed_title)
        elif metric == 'K':
            t.append(K_est_title)
        elif metric == 'htd':
            t.append(htd_title)
        elif metric == 'planarity':
            t.append(planarity_title)
        elif metric == 'nonp':
            t.append(nonp_title)
        elif metric == 'helicity_p':
            t.append(helicity_p_title)
        elif metric == 'helicity_t':
            t.append(helicity_t_title)
        else:
            raise RuntimeError(f'Unrecognised metric: {metric}.')
    t = ' vs '.join(t) + '.'

    if args.dataset is not None:
        t += f' Dataset {args.dataset}.'
    elif args.trial is not None:
        t += f' Trial {args.trial}.'

    if args.smoothing_window is not None:
        t += f'\nSmoothing window = {args.smoothing_window} frames.'

    return t


def _get_metric_values(metric: str, X: np.ndarray, args: Namespace) -> np.ndarray:
    if metric == 'speed':
        vals = calculate_speeds(X, signed=True)
    elif metric == 'htd':
        vals = calculate_htd(X)
    elif metric in ['planarity', 'nonp']:
        pcas, _ = generate_or_load_pca_cache(
            reconstruction_id=args.reconstruction,
            smoothing_window=args.smoothing_window,
            window_size=args.planarity_window,
        )
        r = pcas.explained_variance_ratio.T
        vals = r[2] / np.sqrt(r[1] * r[0])
        if metric == 'planarity':
            vals = 1 - vals
    elif metric == 'K':
        vals = get_K_estimates_from_args(args)
    elif metric == 'helicity_p':
        if X.ndim == 2:
            tp = args.trajectory_point
            args.trajectory_point = -1
            X = get_trajectory_from_args(args)
            args.trajectory_point = tp
        vals = calculate_helicities(X)
    elif metric == 'helicity_t':
        if X.ndim == 3:
            args.trajectory_point = 0.1
            X = get_trajectory_from_args(args)
        vals = calculate_trajectory_helicities(X, args.helicity_window)
    else:
        raise RuntimeError(f'Unrecognised metric {metric}.')

    return vals


def _get_label_for_metric(metric: str):
    if metric == 'speed':
        vals = 'Speed'
    elif metric == 'htd':
        vals = 'HTD'
    elif metric == 'planarity':
        vals = 'Planarity'
    elif metric == 'nonp':
        vals = 'Non-planarity'
    elif metric == 'K':
        vals = 'K_est'
    elif metric == 'helicity_p':
        vals = 'Hp'
    elif metric == 'helicity_t':
        vals = 'Ht'
    else:
        raise RuntimeError(f'Unrecognised metric {metric}.')

    return vals


def plot_reconstruction_metrics(metric_a: str, metric_b: str):
    """
    Plot metrics for a single reconstruction.
    """
    args = get_args()
    assert args.reconstruction is not None or args.trial is not None

    X = get_trajectory_from_args(args)
    vals_a = _get_metric_values(metric_a, X, args)
    vals_b = _get_metric_values(metric_b, X, args)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=vals_a, y=vals_b, s=2, alpha=0.4)
    ax.set_xlabel(_get_label_for_metric(metric_a))
    ax.set_ylabel(_get_label_for_metric(metric_b))
    ax.set_title(make_title(metric_a, metric_b, args))
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename(metric_a, metric_b, args))
    if show_plots:
        plt.show()


def plot_dataset_metrics(metric_a: str, metric_b: str):
    """
    Plot dataset metrics.
    """
    args = get_args()

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset midline source args, trajectory point and trial
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.trajectory_point = None
    args.trial = None

    # Loop over reconstructions
    res = {}
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        logger.info(f'Calculating data for reconstruction={reconstruction.id}.')
        X = get_trajectory_from_args(args)
        c = reconstruction.trial.experiment.concentration
        if c not in res:
            res[c] = {metric_a: [], metric_b: []}
        vals_a = _get_metric_values(metric_a, X, args)
        vals_b = _get_metric_values(metric_b, X, args)
        res[c][metric_a].extend(vals_a)
        res[c][metric_b].extend(vals_b)

    # Sort by concentration
    res = {k: v for k, v in sorted(list(res.items()))}
    concs = list(res.keys())

    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    if metric_a == 'speed':
        ax.axvline(x=0, linestyle='--', color='grey')
    elif metric_b == 'speed':
        ax.axhline(y=0, linestyle='--', color='grey')

    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, len(concs)))
    for i, (c, res_c) in enumerate(res.items()):
        ax.scatter(x=res_c[metric_a], y=res_c[metric_b], s=0.05, alpha=0.3, color=colours[i], label=f'c={c:.2f}%')
    ax.set_xlabel(_get_label_for_metric(metric_a))
    ax.set_ylabel(_get_label_for_metric(metric_b))
    ax.set_title(make_title(metric_a, metric_b, args))
    ax.legend(markerscale=50)
    fig.tight_layout()

    if save_plots:
        plt.savefig(make_filename(metric_a, metric_b, args, excludes=['trial', 'frames', 'src']))
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    args = get_args()
    if args.dataset is not None:
        plot_dataset_metrics('htd', 'speed')
        plot_dataset_metrics('nonp', 'speed')
        plot_dataset_metrics('htd', 'nonp')
    else:
        # plot_reconstruction_metrics('htd', 'speed')
        plot_reconstruction_metrics('helicity_p', 'helicity_t')
