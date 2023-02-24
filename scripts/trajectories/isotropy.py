import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

show_plots = True
save_plots = False
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to check isotropy.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}'


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')

    # Use the reconstruction from the dataset where possible
    reconstructions = {}
    for r_ref in ds.reconstructions:
        r = Reconstruction.objects.get(id=r_ref.id)
        reconstructions[r.trial.id] = r.id

    # Calculate the isotropy for all trials
    poses = []  # todo: fix the cameras!!
    matrices = []
    end_points = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating isotropy for trial={trial.id}.')
        if trial.id in reconstructions:
            X, _ = get_trajectory(
                reconstruction_id=reconstructions[trial.id],
                trajectory_point=args.trajectory_point,
                smoothing_window=args.smoothing_window
            )
        else:
            X, _ = get_trajectory(
                trial_id=trial.id,
                tracking_only=True,
                smoothing_window=args.smoothing_window
            )
        X = X.squeeze()
        if X.ndim != 2:
            raise RuntimeError('Wrong shape trajectory!')

        c = trial.experiment.concentration
        if c not in end_points:
            end_points[c] = []
        end_points[c].append(X[-1] - X[0])

        cams = trial.get_cameras()
        poses.append(np.array(cams.pose))
        matrices.append(np.array(cams.matrix))

    poses = np.array(poses)
    matrices = np.array(matrices)

    # Sort by concentration
    end_points = {c: np.array(v) for c, v in sorted(list(end_points.items()))}

    return end_points


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
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            concs = data['concs']
            res = {}
            for c in concs:
                res[float(c)] = data[f'end_points_{c}']
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = _calculate_data(args, ds)
        data = {'concs': np.array([f'{c:.2f}' for c in res.keys()])}
        for c, x in res.items():
            data[f'end_points_{c:.2f}'] = x
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return res


def plot_isotropy_by_concentrations(
        layout: str = 'thesis'
):
    """
    Plot the isotropy across concentrations.
    """
    args = parse_args()
    res = _generate_or_load_data(args, rebuild_cache=True, cache_only=False)

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    # Set up plot
    if layout == 'thesis':
        plt.rc('axes', labelsize=9, labelpad=0)  # fontsize of the X label
        plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=8)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)
        fig, axes = plt.subplots(1, 3, figsize=(12, 2), gridspec_kw=dict(
            wspace=0.14,
            top=0.97,
            bottom=0.16,
            left=0.035,
            right=0.995,
        ))
    else:
        plt.rc('axes', labelsize=7)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)

    # Determine positions
    concs = [c for c in res.keys() if _is_included(c)]
    positions = np.arange(len(concs))
    lim = np.max([np.max(np.abs(res[c])) for c in concs])

    for i, xyz in enumerate('xyz'):
        ax = axes[i]
        ax.set_ylabel(xyz)
        values = []

        for c in concs:
            values.append(res[c][:, i])

        ax.violinplot(
            values,
            positions,
            widths=0.5,
            showmeans=False,
            showmedians=False,
        )

        ax.axhline(y=0, linestyle='--', color='red')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{c:.2f}%\n({len(res[c])})' for c in concs])  #
        ax.set_ylim(bottom=-lim, top=lim)

    if save_plots:
        if len(args.restrict_concs) > 0:
            c_str = '_c=' + ','.join([f'{c:.2f}' for c in args.restrict_concs])
        else:
            c_str = ''

        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_isotropy'
                            f'_{_identifiers(args)}'
                            f'{c_str}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def plot_isotropy_summary(
        layout: str = 'thesis'
):
    """
    Plot the pauses, durations against activity.
    """
    args = parse_args()
    res = _generate_or_load_data(args, rebuild_cache=True, cache_only=False)

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    # Collate data
    concs = [c for c in res.keys() if _is_included(c)]
    data = [[], [], []]
    for c in concs:
        for i in range(3):
            data[i].extend(res[c][:, i])
    data = np.array(data).T
    lim = np.abs(data).max()

    # Set up plot
    if layout == 'thesis':
        plt.rc('axes', labelsize=9, labelpad=0)  # fontsize of the X label
        plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=8)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)
        fig, ax = plt.subplots(1, figsize=(4, 3), gridspec_kw=dict(
            top=0.97,
            bottom=0.07,
            left=0.13,
            right=0.99,
        ))
    else:
        plt.rc('axes', labelsize=7)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)

    positions = np.arange(3)
    ax.violinplot(
        data,
        positions,
        widths=0.5,
        showmeans=False,
        showmedians=False,
    )

    ax.set_ylabel('Displacement (mm)')
    ax.axhline(y=0, linestyle='--', color='red')
    ax.set_xticks(positions)
    ax.set_xticklabels([xyz for xyz in 'xyz'])
    ax.set_ylim(bottom=-lim * 1.1, top=lim * 1.1)

    if save_plots:
        if len(args.restrict_concs) > 0:
            c_str = '_c=' + ','.join([f'{c:.2f}' for c in args.restrict_concs])
        else:
            c_str = ''

        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_isotropy_summary'
                            f'_{_identifiers(args)}'
                            f'{c_str}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    # plot_isotropy_by_concentrations(layout='thesis')
    plot_isotropy_summary(layout='thesis')
