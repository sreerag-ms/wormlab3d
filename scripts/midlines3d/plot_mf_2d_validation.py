import os
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mongoengine import DoesNotExist
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial, Dataset, Midline2D, Frame
from wormlab3d.data.model.dataset import DatasetMidline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, M3D_SOURCE_RECONST, M3D_SOURCE_WT3D, Midline3D, M3D_SOURCES
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args, str2bool

colours = {
    'rec': 'blue',
    'ann': 'darkorange',
}

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'


class NothingToCompare(Exception):
    pass


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to validate against hand-annotated 2D midlines.')

    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--align', type=str2bool, default=False, help='Align the points to minimise the errors.')
    parser.add_argument('--frame-num', type=int, help='Frame number.')
    parser.add_argument('--user', type=str, help='Restrict to annotations made by this user.')
    parser.add_argument('--camera', type=int, help='Restrict to this camera idx.')
    parser.add_argument('--plot-trial-errors', type=str2bool, default=True, help='Plot the trial errors.')
    parser.add_argument('--plot-n-examples', type=int, default=10, help='Number of examples to plot per trial.')
    args = parser.parse_args()

    print_args(args)

    return args


def _plot_frame_comparison(
        m2d: Midline2D,
        p2d_man: np.ndarray,
        p2d_rec: np.ndarray,
        dists_rec: np.ndarray,
        dists_man: np.ndarray,
        save_dir: Path
):
    """
    Plot the manual annotation vs the reconstruction.
    """
    frame = m2d.frame
    img = frame.images[m2d.camera]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f'Trial {frame.trial.id}. '
        f'Frame {frame.frame_num}. '
        f'Camera {m2d.camera}.'
    )

    ax = axes[0, 0]
    ax.set_title('Annotation' + (f' ({m2d.user})' if m2d.user else ''))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.scatter(x=p2d_man[:, 0], y=p2d_man[:, 1], color='red', s=10, alpha=0.8)
    ax.axis('off')

    ax = axes[0, 1]
    ax.set_title('Reconstruction')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.scatter(x=p2d_rec[:, 0], y=p2d_rec[:, 1], color='red', s=10, alpha=0.8)
    ax.axis('off')

    ax = axes[1, 0]
    ax.set_title('Min. distances\nRec->Annotation')
    ax.plot(dists_rec)

    ax = axes[1, 1]
    ax.set_title('Min. distances\nAnnotation->Rec')
    ax.plot(dists_man)

    ax = axes[2, 0]
    ax.set_title('')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.scatter(x=p2d_rec[:, 0], y=p2d_rec[:, 1], c=dists_rec, cmap='autumn', alpha=0.8)
    ax.axis('off')

    ax = axes[2, 1]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.scatter(x=p2d_rec[:, 0], y=p2d_rec[:, 1], c=dists_man, cmap='autumn', alpha=0.8)
    ax.axis('off')

    fig.tight_layout()

    if save_plots:
        path = save_dir / (f'frame={frame.frame_num:05d}'
                           f'_cam={m2d.camera}'
                           + (f'_{m2d.user}' if m2d.user != '' else '')
                           + f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_simple_frame_comparison():
    """
    Plot a simple single frame comparison.
    """
    args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    assert args.frame_num is not None, 'This script requires setting --frame-num.'
    assert rec.start_frame_valid <= args.frame_num <= rec.end_frame_valid, 'This frame number is not valid for the reconstruction!'
    frame: Frame = rec.trial.get_frame(args.frame_num)

    # Get 2D midline
    filters = {'frame': frame}
    if args.user is not None:
        filters['user'] = None if args.user == '-' else args.user
    if args.camera is not None:
        filters['camera'] = args.camera
    m2ds = Midline2D.objects(**filters)
    if m2ds.count() == 0:
        raise NothingToCompare('No 2D midlines found to compare!')
    if m2ds.count() > 1:
        raise NothingToCompare(f'Multiple ({m2ds.count()}) 2D midlines found to compare!')
    m2d = m2ds[0]

    # Load the reconstruction data
    ts = TrialState(rec)
    p2d = ts.get('points_2d')

    # Get the image points
    p2d_rec = p2d[frame.frame_num, :, m2d.camera]
    p2d_man = m2d.get_prepared_coordinates()
    if len(p2d_man) == 0:
        raise NothingToCompare('Prepared coordinates empty!')

    # Calculate distances
    dists = cdist(p2d_man, p2d_rec)
    dists_rec = dists.min(axis=0)
    dists_man = dists.min(axis=1)
    d_min = 0
    d_max = max(dists_rec.max(), dists_man.max())

    # Get image
    img = 1 - frame.images[m2d.camera]

    # Set up plot
    plt.rc('axes', labelsize=6)  # fontsize of the X label
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('xtick.major', pad=2, size=1)
    plt.rc('ytick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick.major', pad=2, size=2)

    fig, axes = plt.subplots(1, 2, figsize=(3, 1), gridspec_kw={
        'left': 0.03,
        'right': 0.8,
        'top': 0.98,
        'bottom': 0.02,
        'wspace': 0.05,
    })

    scatter_args = dict(vmin=d_min, vmax=d_max, cmap='Reds', alpha=0.6, s=3)
    crop_size = 160
    m = int(img.shape[0] - crop_size) / 2  # margin to remove
    lims = (m, img.shape[0] - m)

    def _plot_img(ax_, p2d_, dists_):
        ax_.imshow(img, cmap='gray', vmin=0, vmax=1)
        scat = ax_.scatter(x=p2d_[:, 0], y=p2d_[:, 1], c=dists_, **scatter_args)
        ax_.set_xlim(left=lims[0], right=lims[1])
        ax_.set_ylim(bottom=lims[0], top=lims[1])
        ax_.axis('off')
        return scat

    ax = axes[0]
    _plot_img(ax, p2d_man, dists_man)
    ax = axes[1]
    scat = _plot_img(ax, p2d_rec, dists_rec)

    # Add colourbar [left, bottom, width, height] in new axis
    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
    cb = fig.colorbar(scat, cax=cbar_ax)
    cb.outline.set_linewidth(0.5)
    cb.solids.set(alpha=1)
    cb.ax.set_ylabel('Pixel distance', labelpad=7, rotation=270)
    cb.set_ticks([0, 10, 20])

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_trial={rec.trial.id:03d}'
                            f'_rec={rec.id}'
                            f'_frame={frame.frame_num:05d}'
                            f'_cam={m2d.camera}'
                            + (f'_{m2d.user}' if m2d.user != '' else '')
                            + f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def validate_reconstruction(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate a reconstruction against any available 2d midlines.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial: Trial = rec.trial

    # Fetch 2D midlines
    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$lookup': {'from': 'midline2d', 'localField': '_id', 'foreignField': 'frame', 'as': 'midline'}},
        {'$unwind': {'path': '$midline'}},
        {'$sort': {'frame_num': 1}},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
            'midline_id': '$midline._id',
        }},
    ]
    cursor = Frame.objects().aggregate(pipeline)
    frame_ids, frame_nums, mids = [], [], []
    for res in cursor:
        frame_ids.append(res['_id'])
        frame_nums.append(res['frame_num'])
        mids.append(res['midline_id'])
    if len(mids) == 0:
        raise NothingToCompare
    logger.info(f'Found {len(mids)} 2d midlines for trial {trial.id}.')

    # Make the plot directories
    if save_dir is None:
        plots_dir = LOGS_PATH / f'{START_TIMESTAMP}' \
                                f'_trial={trial.id:03d}' \
                                f'_rec={rec.id}' \
                                f'_align={args.align}'
    else:
        plots_dir = save_dir / f'trial={trial.id:03d}' \
                               f'_rec={rec.id}'
    if save_plots and args.plot_n_examples > 0 or args.plot_trial_errors:
        os.makedirs(plots_dir, exist_ok=True)

    # Load the reconstruction data
    p2d = {}
    if rec.source == M3D_SOURCE_MF:
        ts = TrialState(rec)
        points_2d = ts.get('points_2d')
        for n in frame_nums:
            if rec.start_frame_valid <= n <= rec.end_frame_valid:
                p2d[n] = points_2d[n]

    else:
        for i, frame_id in enumerate(frame_ids):
            try:
                m3d = Midline3D.objects.get(
                    frame=frame_id,
                    source=rec.source,
                    source_file=rec.source_file,
                )
                X = np.stack(m3d.prepare_2d_coordinates(), axis=1)
                if np.isnan(X).any():
                    continue
                p2d[frame_nums[i]] = X
            except DoesNotExist:
                continue

    def _calculate_error(shift: np.ndarray, p2d_man_: np.ndarray, p2d_rec_: np.ndarray) -> float:
        dists_ = cdist(p2d_man_ + shift, p2d_rec_)
        dists_rec_ = dists_.min(axis=0)
        dists_man_ = dists_.min(axis=1)
        err = dists_rec_.sum() + dists_man_.sum()
        return err

    # Loop over manual annotations
    n_matched = 0
    n_examples_plotted = 0
    errors = {}
    res = {}
    users = {}
    cams = {}
    keys = {}
    for i, mid in enumerate(mids):
        if (i + 1) % 10 == 0:
            logger.info(f'Validating against midline {i + 1}/{len(mids)}.')
        m2d: Midline2D = Midline2D.objects.get(id=mid)
        n = m2d.frame.frame_num
        if n not in p2d:
            continue

        # Get the image points
        p2d_rec = p2d[n][:, m2d.camera]
        p2d_man = m2d.get_prepared_coordinates()
        if len(p2d_man) == 0:
            continue

        # Align the points (accounts for shifts in the camera parameters/bad annotations)
        if args.align:
            s = minimize(_calculate_error, np.array([0, 0]), args=(p2d_man, p2d_rec))
            p2d_man += s.x

        # Calculate distances
        dists = cdist(p2d_man, p2d_rec)

        # Closest distance for each rec point
        dists_rec = dists.min(axis=0)

        # Closest distance for each man point
        dists_man_raw = dists.min(axis=1)

        # Resample to match the same resolution as the rec
        f = interpolate.interp1d(np.linspace(0, 1, len(dists_man_raw)), dists_man_raw, kind='cubic')
        x = np.linspace(0, 1, p2d_rec.shape[0])  # should be N
        dists_man = f(x)

        # Get the overall error
        error = dists_rec.sum() + dists_man.sum()
        if np.isnan(error).any():
            raise RuntimeError('Bad loss!')

        # Plot the comparison
        if n_examples_plotted < args.plot_n_examples:
            _plot_frame_comparison(m2d, p2d_man, p2d_rec, dists_rec, dists_man, plots_dir)
            n_examples_plotted += 1

        # Only record the result if it is for a new frame/cam combination or gives a lower error
        key = f'{n:05d}_{m2d.camera}'
        if key in errors:
            if error > errors[key]:
                continue
            else:
                n_matched -= 1
        n_matched += 1
        keys[key] = key
        users[key] = m2d.user if m2d.user is not None else '-'
        cams[key] = m2d.camera
        res[key] = np.stack([dists_rec, dists_man], axis=-1)
        errors[key] = error

    # Combine the results
    logger.info(f'Matched {n_matched} manual annotations.')
    if n_matched == 0:
        raise NothingToCompare('No annotations found to compare against!')
    keys = list(keys.values())
    users = np.array(list(users.values()))
    cams = np.array(list(cams.values()))
    res = np.stack(list(res.values()))
    means = res.mean(axis=0)
    stds = res.std(axis=0)

    # Plot
    if args.plot_trial_errors:
        logger.info('Plotting trial errors.')
        fig, axes = plt.subplots(1, figsize=(6, 4))
        fig.suptitle(f'Trial {trial.id}. Manual annotations: {n_matched}.')

        def _plot_dist(ax_: Axes, means_: np.ndarray, stds_: np.ndarray, colour: str, label: str):
            ax_.plot(x, means_, color=colour, label=label, linewidth=3, alpha=0.8)
            ax_.fill_between(x, np.clip(means_ - stds_, a_min=0, a_max=np.inf), means_ + stds_, color=colour,
                             alpha=0.2, linewidth=1)

        ax = axes
        _plot_dist(ax, means[:, 0], stds[:, 0], colours['rec'], 'Reconstruction')
        _plot_dist(ax, means[:, 1], stds[:, 1], colours['ann'], 'Annotation')
        ax.legend()
        ax.set_ylabel('Pixel distance')

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=0, right=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['H', 'T'])

        fig.tight_layout()

        if save_plots:
            path = plots_dir / f'errors.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()

    return res, users, cams, keys


def _plot_errors(
        res: np.ndarray,
        save_dir: Path,
        title: str = None,
        filename: str = 'errors',
):
    """
    Plot results
    """
    x = np.linspace(0, 1, res.shape[1])
    means = res.mean(axis=0)
    stds = res.std(axis=0)

    fig, axes = plt.subplots(1, figsize=(6, 4))
    if title is not None:
        fig.suptitle(title)

    def _plot_dist(ax_: Axes, means_: np.ndarray, stds_: np.ndarray, colour: str, label: str):
        ax_.plot(x, means_, color=colour, label=label, linewidth=3, alpha=0.8)
        ax_.fill_between(x, np.clip(means_ - stds_, a_min=0, a_max=np.inf), means_ + stds_, color=colour,
                         alpha=0.2, linewidth=1)

    ax = axes
    _plot_dist(ax, means[:, 0], stds[:, 0], colours['rec'], 'Reconstruction')
    _plot_dist(ax, means[:, 1], stds[:, 1], colours['ann'], 'Annotation')
    ax.legend()

    # Set up x-axis
    ax.set_xticks([])
    ax.set_xlim(left=0, right=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['H', 'T'])

    # Set up y-axis
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Pixel distance')

    fig.tight_layout()

    if save_plots:
        path = save_dir / f'{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def validate_all_in_dataset(plot_all: bool = True):
    """
    Validate all reconstructions in a dataset.
    """
    args = get_args()
    assert args.dataset is not None, 'This script requires setting --dataset=id.'
    ds = Dataset.objects.get(id=args.dataset)
    assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'

    # Make save dir
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_ds={ds.id}_align={args.align}'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Loop over reconstructions
    res = []
    users = []
    cams = []
    n_matched = 0
    for i, rec in enumerate(ds.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(ds.reconstructions)}.')
        args.reconstruction = rec.id
        try:
            res_i, users_i, cams_i, keys_i = validate_reconstruction(args, save_dir)
            res.append(res_i)
            users.append(users_i)
            cams.append(cams_i)
            n_matched += 1
        except NothingToCompare as e:
            logger.info(e)
            continue

    # Combine the results
    logger.info(f'Validated {n_matched} reconstructions.')
    res = np.concatenate(res)
    users_all = np.concatenate(users)
    cams_all = np.concatenate(cams)

    def _make_title(n_matched_: int, n_annotations_: int, filtered: str = None):
        return f'Dataset {ds.id}.' + \
               (f' {filtered}' if filtered is not None else '') + \
               f'\nReconstructions validated: {n_matched_}.' \
               f'\nAnnotations used: {n_annotations_}.'

    if plot_all:
        # Plot all errors combined
        _plot_errors(res, save_dir, _make_title(n_matched, len(res)))

        # Plot errors for each user's annotations
        for user in np.unique(users_all):
            n_matched_user = 0
            for users_rec in users:
                if user in users_rec:
                    n_matched_user += 1
            _plot_errors(
                res[users_all == user],
                save_dir,
                _make_title(n_matched_user, (users_all == user).sum(), f'User: {user}'),
                f'errors_user_{user}'
            )

        # Plot errors for each camera annotations
        for cam in np.unique(cams_all):
            n_matched_cam = 0
            for cams_rec in cams:
                if cam in cams_rec:
                    n_matched_cam += 1
            _plot_errors(
                res[cams_all == cam],
                save_dir,
                _make_title(n_matched_cam, (cams_all == cam).sum(), f'Cam: {cam}'),
                f'errors_cam_{cam}'
            )

    else:
        # Plot simple version of combined results
        x = np.linspace(0, 1, res.shape[1])
        means = res.mean(axis=0)
        stds = res.std(axis=0)

        # Set up plot
        plt.rc('axes', labelsize=6)  # fontsize of the X label
        plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=6)  # fontsize of the legend
        plt.rc('xtick.major', pad=2)
        plt.rc('ytick.major', pad=1, size=2)

        fig, ax = plt.subplots(1, figsize=(3, 1.2), gridspec_kw={
            'left': 0.1,
            'right': 0.96,
            'top': 0.98,
            'bottom': 0.17,
        })

        def _plot_dist(ax_: Axes, means_: np.ndarray, stds_: np.ndarray, colour: str, label: str):
            ax_.plot(x, means_, color=colour, label=label, linewidth=3, alpha=0.8)
            ax_.fill_between(x, np.clip(means_ - stds_, a_min=0, a_max=np.inf), means_ + stds_, color=colour,
                             alpha=0.2, linewidth=1)

        _plot_dist(ax, means[:, 0], stds[:, 0], colours['rec'], 'Reconstructions')
        _plot_dist(ax, means[:, 1], stds[:, 1], colours['ann'], 'Annotations')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=0, right=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['H', 'T'])

        # Set up y-axis
        ax.set_ylim(bottom=0)
        ax.set_yticks([0, 5, 10])
        ax.set_ylabel('Pixel distance', labelpad=1)

        if save_plots:
            path = save_dir / f'errors_simple.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()


def _get_recs_to_compare(trial: Trial) -> Dict[str, Reconstruction]:
    """
    Fetch reconstructions to compare against, max one from each source
    """
    recs = Reconstruction.objects(trial=trial, source__ne=M3D_SOURCE_MF)
    n_results = recs.count()
    if n_results == 0:
        raise NothingToCompare('No reconstructions found to compare against!')
    recs_to_compare = {}
    for rec in recs:
        if rec.source not in recs_to_compare:
            recs_to_compare[rec.source] = rec
        elif rec.source == M3D_SOURCE_RECONST and len(rec.source_file) < len(recs_to_compare[rec.source].source_file):
            recs_to_compare[rec.source] = rec
        elif rec.source == M3D_SOURCE_WT3D:
            sfA = recs_to_compare[rec.source].source_file[:8]
            sfB = rec.source_file[:8]
            if sfB.isnumeric() and (not sfA.isnumeric() or (sfA.isnumeric() and int(sfA) < int(sfB))):
                recs_to_compare[rec.source] = rec

    return recs_to_compare


def validation_comparisons():
    """
    Validate all reconstructions in a dataset and compare between methods.
    """
    args = get_args()
    assert args.dataset is not None, 'This script requires setting --dataset=id.'
    ds = Dataset.objects.get(id=args.dataset)
    assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'

    # Make save dir
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_comparisons_ds={ds.id}_align={args.align}'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Loop over reconstructions
    res_MF = {}
    res_reconst = {}
    res_WT3D = {}
    n_matched = 0
    for i, rec in enumerate(ds.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(ds.reconstructions)}.')

        # Fetch any comparable reconstructions
        try:
            recs_to_compare = _get_recs_to_compare(rec.trial)
        except NothingToCompare as e:
            recs_to_compare = {}
            logger.info(e)

        # Get the validation errors
        for src in M3D_SOURCES:
            if src == M3D_SOURCE_MF:
                res = res_MF
                rec_id = rec.id
            else:
                if src not in recs_to_compare:
                    continue
                if src == M3D_SOURCE_RECONST:
                    res = res_reconst
                else:
                    res = res_WT3D
                rec_id = recs_to_compare[src].id

            args.reconstruction = rec_id
            try:
                res_i, users_i, cams_i, keys_i = validate_reconstruction(args, save_dir)
                for j, key in enumerate(keys_i):
                    res[f'{rec.trial.id}_{key}'] = res_i[j]
            except NothingToCompare as e:
                logger.info(e)
                continue

    # Combine the results
    logger.info(f'Found {n_matched} reconstructions for comparisons.')

    # MF vs reconst
    keys_mf_reconst = set(res_MF.keys()).intersection(set(res_reconst.keys()))
    errors_mf_reconst = np.zeros((2, len(keys_mf_reconst), 2))
    for i, k in enumerate(keys_mf_reconst):
        errors_mf_reconst[0, i] = res_MF[k].mean(axis=0)
        errors_mf_reconst[1, i] = res_reconst[k].mean(axis=0)

    # MF vs WT3D
    keys_mf_wt3d = set(res_MF.keys()).intersection(set(res_WT3D.keys()))
    errors_mf_wt3d = np.zeros((2, len(keys_mf_wt3d), 2))
    for i, k in enumerate(keys_mf_wt3d):
        errors_mf_wt3d[0, i] = res_MF[k].mean(axis=0)
        errors_mf_wt3d[1, i] = res_WT3D[k].mean(axis=0)

    # Print results
    logger.info(
        '\n\n------ MF vs reconst ------\n'
        f'N={len(keys_mf_reconst)}\n'
        f'MF = {errors_mf_reconst[0].mean():.4f} ({errors_mf_reconst[0].std():.4f})\n'
        f'reconst = {errors_mf_reconst[1].mean():.4f} ({errors_mf_reconst[1].std():.4f})\n'
        '\n------ MF vs WT3D ------\n'
        f'N={len(keys_mf_wt3d)}\n'
        f'MF = {errors_mf_wt3d[0].mean():.4f} ({errors_mf_wt3d[0].std():.4f})\n'
        f'WT3D = {errors_mf_wt3d[1].mean():.4f} ({errors_mf_wt3d[1].std():.4f})\n'
    )


def duration_comparisons():
    """
    Count the total reconstruction durations across each source.
    (Doesn't really belong here but...)
    """
    args = get_args()
    assert args.dataset is not None, 'This script requires setting --dataset=id.'
    ds = Dataset.objects.get(id=args.dataset)
    assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'

    # Loop over reconstructions
    num_frames = {src: 0 for src in M3D_SOURCES}
    durations = {src: 0 for src in M3D_SOURCES}
    for i, rec_mf in enumerate(ds.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(ds.reconstructions)}.')

        # Fetch any comparable reconstructions
        try:
            recs_to_compare = _get_recs_to_compare(rec_mf.trial)
        except NothingToCompare:
            recs_to_compare = {}

        # Get the durations
        for src in M3D_SOURCES:
            if src == M3D_SOURCE_MF:
                rec = rec_mf
            else:
                if src not in recs_to_compare:
                    continue
                rec = recs_to_compare[src]

            num_frames[src] += rec.n_frames
            durations[src] += rec.n_frames / rec.trial.fps

    # Print the results
    print('num_frames', num_frames)
    print('durations', durations)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    # plot_simple_frame_comparison()
    # validate_reconstruction()
    # validate_all_in_dataset(plot_all=False)
    validation_comparisons()
    # duration_comparisons()
