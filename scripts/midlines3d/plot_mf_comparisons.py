import os
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate
from torch.backends import cudnn

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGES_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial, Dataset
from wormlab3d.data.model.dataset import DatasetMidline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D, M3D_SOURCE_WT3D, M3D_SOURCE_RECONST
from wormlab3d.midlines3d.project_render_score import render_points
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args, to_numpy, str2bool
from wormlab3d.trajectories.cache import get_trajectory

POINTS_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(POINTS_CACHE_PATH, exist_ok=True)

prop_cycle = plt.rcParams['axes.prop_cycle']
default_colours = prop_cycle.by_key()['color']
colours = {k: default_colours[i] for i, k in enumerate([
    M3D_SOURCE_MF,
    M3D_SOURCE_RECONST,
    M3D_SOURCE_WT3D
])}

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to compare MF losses against reconst/WT3D.')

    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size.')
    parser.add_argument('--gpu-id', type=int, default=-1, help='GPU id to use if using GPUs.')
    parser.add_argument('--x-label', type=str, default='frame', help='Label x-axis with time or frame number.')
    parser.add_argument('--stats-window', type=int, default=5, help='Averaging window for the stats.')
    parser.add_argument('--rebuild-cache', type=str2bool, default=False, help='Rebuild caches.')
    parser.add_argument('--cache-only', type=str2bool, default=False, help='Use cache only.')
    parser.add_argument('--plot-n-examples', type=int, default=3, help='Number of examples to plot.')
    parser.add_argument('--plot-example-frames', type=lambda s: [int(item) for item in s.split(',')],
                        help='Plot these frame numbers.')
    args = parser.parse_args()

    print_args(args)

    return args


def _get_recs_to_compare(trial: Trial) -> Dict[str, Reconstruction]:
    """
    Fetch reconstructions to compare against, max one from each source
    """
    recs = Reconstruction.objects(trial=trial, source__ne=M3D_SOURCE_MF)
    n_results = recs.count()
    if n_results == 0:
        raise RuntimeError('No reconstructions found to compare against!')
    recs_to_compare = {}
    for rec in recs:
        if rec.source not in recs_to_compare \
                or (rec.source in recs_to_compare
                    and len(rec.source_file) < len(recs_to_compare[rec.source].source_file)):
            recs_to_compare[rec.source] = rec

    return recs_to_compare


def _init_devices(args: Namespace):
    """
    Find available devices and try to use what we want.
    """
    if args.gpu_id == -1:
        cpu_or_gpu = 'cpu'
    else:
        cpu_or_gpu = 'gpu'

    if cpu_or_gpu == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info('Using GPU.')
        cudnn.benchmark = True  # optimises code for constant input sizes
    else:
        if cpu_or_gpu == 'gpu':
            raise RuntimeError('GPU requested but not available. Aborting.')
        logger.info('Using CPU.')

    return device


def _resample(X: np.ndarray, N: int) -> np.ndarray:
    """
    Resample 3D curve to N vertices.
    """
    if X.shape[1] != N:
        X_new = np.zeros((N, 3))
        sl = np.linalg.norm(X[:-1] - X[1:], axis=-1)
        u = np.r_[np.array([0, ]), sl.cumsum(axis=-1)]
        u = u / u[-1]
        u_new = np.linspace(0, 1, N)

        for j in range(3):
            tck = interpolate.splrep(u, X[:, j], s=1e-4, k=3)
            X_new[:, j] = interpolate.splev(u_new, tck)

        X = X_new

    return X


def _overlay_images(
        images: np.ndarray,
        points_2d: np.ndarray,
        midline_width: int = 3
) -> Image:
    """
    Make triplet of images with overlaid midlines for debug.
    """
    if points_2d.shape[1] == 3:
        points_2d = points_2d.transpose(1, 0, 2)
    points_2d = points_2d.astype(np.int32)
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = cmap(np.linspace(0, 1, points_2d.shape[1]))
    colours = np.round(colours * 255).astype(np.uint8)

    views = []
    for c, img_array in enumerate(images):
        z = (img_array * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGRA)
        p2d = points_2d[c]

        for j, p in enumerate(p2d):
            col = colours[j].tolist()

            # Draw markers and connecting lines
            z = cv2.drawMarker(
                z,
                p,
                color=col,
                markerType=cv2.MARKER_CROSS,
                markerSize=2,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(z, p2d[j - 1], p2d[j], color=col, thickness=midline_width, lineType=cv2.LINE_AA)

        views.append(z)

    # Convert to PIL image
    img = Image.fromarray(np.concatenate(views, axis=1), 'RGBA')

    return img


def _calculate_2d_data(
        rec: Reconstruction,
        N: int,
        rebuild_cache: bool = False,
) -> np.ndarray:
    """
    Calculate the r values across a range of sigmas, durations and pauses.
    """
    frame_nums = np.arange(rec.start_frame, rec.end_frame + 1)
    X = np.zeros((len(frame_nums), N, 3, 2))
    for j, frame_num in enumerate(frame_nums):
        if (j + 1) % 10 == 0:
            logger.info(f'Preparing 2D data for frame {j + 1}/{len(frame_nums)}')
        frame = rec.trial.get_frame(frame_num)
        m3d = Midline3D.objects.get(
            frame=frame.id,
            source=rec.source,
            source_file=rec.source_file,
        )
        if len(m3d.X) == N:
            X[j] = np.stack(m3d.get_prepared_2d_coordinates(regenerate=rebuild_cache), axis=1)
        else:
            Xr = _resample(m3d.X, N)
            X[j] = np.stack(m3d.prepare_2d_coordinates(X=Xr), axis=1)

    return X


def _generate_or_load_2d_data(
        rec: Reconstruction,
        N: int,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the 2d data.
    """
    cache_path = POINTS_CACHE_PATH / f'rec_{rec.id}_N={N}'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    data = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            data = data['data']
            n_frames = rec.end_frame - rec.start_frame + 1
            if len(data) != n_frames:
                raise RuntimeError(f'Number of points {len(data)} != expected {n_frames}.')
            logger.info(f'Loaded points data from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating points data.')
        data = _calculate_2d_data(rec, N, rebuild_cache)
        save_arrs = {'data': data}
        logger.info(f'Saving points data to {cache_fn}.')
        np.savez(cache_path, **save_arrs)

    return data


def _fetch_2d_data(
        rec_mf: Reconstruction,
        recs_to_compare: Dict[str, Reconstruction],
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> List[np.ndarray]:
    """
    Fetch the 2d data
    """
    N = rec_mf.mf_parameters.n_points_total

    # Fetch the MF data directly
    ts = TrialState(rec_mf, start_frame=rec_mf.start_frame_valid,
                    end_frame=rec_mf.end_frame_valid + 1)
    Xs = [ts.get('points_2d'), ]

    # Load cached data for the comparisons
    for i, (src, rec) in enumerate(recs_to_compare.items()):
        X = _generate_or_load_2d_data(rec, N, rebuild_cache, cache_only)
        if rec.start_frame < rec_mf.start_frame_valid:
            X = X[rec_mf.start_frame_valid - rec.start_frame:]
        if rec.end_frame > rec_mf.end_frame_valid:
            X = X[:rec_mf.end_frame_valid - rec.end_frame]
        Xs.append(X)

    return Xs


def _make_renders(
        points_2d: torch.Tensor,
        sigmas: torch.Tensor,
        sigmas_min: float,
        exponents: torch.Tensor,
        intensities: torch.Tensor,
        intensities_min: float,
        camera_sigmas: torch.Tensor,
        camera_exponents: torch.Tensor,
        camera_intensities: torch.Tensor,
        image_size: int
) -> torch.Tensor:
    """
    Render the 2D points into images.
    """
    N = points_2d.shape[1]
    device = points_2d.device

    # Prepare sigmas, exponents and intensities
    N5 = int(N / 5)

    # Sigmas should be equal in the middle section but taper towards the ends
    sigmas = sigmas.clamp(min=sigmas_min)
    slopes = (sigmas - sigmas_min) / N5 * torch.arange(N5, device=device)[None, :] + sigmas_min
    sigmas = torch.cat([
        slopes,
        torch.ones(1, N - 2 * N5, device=device) * sigmas,
        slopes.flip(dims=(1,))
    ], dim=1)

    # Make exponents equal everywhere
    exponents = torch.ones(1, N, device=device) * exponents

    # Intensities should be equal in the middle section but taper towards the ends
    intensities = intensities.clamp(min=intensities_min)
    slopes = (intensities - intensities_min) / N5 \
             * torch.arange(N5, device=device)[None, :] + intensities_min
    intensities = torch.cat([
        slopes,
        torch.ones(1, N - 2 * N5, device=device) * intensities,
        slopes.flip(dims=(1,))
    ], dim=1)

    masks, blobs = render_points(
        points_2d.transpose(1, 2),
        sigmas,
        exponents,
        intensities,
        camera_sigmas,
        camera_exponents,
        camera_intensities,
        image_size,
    )

    return masks


def _calculate_errors(
        rec_mf: Reconstruction,
        rec: Reconstruction,
        points_2d: np.ndarray,
        batch_size: int,
        device: torch.device
) -> np.ndarray:
    """
    Render the 2D points and compute errors.
    """
    n_frames = len(points_2d)
    n_batches = int(n_frames / batch_size) + 1
    errors = np.zeros(n_frames)

    # Rendering parameters come from the MF reconstruction
    ts = TrialState(
        rec_mf,
        start_frame=max(rec.start_frame, rec_mf.start_frame_valid),
        end_frame=min(rec.end_frame, rec_mf.end_frame_valid) + 1
    )
    sigmas = ts.get('sigmas')
    intensities = ts.get('intensities')
    exponents = ts.get('exponents')
    camera_sigmas = ts.get('camera_sigmas')
    camera_intensities = ts.get('camera_intensities')
    camera_exponents = ts.get('camera_exponents')

    for i in range(n_batches):
        logger.info(f'Calculating errors for batch {i + 1}/{n_batches}.')
        start_idx = i * batch_size
        end_idx = min(n_frames + 1, (i + 1) * batch_size)
        if end_idx == start_idx:
            continue
        renders = _make_renders(
            points_2d=torch.from_numpy(points_2d[start_idx:end_idx]).to(device),
            sigmas=torch.from_numpy(sigmas[start_idx:end_idx]).to(device),
            sigmas_min=ts.parameters.sigmas_min,
            exponents=torch.from_numpy(exponents[start_idx:end_idx]).to(device),
            intensities=torch.from_numpy(intensities[start_idx:end_idx]).to(device),
            intensities_min=ts.parameters.intensities_min,
            camera_sigmas=torch.from_numpy(camera_sigmas[start_idx:end_idx]).to(device),
            camera_exponents=torch.from_numpy(camera_exponents[start_idx:end_idx]).to(device),
            camera_intensities=torch.from_numpy(camera_intensities[start_idx:end_idx]).to(device),
            image_size=ts.trial.crop_size
        )

        # Get targets
        start_frame = rec.start_frame + start_idx
        end_frame = rec.start_frame + end_idx
        images = torch.from_numpy(np.stack([
            np.load(PREPARED_IMAGES_PATH / f'{ts.trial.id:03d}' / f'{n:06d}.npz')['images']
            for n in range(start_frame, end_frame)
        ])).to(device)

        # MSE
        errors[start_idx:end_idx] = to_numpy(((renders - images)**2).mean(axis=(1, 2, 3)))

    return errors


def _generate_or_load_errors(
        rec_mf: Reconstruction,
        rec: Reconstruction,
        N: int,
        points_2d: np.ndarray,
        batch_size: int,
        device: torch.device,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the errors.
    """
    cache_path = POINTS_CACHE_PATH / f'rec_{rec.id}_N={N}_errors'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    data = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            data = data['data']
            if len(data) != len(points_2d):
                raise RuntimeError(f'Number of errors {len(data)} != number of points {len(points_2d)}.')
            logger.info(f'Loaded errors from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Calculating errors.')
        data = _calculate_errors(
            rec_mf=rec_mf,
            rec=rec,
            points_2d=points_2d,
            batch_size=batch_size,
            device=device
        )
        save_arrs = {'data': data}
        logger.info(f'Saving errors data to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return data


def _fetch_errors(
        rec_mf: Reconstruction,
        recs_to_compare: Dict[str, Reconstruction],
        batch_size: int,
        device: torch.device,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> List[np.ndarray]:
    """
    Generate or load the errors.
    """
    N = rec_mf.mf_parameters.n_points_total

    # Generate or load the 2D data
    points_2d = _fetch_2d_data(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        rebuild_cache=rebuild_cache,
        cache_only=cache_only,
    )

    # Generate or load pixel-losses
    errors = []
    for i in range(1 + len(recs_to_compare)):
        if i == 0:
            rec = rec_mf
            logger.info(f'Calculating pixel errors for MF reconstruction.')
        else:
            src = list(recs_to_compare.keys())[i - 1]
            rec = recs_to_compare[src]
            logger.info(f'Calculating pixel errors for rec={rec.id}: {src}.')

        e = _generate_or_load_errors(
            rec_mf=rec_mf,
            rec=rec,
            N=N,
            points_2d=points_2d[i],
            batch_size=batch_size,
            device=device,
            rebuild_cache=rebuild_cache,
            cache_only=cache_only,
        )
        errors.append(e)

    return errors


def _calculate_smoothness(
        rec_mf: Reconstruction,
        rec: Reconstruction,
) -> np.ndarray:
    """
    Compute smoothness from the 3D points.
    """
    X, _ = get_trajectory(
        reconstruction_id=rec.id,
        start_frame=max(rec.start_frame, rec_mf.start_frame_valid),
        end_frame=min(rec.end_frame, rec_mf.end_frame_valid)
    )

    # Centre trajectory
    X = X - X.mean(axis=0)

    # Calculate angles between segments
    v1 = X[:, 1:-1] - X[:, :-2]
    v2 = X[:, 2:] - X[:, 1:-1]
    abs_val = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    dot = np.einsum('bij,bij->bi', v1, v2)
    cos = dot / abs_val
    angles = np.arccos(cos)

    # Calculate gradient of the angles along the body
    angles_grad = np.gradient(np.unwrap(angles), axis=1)

    # Take the loss as the sum of the absolute gradients
    loss = np.abs(angles_grad).sum(axis=1)

    return loss


def _generate_or_load_smoothness(
        rec_mf: Reconstruction,
        rec: Reconstruction,
        N: int,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the smoothness losses.
    """
    cache_path = POINTS_CACHE_PATH / f'rec_{rec.id}_N={N}_smoothness'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    data = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            data = data['data']
            logger.info(f'Loaded smoothness losses from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Calculating smoothness losses.')
        data = _calculate_smoothness(
            rec_mf=rec_mf,
            rec=rec,
        )
        save_arrs = {'data': data}
        logger.info(f'Saving smoothness losses to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return data


def _fetch_smoothness(
        rec_mf: Reconstruction,
        recs_to_compare: Dict[str, Reconstruction],
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> List[np.ndarray]:
    """
    Generate or load the smoothness losses.
    """
    N = rec_mf.mf_parameters.n_points_total

    # Generate or load smoothness losses
    losses = []
    for i in range(1 + len(recs_to_compare)):
        if i == 0:
            rec = rec_mf
            logger.info(f'Calculating smoothness for MF reconstruction.')
        else:
            src = list(recs_to_compare.keys())[i - 1]
            rec = recs_to_compare[src]
            logger.info(f'Calculating smoothness for rec={rec.id}: {src}.')

        l = _generate_or_load_smoothness(
            rec_mf=rec_mf,
            rec=rec,
            N=N,
            rebuild_cache=rebuild_cache,
            cache_only=cache_only,
        )
        losses.append(l)

    return losses


def _rolling_stats(errors: List[np.ndarray], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rolling mean and standard deviations.
    """
    means = []
    stds = []

    for errs in errors:
        pl = np.ones(int((window_size - 1) / 2)) * errs[0]
        pr = np.ones(window_size - len(pl) - 1) * errs[-1]
        errs_padded = np.r_[pl, errs, pr]
        x = sliding_window_view(errs_padded, window_size)
        means.append(x.mean(axis=1))
        stds.append(x.std(axis=1))

    return means, stds


def plot_mf_comparisons(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
):
    """
    Plot the loss comparison between a MF reconstruction and any other available types.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec_mf: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec_mf.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    trial: Trial = rec_mf.trial
    start_frame = rec_mf.start_frame_valid
    end_frame = rec_mf.end_frame_valid
    frame_nums = np.arange(start_frame, end_frame + 1)
    device = _init_devices(args)
    recs_to_compare = _get_recs_to_compare(trial)

    # Generate or load the errors
    errors = _fetch_errors(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        batch_size=args.batch_size,
        device=device,
        rebuild_cache=args.rebuild_cache,
        cache_only=args.cache_only,
    )

    # Get moving averages
    means, stds = _rolling_stats(errors, args.stats_window)

    # Make plot
    fig, axes = plt.subplots(1, figsize=(10, 7))
    ax = axes
    ax.set_title(f'Pixel Losses\nTrial {trial.id}. Smoothing window: {args.stats_window} frames.')
    ax.set_ylabel('MSE')
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    for i in range(1 + len(recs_to_compare)):
        if i == 0:
            src = M3D_SOURCE_MF
            x = frame_nums
        else:
            src = list(recs_to_compare.keys())[i - 1]
            rec = recs_to_compare[src]
            x = np.arange(
                max(rec.start_frame, rec_mf.start_frame_valid),
                min(rec.end_frame, rec_mf.end_frame_valid) + 1
            )

        if args.x_label == 'time':
            x = x / trial.fps

        ax.plot(x, means[i], label=src, color=colours[src])
        ax.fill_between(x, means[i] - 2 * stds[i], means[i] + 2 * stds[i], color=colours[src],
                        alpha=0.2, linewidth=0)

    ax.set_xlim(left=start_frame, right=end_frame)
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    fig.tight_layout()

    if save_plots:
        if save_dir is None:
            path = LOGS_PATH / f'{START_TIMESTAMP}_losses' \
                               f'_trial={trial.id:03d}' \
                               f'_mf={rec_mf.id}' \
                               f'_comp={",".join([str(rec.id) for rec in recs_to_compare.values()])}' \
                               f'_sw={args.stats_window}' \
                               f'.{img_extension}'
        else:
            path = save_dir / f'trial={trial.id:03d}' \
                              f'_mf={rec_mf.id}' \
                              f'_comp={",".join([str(rec.id) for rec in recs_to_compare.values()])}' \
                              f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_examples(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
):
    """
    Plot some examples from the different reconstructions.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec_mf: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec_mf.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    trial: Trial = rec_mf.trial
    start_frame = rec_mf.start_frame_valid
    end_frame = rec_mf.end_frame_valid
    frame_nums = np.arange(start_frame, end_frame)
    recs_to_compare = _get_recs_to_compare(trial)
    n_recs = 1 + len(recs_to_compare)

    if args.plot_example_frames is None:
        # Find the frame numbers in common
        frame_nums_in_common = frame_nums
        for src, rec in recs_to_compare.items():
            frame_nums_rec = np.arange(rec.start_frame, rec.end_frame + 1)
            frame_nums_in_common = np.intersect1d(frame_nums_in_common, frame_nums_rec)

        # Select frames at random
        frame_nums_to_plot = sorted(np.random.choice(frame_nums_in_common, args.plot_n_examples, replace=False))
    else:
        frame_nums_to_plot = sorted(np.array(args.plot_example_frames))

    # Generate or load the 2D data
    points_2d = _fetch_2d_data(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        rebuild_cache=args.rebuild_cache,
        cache_only=args.cache_only,
    )

    # Plot the comparisons for each frame
    for frame_num in frame_nums_to_plot:
        frame = trial.get_frame(frame_num)
        fig, axes = plt.subplots(n_recs, figsize=(8, 3 * n_recs), gridspec_kw={
            'left': 0.05,
            'right': 0.95,
            'bottom': 0.05,
            'hspace': 0.1
        })
        fig.suptitle(f'Trial {trial.id}. Frame {frame_num}.')

        for i in range(n_recs):
            if i == 0:
                src = M3D_SOURCE_MF
                rec = rec_mf
            else:
                src = list(recs_to_compare.keys())[i - 1]
                rec = recs_to_compare[src]

            # Get the images with midline overlays
            idx = frame_num - (rec.start_frame_valid if rec.start_frame_valid is not None else rec.start_frame)
            if idx > len(points_2d[i]):
                continue
            imgs_i = _overlay_images(
                images=frame.images,
                points_2d=points_2d[i][idx],
                midline_width=3
            )

            # Plot
            ax = axes[i]
            ax.set_title(f'Source: {src} ({rec.id})')
            ax.imshow(np.array(imgs_i), aspect='equal')
            ax.axis('off')

        fig.tight_layout()

        if save_plots:
            if save_dir is None:
                trial_dir = LOGS_PATH / f'{START_TIMESTAMP}_examples' \
                                        f'_trial={trial.id:03d}' \
                                        f'_mf={rec_mf.id}'
                os.makedirs(trial_dir, exist_ok=True)
                path = trial_dir / f'frame={frame_num:05d}.{img_extension}'
            else:
                path = save_dir / f'trial={trial.id:03d}' \
                                  f'_mf={rec_mf.id}' \
                                  f'_frame={frame_num:05d}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()
        plt.close(fig)


def plot_smoothness_comparisons(
        args: Optional[Namespace] = None,
        save_dir: Optional[Path] = None
):
    """
    Plot the smoothness comparison between a MF reconstruction and any other available types.
    """
    if args is None:
        args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    rec_mf: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec_mf.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    trial: Trial = rec_mf.trial
    start_frame = rec_mf.start_frame_valid
    end_frame = rec_mf.end_frame_valid
    frame_nums = np.arange(start_frame, end_frame + 1)
    recs_to_compare = _get_recs_to_compare(trial)

    # Generate or load the errors
    losses = _fetch_smoothness(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        rebuild_cache=args.rebuild_cache,
        cache_only=args.cache_only,
    )

    # Get moving averages
    means, stds = _rolling_stats(losses, args.stats_window)

    # Make plot
    fig, axes = plt.subplots(1, figsize=(10, 7))
    ax = axes
    ax.set_title(f'Smoothness\nTrial {trial.id}. Smoothing window: {args.stats_window} frames.')
    ax.set_ylabel('Smoothness loss')
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    for i in range(1 + len(recs_to_compare)):
        if i == 0:
            src = M3D_SOURCE_MF
            x = frame_nums
        else:
            src = list(recs_to_compare.keys())[i - 1]
            rec = recs_to_compare[src]
            x = np.arange(
                max(rec.start_frame, rec_mf.start_frame_valid),
                min(rec.end_frame, rec_mf.end_frame_valid) + 1
            )

        if args.x_label == 'time':
            x = x / trial.fps

        ax.plot(x, means[i], label=src, color=colours[src])
        ax.fill_between(x, means[i] - 2 * stds[i], means[i] + 2 * stds[i], color=colours[src],
                        alpha=0.2, linewidth=0)

    ax.set_xlim(left=start_frame, right=end_frame)
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    fig.tight_layout()

    if save_plots:
        if save_dir is None:
            path = LOGS_PATH / f'{START_TIMESTAMP}_smoothness' \
                               f'_trial={trial.id:03d}' \
                               f'_mf={rec_mf.id}' \
                               f'_comp={",".join([str(rec.id) for rec in recs_to_compare.values()])}' \
                               f'_sw={args.stats_window}' \
                               f'.{img_extension}'
        else:
            path = save_dir / f'trial={trial.id:03d}' \
                              f'_mf={rec_mf.id}' \
                              f'_comp={",".join([str(rec.id) for rec in recs_to_compare.values()])}' \
                              f'_smoothness' \
                              f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_all_comparisons_in_dataset():
    """
    Generate comparison plots and examples for all reconstructions in a dataset.
    """
    args = get_args()
    assert args.dataset is not None, 'This script requires setting --dataset=id.'
    ds = Dataset.objects.get(id=args.dataset)
    assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'
    args.plot_example_frames = None

    # Make save dir and save spec
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_ds={ds.id}'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

        spec = dict(
            dataset=str(ds.id),
            batch_size=args.batch_size,
            x_label=args.x_label,
            stats_window=args.stats_window,
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for i, rec in enumerate(ds.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(ds.reconstructions)}.')
        args.reconstruction = rec.id
        try:
            plot_mf_comparisons(args, save_dir)
            plot_examples(args, save_dir)
            plot_smoothness_comparisons(args, save_dir)
        except Exception as e:
            logger.warning(f'Unable to plot comparisons: {e}')
            continue


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # plot_mf_comparisons()
    # plot_examples()
    # plot_smoothness_comparisons()
    plot_all_comparisons_in_dataset()
