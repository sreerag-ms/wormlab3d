import os
from argparse import Namespace, ArgumentParser
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import interpolate
from torch.backends import cudnn

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGES_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D
from wormlab3d.midlines3d.project_render_score import render_points
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args

POINTS_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(POINTS_CACHE_PATH, exist_ok=True)

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

    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size.')
    parser.add_argument('--gpu-id', type=int, default=-1, help='GPU id to use if using GPUs.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


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
) -> np.ndarray:
    """
    Calculate the r values across a range of sigmas, durations and pauses.
    """
    frame_nums = np.arange(rec.start_frame, rec.end_frame)
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
            X[j] = np.stack(m3d.get_prepared_2d_coordinates(), axis=1)
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
            logger.info(f'Loaded points data from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating points data.')
        data = _calculate_2d_data(rec, N)
        save_arrs = {'data': data}
        logger.info(f'Saving points data to {cache_fn}.')
        np.savez(cache_path, **save_arrs)

    return data


def _fetch_2d_data(
        reconstruction: Reconstruction,
        recs_to_compare: Dict[str, Reconstruction],
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> List[np.ndarray]:
    """
    Fetch the 2d data
    """
    N = reconstruction.mf_parameters.n_points_total

    # Fetch the MF data directly
    ts = TrialState(reconstruction, start_frame=reconstruction.start_frame_valid,
                    end_frame=reconstruction.end_frame_valid)
    Xs = [ts.get('points_2d'), ]

    # Load cached data for the comparisons
    for i, (src, rec) in enumerate(recs_to_compare.items()):
        X = _generate_or_load_2d_data(rec, N, rebuild_cache, cache_only)
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

    # Prepare sigmas, exponents and intensities
    N5 = int(N / 5)

    # Sigmas should be equal in the middle section but taper towards the ends
    sigmas = sigmas.clamp(min=sigmas_min)
    slopes = (sigmas - sigmas_min) / N5 * torch.arange(N5)[None, :] + sigmas_min
    sigmas = torch.cat([
        slopes,
        torch.ones(1, N - 2 * N5) * sigmas,
        slopes.flip(dims=(1,))
    ], dim=1)

    # Make exponents equal everywhere
    exponents = torch.ones(1, N) * exponents

    # Intensities should be equal in the middle section but taper towards the ends
    intensities = intensities.clamp(min=intensities_min)
    slopes = (intensities - intensities_min) / N5 \
             * torch.arange(N5)[None, :] + intensities_min
    intensities = torch.cat([
        slopes,
        torch.ones(1, N - 2 * N5) * intensities,
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
        ts: TrialState,
        rec: Reconstruction,
        points_2d: np.ndarray,
        batch_size: int,
) -> np.ndarray:
    """
    Render the 2D points and compute errors.
    """
    n_frames = len(points_2d)
    n_batches = int(n_frames / batch_size) + 1
    errors = np.zeros(n_frames)

    # Rendering parameters come from the MF reconstruction
    sigmas = ts.get('sigmas')
    intensities = ts.get('intensities')
    exponents = ts.get('exponents')
    camera_sigmas = ts.get('camera_sigmas')
    camera_intensities = ts.get('camera_intensities')
    camera_exponents = ts.get('camera_exponents')

    for i in range(n_batches):
        logger.info(f'Calculating errors for batch {i + 1}/{n_batches}.')
        start_idx = i * batch_size
        end_idx = min(n_frames, (i + 1) * batch_size)
        start_frame = rec.start_frame + start_idx
        end_frame = rec.start_frame + end_idx
        renders = _make_renders(
            points_2d=torch.from_numpy(points_2d[start_idx:end_idx]),
            sigmas=torch.from_numpy(sigmas[start_frame:end_frame]),
            sigmas_min=ts.parameters.sigmas_min,
            exponents=torch.from_numpy(exponents[start_frame:end_frame]),
            intensities=torch.from_numpy(intensities[start_frame:end_frame]),
            intensities_min=ts.parameters.intensities_min,
            camera_sigmas=torch.from_numpy(camera_sigmas[start_frame:end_frame]),
            camera_exponents=torch.from_numpy(camera_exponents[start_frame:end_frame]),
            camera_intensities=torch.from_numpy(camera_intensities[start_frame:end_frame]),
            image_size=ts.trial.crop_size
        )

        # Get targets
        images = np.stack([
            np.load(PREPARED_IMAGES_PATH / f'{ts.trial.id:03d}' / f'{n:06d}.npz')['images']
            for n in range(start_frame, end_frame)
        ])

        # MSE
        errors[start_idx:end_idx] = ((renders - images)**2).mean(axis=(1, 2, 3))

    return errors


def _generate_or_load_errors(
        ts: TrialState,
        rec: Reconstruction,
        N: int,
        points_2d: np.ndarray,
        batch_size: int,
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
            logger.info(f'Loaded errors from cache: {cache_fn}')
        except Exception as e:
            data = None
            logger.warning(f'Could not load cache: {e}')

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Calculating errors.')
        data = _calculate_errors(
            ts=ts,
            rec=rec,
            points_2d=points_2d,
            batch_size=batch_size
        )
        save_arrs = {'data': data}
        logger.info(f'Saving errors data to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return data


def _fetch_errors(
        rec_mf: Reconstruction,
        recs_to_compare: Dict[str, Reconstruction],
        batch_size: int,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> List[np.ndarray]:
    """
    Generate or load the errors.
    """
    N = rec_mf.mf_parameters.n_points_total
    start_frame = rec_mf.start_frame_valid
    end_frame = rec_mf.end_frame_valid
    ts = TrialState(rec_mf)

    # Generate or load the 2D data
    points_2d_mf = ts.get('points_2d', start_frame=start_frame, end_frame=end_frame)
    points_2d_to_compare = _fetch_2d_data(
        reconstruction=rec_mf,
        recs_to_compare=recs_to_compare,
        rebuild_cache=rebuild_cache,
        cache_only=cache_only,
    )
    points_2d = [points_2d_mf, *points_2d_to_compare]

    # Generate or load pixel-losses
    errors = []
    for i in range(1 + len(recs_to_compare)):
        if i == 0:
            rec = rec_mf
            logger.info(f'Calculating pixel errors for MF reconstruction.')
        else:
            src = list(recs_to_compare.keys())[i + 1]
            rec = recs_to_compare[src]
            logger.info(f'Calculating pixel errors for rec={rec.id}: {src}.')

        e = _generate_or_load_errors(
            ts,
            rec=rec,
            N=N,
            points_2d=points_2d[i],
            batch_size=batch_size,
            rebuild_cache=rebuild_cache,
            cache_only=cache_only,
        )
        errors.append(e)

    return errors


def plot_mf_comparisons():
    """
    Plot the loss comparison between a MF reconstruction and any other available types.
    """
    args = get_args()
    rec_mf: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec_mf.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    trial: Trial = rec_mf.trial
    start_frame = rec_mf.start_frame_valid
    end_frame = rec_mf.end_frame_valid
    frame_nums = np.arange(start_frame, end_frame)
    ts = TrialState(rec_mf)

    # Fetch reconstructions to compare against, max one from each source
    recs = Reconstruction.objects(trial=trial, source__ne=M3D_SOURCE_MF)
    n_results = recs.count()
    if n_results == 0:
        raise RuntimeError('No reconstructions found to compare against!')
    recs_to_compare = {}
    frame_nums_to_compare = {}
    for rec in recs:
        if rec.source not in recs_to_compare \
                or (
                rec.source in recs_to_compare and len(rec.source_file) < len(recs_to_compare[rec.source].source_file)):
            recs_to_compare[rec.source] = rec
            frame_nums_to_compare[rec.source] = np.arange(rec.start_frame, rec.end_frame)
    sources_to_compare = list(recs_to_compare.keys())

    # Generate or load the errors
    errors = _fetch_errors(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        batch_size=args.batch_size,
        rebuild_cache=False,
        cache_only=False,
    )

    # Make plot
    fig, axes = plt.subplots(1, figsize=(10, 12))
    ax = axes
    ax.title(f'Pixel Losses: Trial {trial.id}')
    ax.set_ylabel('MSE')
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    for i in range(1 + len(sources_to_compare)):
        if i == 0:
            src = 'MF'
            x = frame_nums
        else:
            src = sources_to_compare[i - 1]
            x = frame_nums_to_compare[src]

        if args.x_label == 'time':
            x /= ts.trial.fps

        ax.plot(x, errors[i], label=src)
    ax.legend()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_losses' \
                           f'_trial={trial.id}' \
                           f'_mf={rec_mf.id}' \
                           f'_comp={",".join([rec.id for rec in recs_to_compare.values()])}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    plot_mf_comparisons()
