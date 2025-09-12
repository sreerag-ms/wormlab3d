import gc
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
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate
from torch import nn
from torch.backends import cudnn

from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGES_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial, Dataset
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D, M3D_SOURCE_WT3D, M3D_SOURCE_RECONST
from wormlab3d.midlines3d.project_render_score import render_points
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args, to_numpy, str2bool, is_bad
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
colours = {
    M3D_SOURCE_MF: '#2196F3',
    M3D_SOURCE_RECONST: 'orange',
    M3D_SOURCE_WT3D: '#4CAF50',
    'highlight': '#F44336'
}
colours_opt = {
    M3D_SOURCE_RECONST: 'red',
    M3D_SOURCE_WT3D: 'lime',
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
    parser = ArgumentParser(description='Wormlab3D script to compare MF losses against reconst/WT3D.')

    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')

    # Processing args
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size.')
    parser.add_argument('--gpu-id', type=int, default=-1, help='GPU id to use if using GPUs.')
    parser.add_argument('--rebuild-cache', type=str2bool, default=False, help='Rebuild caches.')
    parser.add_argument('--cache-only', type=str2bool, default=False, help='Use cache only.')

    # Optimisation args
    parser.add_argument('--lr', type=float, help='Learning rate.')
    parser.add_argument('--max-train-steps', type=int, default=0, help='Maximum training steps.')
    parser.add_argument('--conv-tol', type=float, default=0.1, help='Convergence relative tolerance.')
    parser.add_argument('--conv-patience', type=int, default=5,
                        help='Convergence patience (number of train steps to wait).')

    # Plot args
    parser.add_argument('--x-label', type=str, default='frame', help='Label x-axis with time or frame number.')
    parser.add_argument('--stats-window', type=int, default=5, help='Averaging window for the stats.')
    parser.add_argument('--plot-n-examples', type=int, default=3, help='Number of examples to plot.')
    parser.add_argument('--plot-example-frames', type=lambda s: [int(item) for item in s.split(',')], default=[],
                        help='Plot these frame numbers.')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process.')
    parser.add_argument('--reconstruction-b', type=str, help='Second MF reconstruction by id.')

    args = parser.parse_args()
    print_args(args)

    return args


def _tex_mode():
    """Use latex font rendering."""
    plt.rcParams.update({'text.usetex': True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


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

    # Only keep the M3D_SOURCE_WT3D key if present
    # recs_to_compare = {k: v for k, v in recs_to_compare.items() if k == M3D_SOURCE_WT3D}
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


def _calculate_2d_data(
        rec: Reconstruction,
        N: int,
        rebuild_cache: bool = False,
        force_resample: bool = False
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
        if len(m3d.X) == N and not force_resample:
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
    ts = TrialState(rec_mf, start_frame=rec_mf.start_frame,
                    end_frame=rec_mf.end_frame)
    Xs = [ts.get('points_2d'), ]

    # Load cached data for the comparisons
    for i, (src, rec) in enumerate(recs_to_compare.items()):
        X = _generate_or_load_2d_data(rec, N, rebuild_cache, cache_only)
        if rec.start_frame < rec_mf.start_frame:
            X = X[rec_mf.start_frame - rec.start_frame:]
        if rec.end_frame > rec_mf.end_frame:
            X = X[:rec_mf.end_frame - rec.end_frame]
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


def _optimise_parameters(
        points_2d: torch.Tensor,
        sigmas: torch.Tensor,
        sigmas_min: float,
        exponents: torch.Tensor,
        intensities: torch.Tensor,
        intensities_min: float,
        camera_sigmas: torch.Tensor,
        camera_exponents: torch.Tensor,
        camera_intensities: torch.Tensor,
        image_size: int,
        images: torch.Tensor,
        lr: float,
        max_train_steps: int,
        conv_tol: float,
        conv_patience: int,
):
    """
    Optimise the parameters.
    """
    sigmas = nn.Parameter(sigmas.detach().clone(), requires_grad=True)
    exponents = nn.Parameter(exponents.detach().clone(), requires_grad=True)
    intensities = nn.Parameter(intensities.detach().clone(), requires_grad=True)
    camera_sigmas = nn.Parameter(camera_sigmas.detach().clone(), requires_grad=True)
    camera_exponents = nn.Parameter(camera_exponents.detach().clone(), requires_grad=True)
    camera_intensities = nn.Parameter(camera_intensities.detach().clone(), requires_grad=True)
    params = {
        'sigmas': sigmas,
        'exponents': exponents,
        'intensities': intensities,
        'camera_sigmas': camera_sigmas,
        'camera_exponents': camera_exponents,
        'camera_intensities': camera_intensities,
    }
    if max_train_steps == 0:
        return params


    actual_lr = lr
    
    optimiser = torch.optim.AdamW(params=list(params.values()), lr=actual_lr, weight_decay=0)

    def _train_loop():
        optimiser.zero_grad()
        renders = _make_renders(
            points_2d=points_2d,
            sigmas=sigmas,
            sigmas_min=sigmas_min,
            exponents=exponents,
            intensities=intensities,
            intensities_min=intensities_min,
            camera_sigmas=camera_sigmas,
            camera_exponents=camera_exponents,
            camera_intensities=camera_intensities,
            image_size=image_size
        )
        loss = ((renders - images)**2).mean()
        loss.backward()
        if is_bad(loss):
            logger.warning(f"Encountered bad loss value: {loss.item()}")
            # Return the previous loss instead
            return float('inf')
        
        optimiser.step()
        return loss.item()

    l_prev = np.inf
    conv_count = 0
    for i in range(max_train_steps):
        try:
            l = _train_loop()
            if l == float('inf'):
                logger.warning(f"Skipping step {i+1} due to bad loss")
                continue
                
            if i % 1 == 0:
                logger.info(f'Optimisation step {i + 1}/{max_train_steps}: Loss = {l:.5E} \t (cc={conv_count})')
            is_converged = abs(l - l_prev) / l_prev < conv_tol
            if is_converged:
                conv_count += 1
                if conv_count > conv_patience:
                    logger.info(f'Converged after {i + 1} steps. Loss = {l:.5E}')
                    break
            else:
                conv_count = 0
            l_prev = l
        except Exception as e:
            logger.warning(f"Error during optimization step {i+1}: {str(e)}")
            break

    params = {k: v.clone().detach() for k, v in params.items()}
    gc.collect()

    return params


def _calculate_errors(
        rec_mf: Reconstruction,
        rec: Reconstruction,
        points_2d: np.ndarray,
        batch_size: int,
        device: torch.device,
        lr: float,
        max_train_steps: int,
        conv_tol: float,
        conv_patience: int,
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
        start_frame=max(rec.start_frame, rec_mf.start_frame),
        end_frame=min(rec.end_frame, rec_mf.end_frame)
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
        end_idx = min(n_frames, (i + 1) * batch_size)
        if end_idx <= start_idx:
            continue
        p2d_batch = torch.from_numpy(points_2d[start_idx:end_idx]).to(device)

        # Get targets
        start_frame = rec.start_frame + start_idx
        end_frame = start_frame + len(p2d_batch)
        images = torch.from_numpy(np.stack([
            np.load(PREPARED_IMAGES_PATH / f'{ts.trial.id:03d}' / f'{n:06d}.npz')['images']
            for n in range(start_frame, end_frame)
        ])).to(device)

        parameters = _optimise_parameters(
            points_2d=p2d_batch,
            sigmas=torch.from_numpy(sigmas[start_idx:end_idx]).to(device),
            sigmas_min=ts.parameters.sigmas_min,
            exponents=torch.from_numpy(exponents[start_idx:end_idx]).to(device),
            intensities=torch.from_numpy(intensities[start_idx:end_idx]).to(device),
            intensities_min=ts.parameters.intensities_min,
            camera_sigmas=torch.from_numpy(camera_sigmas[start_idx:end_idx]).to(device),
            camera_exponents=torch.from_numpy(camera_exponents[start_idx:end_idx]).to(device),
            camera_intensities=torch.from_numpy(camera_intensities[start_idx:end_idx]).to(device),
            image_size=ts.trial.crop_size,
            images=images,
            lr=lr,
            max_train_steps=max_train_steps,
            conv_tol=conv_tol,
            conv_patience=conv_patience,
        )
        renders = _make_renders(
            **parameters,
            points_2d=p2d_batch,
            sigmas_min=ts.parameters.sigmas_min,
            intensities_min=ts.parameters.intensities_min,
            image_size=ts.trial.crop_size
        )

        # MSE
        errors[start_idx:end_idx] = to_numpy(((renders - images)**2).mean(axis=(1, 2, 3)))

        parameters = None
        del parameters
        renders = None
        del renders
        gc.collect()

    return errors


def _generate_or_load_errors(
        rec_mf: Reconstruction,
        rec: Reconstruction,
        N: int,
        points_2d: np.ndarray,
        batch_size: int,
        device: torch.device,
        lr: float,
        max_train_steps: int,
        conv_tol: float,
        conv_patience: int,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the errors.
    """
    cache_id = f'rec_{rec.id}_N={N}_errors'
    if max_train_steps > 0 and rec.source != M3D_SOURCE_MF:
        cache_id += f'_lr={lr:.2E}_mts={max_train_steps}_ctol={conv_tol:.3f}_cpat={conv_patience}'
    cache_path = POINTS_CACHE_PATH / cache_id
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
            device=device,
            lr=lr,
            max_train_steps=max_train_steps,
            conv_tol=conv_tol,
            conv_patience=conv_patience,
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
        lr: float,
        max_train_steps: int,
        conv_tol: float,
        conv_patience: int,
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
            lr=lr,
            max_train_steps=max_train_steps,
            conv_tol=conv_tol,
            conv_patience=conv_patience,
            rebuild_cache=rebuild_cache,
            cache_only=cache_only,
        )
        errors.append(e)

    return errors




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

def plot_pixel_losses_mf_pair():
    """
    Plot the pixel losses for TWO MF reconstructions on the same trial.
    """
    args = get_args()
    assert args.reconstruction is not None, 'Set --reconstruction=<id> for the first MF reconstruction.'
    assert args.reconstruction_b is not None, 'Set --reconstruction-b=<id> for the second MF reconstruction.'

    rec_a: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    rec_b: Reconstruction = Reconstruction.objects.get(id=args.reconstruction_b)

    # sanity checks
    assert rec_a.source == M3D_SOURCE_MF, 'First reconstruction must be MF.'
    assert rec_b.source == M3D_SOURCE_MF, 'Second reconstruction must be MF.'
    assert rec_a.trial.id == rec_b.trial.id, 'Both reconstructions must belong to the same trial.'

    device = _init_devices(args)
    trial: Trial = rec_a.trial

    # Fetch raw pixel errors for each MF reconstruction independently
    common_fetch = dict(
        batch_size=args.batch_size,
        device=device,
        rebuild_cache=args.rebuild_cache,
        cache_only=args.cache_only,
    )

    # MF A
    lp_a_list = _fetch_errors(
        rec_mf=rec_a,
        recs_to_compare={},
        lr=0,
        max_train_steps=0,
        conv_tol=0,
        conv_patience=0,
        **common_fetch,
    )
    lp_a = lp_a_list[0] 

    # MF B
    lp_b_list = _fetch_errors(
        rec_mf=rec_b,
        recs_to_compare={},
        lr=0,
        max_train_steps=0,
        conv_tol=0,
        conv_patience=0,
        **common_fetch,
    )
    lp_b = lp_b_list[0]

    (means, stds) = _rolling_stats([lp_a, lp_b], args.stats_window)
    mean_a, mean_b = means[0], means[1]

    if args.max_frames is not None:
        fa = np.arange(rec_a.start_frame, min(rec_a.start_frame + args.max_frames, rec_a.end_frame + 1))
        fb = np.arange(rec_b.start_frame, min(rec_b.start_frame + args.max_frames, rec_b.end_frame + 1))
    else:
        fa = np.arange(rec_a.start_frame, rec_a.end_frame + 1)
        fb = np.arange(rec_b.start_frame, rec_b.end_frame + 1)

    mean_a = mean_a[:len(fa)]
    mean_b = mean_b[:len(fb)]
    if args.x_label == 'time':
        xa = fa / trial.fps
        xb = fb / trial.fps
        x_label = 'Time (s)'
    else:
        xa = fa
        xb = fb
        x_label = 'Frame #'

    plt.figure(figsize=(12, 8))
    
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    col_a = colours[M3D_SOURCE_MF]
    col_b = colours['highlight']

    plt.plot(xa, mean_a, label="MF (With HT)", color=col_a, linewidth=3, alpha=0.9)
    plt.plot(xb, mean_b, label="MF (Without HT)", color=col_b, linewidth=3, alpha=0.9)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(r'Pixel Loss ($L_{px}$)', fontsize=14)
    plt.title('Pixel Loss Comparison: MF Methods\n(With vs Without Head-Tail Constraints)', 
              fontsize=16, pad=20)

    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.ylim(bottom=1e-4, top=1e-1)
    plt.yticks([1e-4, 1e-3, 1e-2, 1e-1], fontsize=12)

    if args.x_label == 'time':
        xmin = min(xa[0], xb[0])
        xmax = max(xa[-1], xb[-1])
    else:
        xmin = min(fa[0], fb[0])
        xmax = max(fa[-1], fb[-1])
    plt.xlim(left=xmin, right=xmax)

    # X-ticks formatting
    if args.x_label != 'time':
        start = xmin
        end = xmax
        # choose a tick step that makes ~10 ticks
        n_ticks = 10
        step = max(1, int(np.ceil((end - start) / n_ticks)))
        plt.xticks(np.arange(start, end + 1, step), fontsize=12)
    else:
        plt.xticks(fontsize=12)

    plt.legend(fontsize=12, framealpha=0.9, loc='upper right')
    
    plt.tight_layout()

    if save_plots:
        fn = (
            f'trial={trial.id:03d}'
            f'_mfA={rec_a.id}'
            f'_mfB={rec_b.id}'
            f'_sw={args.stats_window}'
        )
        path = LOGS_PATH / f'{START_TIMESTAMP}_losses_mf_pair_{fn}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    if show_plots:
        plt.show()


def plot_pixel_losses_opt_comparison():
    """
    Plot the pixel losses with and without optimisations
    Can also plot a second MF reconstruction if reconstruction_b is provided
    """
    args = get_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'
    
    if args.reconstruction_b is None:
        assert args.max_train_steps > 0, '--max-train-steps must be > 0 to find a comparison when not using reconstruction_b.'
    
    rec_mf: Reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    assert rec_mf.source == M3D_SOURCE_MF, 'A MF reconstruction is required!'
    
    rec_mf_b = None
    if args.reconstruction_b is not None:
        rec_mf_b = Reconstruction.objects.get(id=args.reconstruction_b)
        assert rec_mf_b.source == M3D_SOURCE_MF, 'Second reconstruction must be MF.'
        assert rec_mf.trial.id == rec_mf_b.trial.id, 'Both reconstructions must belong to the same trial.'
    
    device = _init_devices(args)
    trial: Trial = rec_mf.trial
    start_frame = rec_mf.start_frame
    if args.max_frames is not None:
        end_frame = min(rec_mf.start_frame + args.max_frames, rec_mf.end_frame)
    else:
        end_frame = rec_mf.end_frame
    frame_nums = np.arange(start_frame, end_frame + 1)
    recs_to_compare = _get_recs_to_compare(trial)

    # Generate or load the errors with and without optimisations
    common_args = dict(
        rec_mf=rec_mf,
        recs_to_compare=recs_to_compare,
        batch_size=args.batch_size,
        device=device,
        rebuild_cache=args.rebuild_cache,
        cache_only=args.cache_only,
    )
    lp_raw = _fetch_errors(
         **common_args,
        lr=0,
        max_train_steps=0,
        conv_tol=0,
        conv_patience=0,
    )
    
    lp_raw_b = None
    if rec_mf_b is not None:
        common_args_b = dict(
            rec_mf=rec_mf_b,
            recs_to_compare={},
            batch_size=args.batch_size,
            device=device,
            rebuild_cache=args.rebuild_cache,
            cache_only=args.cache_only,
        )
        lp_raw_b = _fetch_errors(
            **common_args_b,
            lr=0,
            max_train_steps=0,
            conv_tol=0,
            conv_patience=0,
        )
    
    # lp_opt = _fetch_errors(
    #     **common_args,
    #     lr=args.lr,
    #     max_train_steps=args.max_train_steps,
    #     conv_tol=args.conv_tol,
    #     conv_patience=args.conv_patience,
    # )

    lpr_means, lpr_stds = _rolling_stats(lp_raw, args.stats_window)
    
    lpr_means_b = None
    if lp_raw_b is not None:
        lpr_means_b, lpr_stds_b = _rolling_stats(lp_raw_b, args.stats_window)

    plt.figure(figsize=(12, 8))
    
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    def _make_plot(means_raw: np.ndarray, means_opt: np.ndarray):
        # Plot primary MF reconstruction and comparison sources
        for i in range(1 + len(recs_to_compare)):
            if i == 0:
                src = M3D_SOURCE_MF
                lbl = src
                x = frame_nums
                data_raw = means_raw[i][:len(frame_nums)]
            else:
                src = list(recs_to_compare.keys())[i - 1]
                lbl = src
                rec = recs_to_compare[src]
                x = np.arange(
                    max(rec.start_frame, start_frame),
                    min(rec.end_frame, end_frame) + 1
                )
                data_raw = means_raw[i][:len(x)]
            if args.x_label == 'time':
                x = x / trial.fps

            plt.plot(x, data_raw, label=lbl, color=colours[src], linewidth=3, alpha=0.9)
        
        if rec_mf_b is not None and lpr_means_b is not None:
            start_frame_b = rec_mf_b.start_frame
            if args.max_frames is not None:
                end_frame_b = min(rec_mf_b.start_frame + args.max_frames, rec_mf_b.end_frame)
            else:
                end_frame_b = rec_mf_b.end_frame
            frame_nums_b = np.arange(start_frame_b, end_frame_b + 1)
            x_b = frame_nums_b
            if args.x_label == 'time':
                x_b = x_b / trial.fps
            
            data_raw_b = lpr_means_b[0][:len(frame_nums_b)]
            plt.plot(x_b, data_raw_b, label='MFHT', color=colours['highlight'], linewidth=3, alpha=0.9)

    _make_plot(lpr_means, lpr_means)
    
    if args.x_label == 'time':
        x_label = 'Time (s)'
    else:
        x_label = 'Frame Number'
    
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(r'Pixel Loss ($\mathcal{L}_{px}$)', fontsize=14)
    plt.title('Pixel Loss Comparison: MFHT vs State of the Art Methods', 
              fontsize=16, pad=20)
    
    plt.xlim(left=start_frame, right=end_frame)
    plt.xticks(np.linspace(start_frame, end_frame, min(8, end_frame - start_frame + 1), dtype=int), fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(bottom=1e-3, top=1e-1)
    plt.yticks([1e-3, 1e-2, 1e-1], fontsize=12)
    
    plt.legend(fontsize=12, framealpha=0.9, loc='upper right')
    
    plt.tight_layout()

    if save_plots:
        fn = f'trial={trial.id:03d}' \
             f'_mf={rec_mf.id}'
        if rec_mf_b is not None:
            fn += f'_mfB={rec_mf_b.id}'
        fn += f'_comp={",".join([str(rec.id) for rec in recs_to_compare.values()])}' \
             f'_sw={args.stats_window}'
        if args.max_train_steps > 0:
            fn += f'_lr={args.lr:.2E}_mts={args.max_train_steps}_ctol={args.conv_tol:.3f}_cpat={args.conv_patience}'
        path = LOGS_PATH / f'{START_TIMESTAMP}_losses_opt_comparison_{fn}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    plot_pixel_losses_opt_comparison()
    # plot_pixel_losses_mf_pair()
