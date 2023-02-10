import gc
import os
from argparse import Namespace, ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from torch.backends import cudnn

from wormlab3d import logger, LOGS_PATH, PREPARED_IMAGES_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial
from wormlab3d.midlines3d.project_render_score import render_points
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args, to_numpy, str2bool, to_dict

ERRORS_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(ERRORS_CACHE_PATH, exist_ok=True)
ABLATION_KEYS = ['good', 'no-cams', 'no-renders', 'no-cs', 'no-lpx', 'no-lsc', 'no-masks', 'no-regs']
ABLATION_KEYS_US = [k.replace('-', '_') for k in ABLATION_KEYS]

prop_cycle = plt.rcParams['axes.prop_cycle']
default_colours = prop_cycle.by_key()['color']
colours = {k: default_colours[i] for i, k in enumerate(ABLATION_KEYS_US)}

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to compare MF losses against ablations.')

    # Reconstructions for comparison
    for k in ABLATION_KEYS:
        parser.add_argument(
            f'--rec-{k}', type=str, required=k == 'good',
            help='Good (reference) reconstruction by id.' if k == 'good' else f'Reconstruction with {k} ablation.'
        )

    # Frame range
    parser.add_argument('--start-frame', type=int, required=True, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, required=True, help='Frame number to end at.')

    # Processing arguments
    parser.add_argument('--rebuild-cache', type=str2bool, default=False, help='Rebuild caches.')
    parser.add_argument('--cache-only', type=str2bool, default=False, help='Use cache only.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size.')
    parser.add_argument('--gpu-id', type=int, default=-1, help='GPU id to use if using GPUs.')

    # Plotting arguments
    parser.add_argument('--x-label', type=str, default='frame', help='Label x-axis with time or frame number.')
    parser.add_argument('--stats-window', type=int, default=5, help='Averaging window for the stats.')

    args = parser.parse_args()
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
        rec: Reconstruction,
        start_frame: int,
        end_frame: int,
        batch_size: int,
        device: torch.device
) -> np.ndarray:
    """
    Render the 2D points and compute errors.
    """
    ts = TrialState(
        rec,
        start_frame=start_frame,
        end_frame=min(end_frame, rec.end_frame - 1)
    )
    n_frames = len(ts)
    n_batches = int(n_frames / batch_size) + 1
    errors = np.zeros(n_frames)

    # Rendering parameters come from the MF reconstruction
    points_2d = ts.get('points_2d')
    sigmas = ts.get('sigmas')
    intensities = ts.get('intensities')
    exponents = ts.get('exponents')
    camera_sigmas = ts.get('camera_sigmas')
    camera_intensities = ts.get('camera_intensities')
    camera_exponents = ts.get('camera_exponents')

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(n_frames + 1, (i + 1) * batch_size)
        if end_idx == start_idx or start_idx == len(points_2d):
            continue

        batch_params = {
            'points_2d': torch.from_numpy(points_2d[start_idx:end_idx].copy()).to(device),
            'sigmas': torch.from_numpy(sigmas[start_idx:end_idx].copy()).to(device),
            'exponents': torch.from_numpy(exponents[start_idx:end_idx].copy()).to(device),
            'intensities': torch.from_numpy(intensities[start_idx:end_idx].copy()).to(device),
            'camera_sigmas': torch.from_numpy(camera_sigmas[start_idx:end_idx].copy()).to(device),
            'camera_exponents': torch.from_numpy(camera_exponents[start_idx:end_idx].copy()).to(device),
            'camera_intensities': torch.from_numpy(camera_intensities[start_idx:end_idx].copy()).to(device),
        }
        bs = batch_params['points_2d'].shape[0]
        logger.info(f'Calculating errors for batch {i + 1}/{n_batches} (batch size = {bs}).')

        # Pad the last batch so the batch size is the same
        if bs != batch_size:
            n_pad = batch_size - bs
            logger.info(f'Padding batch with {n_pad} vals.')
            for k, v in batch_params.items():
                batch_params[k] = torch.cat([
                    batch_params[k],
                    torch.zeros((n_pad, *batch_params[k].shape[1:]), device=device)
                ])

        renders = _make_renders(
            sigmas_min=ts.parameters.sigmas_min,
            intensities_min=ts.parameters.intensities_min,
            image_size=ts.trial.crop_size,
            **batch_params
        )

        if bs != batch_size:
            renders = renders[:bs]

        # Get targets
        start_frame = ts.start_frame + start_idx
        end_frame = start_frame + len(renders)
        images = torch.from_numpy(np.stack([
            np.load(PREPARED_IMAGES_PATH / f'{ts.trial.id:03d}' / f'{n:06d}.npz')['images']
            for n in range(start_frame, end_frame)
        ])).to(device)

        # MSE
        errors[start_idx:end_idx] = to_numpy(((renders - images)**2).mean(axis=(1, 2, 3)))

        renders = None
        del renders
        gc.collect()

    return errors


def _generate_or_load_errors(
        rec: Reconstruction,
        start_frame: int,
        end_frame: int,
        batch_size: int,
        device: torch.device,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the errors.
    """
    cache_path = ERRORS_CACHE_PATH / f'rec_{rec.id}_frames={start_frame}-{end_frame}_errors'
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
            rec=rec,
            start_frame=start_frame,
            end_frame=end_frame,
            batch_size=batch_size,
            device=device
        )
        save_arrs = {'data': data}
        logger.info(f'Saving errors data to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return data


def _rolling_stats(errors: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rolling mean and standard deviations.
    """
    pl = np.ones(int((window_size - 1) / 2)) * errors[0]
    pr = np.ones(window_size - len(pl) - 1) * errors[-1]
    errs_padded = np.r_[pl, errors, pr]
    x = sliding_window_view(errs_padded, window_size)
    return x.mean(axis=1), x.std(axis=1)


def plot_comparisons(
        make_plot: bool = True
):
    """
    Plot the loss comparison between a MF reconstruction and its ablations.
    """
    args = get_args()
    device = _init_devices(args)
    frame_nums = np.arange(args.start_frame, args.end_frame + 1)

    # Fetch errors for each ablation
    trial_ids = []
    recs = {}
    errors = {}
    means = {}
    stds = {}
    for i, k in enumerate(ABLATION_KEYS_US):
        if not hasattr(args, f'rec_{k}'):
            continue
        rec = Reconstruction.objects.get(id=getattr(args, f'rec_{k}'))
        recs[k] = rec
        trial_ids.append(rec.trial.id)
        logger.info(f'Fetching errors for key {i + 1}/{len(ABLATION_KEYS)}: "{k}" ({rec.id}).')

        # Generate or load the errors
        errors[k] = _generate_or_load_errors(
            rec=rec,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            batch_size=args.batch_size,
            device=device,
            rebuild_cache=args.rebuild_cache,
            cache_only=args.cache_only,
        )

        # Get moving averages
        means[k], stds[k] = _rolling_stats(errors[k], args.stats_window)

    # Check they all apply to the same trial
    assert all([tid == trial_ids[0] for tid in trial_ids]), 'All reconstructions should be for the same trial!'
    trial = Trial.objects.get(id=trial_ids[0])

    # Calculate errors relative to the good version
    rel_errors = {}
    for k, errs in errors.items():
        # rel_errors[k] = (errors['good'][:len(errs)] - errs) / errors['good'][:len(errs)] * 100
        rel_errors[k] = errs / errors['good'][:len(errs)]

    # Print summary results
    log_str = '\n\nSummary:\n\n'
    for k, rec in recs.items():
        log_str += f'{k:>13} ({len(errors[k]):04d}) = {errors[k].mean():.6f} ({errors[k].std():.6f}) ' \
                   f'\trel:\t {rel_errors[k].mean():.3f} ({rel_errors[k].std():.3f})\n'
    log_str += f'\nTotal frames = {len(errors["good"])}\n'
    logger.info(log_str)

    # Make plot
    if make_plot:
        fig, axes = plt.subplots(1, figsize=(10, 7))
        ax = axes
        ax.set_title(f'Pixel Losses\nTrial {trial.id}. Smoothing window: {args.stats_window} frames.')
        ax.set_ylabel('MSE')
        if args.x_label == 'time':
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xlabel('Frame #')

        for k, rec in recs.items():
            x = frame_nums[:len(means[k])]
            if args.x_label == 'time':
                x = x / trial.fps
            ax.plot(x, means[k], label=k, color=colours[k])
            ax.fill_between(x, means[k] - 2 * stds[k], means[k] + 2 * stds[k], color=colours[k],
                            alpha=0.2, linewidth=0)

        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        fig.tight_layout()

        if save_plots:
            path = LOGS_PATH / f'{START_TIMESTAMP}_losses' \
                               f'_trial={trial.id:03d}' \
                               f'_frames={args.start_frame}-{args.start_frame}' \
                               f'_sw={args.stats_window}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path.with_suffix(f'.{img_extension}'), transparent=True)

            # Write meta data
            meta = to_dict(args)
            meta['date'] = START_TIMESTAMP

            # Add results
            res = {}
            for k, rec in recs.items():
                res[k] = {
                    'n_frames': len(errors[k]),
                    'error_mean': float(errors[k].mean()),
                    'error_std': float(errors[k].std()),
                    'rel_mean': float(rel_errors[k].mean()),
                    'rel_std': float(rel_errors[k].std()),
                }
            meta['results'] = res
            with open(path.with_suffix('.yml'), 'w') as f:
                yaml.dump(meta, f)
        if show_plots:
            plt.show()


def count_frames():
    """
    Count the numbers of frames in each reconstruction.
    """
    args = get_args()
    log_str = '\n\nNumbers of frames:\n\n'
    for i, k in enumerate(ABLATION_KEYS_US):
        if not hasattr(args, f'rec_{k}'):
            continue
        rec = Reconstruction.objects.get(id=getattr(args, f'rec_{k}'))
        log_str += f'{rec.id} {k:>13} {rec.start_frame}-{rec.end_frame - 1} ({rec.n_frames})\n'
    logger.info(log_str)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_comparisons(make_plot=True)
    # count_frames()
