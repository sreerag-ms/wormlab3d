import json
import os
from argparse import Namespace, ArgumentParser
from pathlib import PosixPath
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from wormlab3d import PREPARED_IMAGES_PATH, logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial
from wormlab3d.midlines3d.frame_state import CAM_PARAMETER_NAMES
from wormlab3d.midlines3d.mf_methods import integrate_curvature, smooth_parameter, make_rotation_matrix, normalise, \
    orthogonalise
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d
from wormlab3d.toolkit.util import print_args, to_dict, str2bool, to_numpy
from wormlab3d.trajectories.cache import get_trajectory

show_plots = False
save_plots = True
img_extension = 'png'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to fix MF output.')

    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--dry-run', type=str2bool, default=True,
                        help='Don\'t update any parameters, just generate some output.')

    # -- Which fixes to run
    parser.add_argument('--check-parameters', type=str2bool, default=True,
                        help='Check the parameters for any which don\'t reproduce the points.')
    parser.add_argument('--set-valid-range', type=lambda s: [int(item) for item in s.split(',')],
                        help='Start and end frame numbers to trim reconstruction to.')
    parser.add_argument('--flip-frames', type=lambda s: [int(item) for item in s.split(',')],
                        help='Flip HT at these frames (and subsequent).')
    parser.add_argument('--fix-camera-positions', type=str2bool, default=False,
                        help='Fix camera position drift.')
    parser.add_argument('--fix-curvature', type=str2bool, default=False,
                        help='Fix curvature kinks and consistency.')

    # -- Generic arguments
    parser.add_argument('--frame-idx-from', type=str, default='reconst', choices=['reconst', 'trial'],
                        help='Frame indexing starts at 0=reconstruction.start_frame (reconst) or 0=0 (trial).')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Verification batch size.')
    parser.add_argument('--error-threshold', type=float, default=0.001,
                        help='Error threshold required for updating the parameters.')
    parser.add_argument('--plot-error-threshold', type=float, default=0.005,
                        help='Error threshold above which to plot examples.')
    parser.add_argument('--plot-n-examples', type=int, default=0,
                        help='Plot n examples above the plotting error threshold.')
    parser.add_argument('--plot-n-examples-per-batch', type=int, default=0,
                        help='Plot n examples above the plotting error threshold per batch.')
    parser.add_argument('--clamp-X0', type=str2bool, default=True)

    # -- Optimisation arguments
    parser.add_argument('--gpu-id', type=int, default=-1,
                        help='GPU id to use if using GPUs.')
    parser.add_argument('--train-steps', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--learning-rate-K', type=float, default=1e-4)
    parser.add_argument('--learning-rate-decay', type=float, default=0.99)
    parser.add_argument('--learning-rate-min', type=float, default=1e-5)
    parser.add_argument('--optimise-X0', type=str2bool, default=True)
    parser.add_argument('--optimise-T0', type=str2bool, default=True)
    parser.add_argument('--optimise-M10', type=str2bool, default=True)
    parser.add_argument('--optimise-K', type=str2bool, default=True)
    parser.add_argument('--optimise-lengths', type=str2bool, default=True)
    parser.add_argument('--loss-batch-mean-threshold', type=float, default=1e-2)

    # -- Camera parameter fix arguments
    parser.add_argument('--reg-weight', type=float, default=1e-1)
    parser.add_argument('--optimise-M10-threshold', type=float, default=5.)
    parser.add_argument('--optimise-K-threshold', type=float, default=5.)
    parser.add_argument('--optimise-shifts', type=str2bool, default=True)
    for k in CAM_PARAMETER_NAMES:
        if k == 'shifts':
            continue
        parser.add_argument(f'--use-mean-{k.replace("_", "-")}', type=str2bool, default=True)

    # -- Curvature fix arguments
    parser.add_argument('--loss-w-fh', type=float, default=1.)
    parser.add_argument('--loss-w-ft', type=float, default=1.)
    parser.add_argument('--loss-w-ht', type=float, default=1.)
    parser.add_argument('--loss-w-of', type=float, default=1.)
    parser.add_argument('--loss-w-oh', type=float, default=1.)
    parser.add_argument('--loss-w-ot', type=float, default=1.)
    parser.add_argument('--reg-w-ds', type=float, default=1e-1)
    parser.add_argument('--reg-w-dt', type=float, default=1e-1)
    parser.add_argument('--parabola-baseline', type=float, default=1e-1)
    parser.add_argument('--parabola-gradient', type=float, default=1.)
    parser.add_argument('--convergence-patience', type=int, default=100)
    parser.add_argument('--min-steps', type=int, default=50)

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def _check_bad_parameters(
        ts: TrialState,
        args: Namespace,
        save_dir: PosixPath,
):
    """
    Check for parameters which don't produce the target curves.
    """
    logger.info(f'Checking for bad parameters.')
    trial = ts.trial
    prs = ProjectRenderScoreModel(image_size=trial.crop_size, clamp_X0=args.clamp_X0)

    # Make output directory
    save_dir_n = save_dir / f'{START_TIMESTAMP}_check_params'
    os.makedirs(save_dir_n, exist_ok=True)

    # Fetch parameters
    points = ts.get('points')
    X0 = ts.get('X0')
    T0 = ts.get('T0')
    M10 = ts.get('M10')
    K = ts.get('curvatures')
    points_2d = ts.get('points_2d')

    # Load cam coeffs and base points for verification
    cam_coeffs = np.concatenate([
        ts.get(f'cam_{k}').copy()
        for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
    ], axis=2)
    points_3d_base = ts.get('points_3d_base')
    points_2d_base = ts.get('points_2d_base')
    lengths = ts.get('length')

    # Check frames
    points_r = np.zeros_like(points)
    points_2d_r1 = np.zeros_like(points_2d)
    points_2d_r2 = np.zeros_like(points_2d)

    n_frames = ts.n_frames
    n_batches = int(n_frames / args.batch_size) + 1
    errors_3d = np.zeros(n_frames)
    errors_2d_a = np.zeros(n_frames)
    errors_2d_b = np.zeros(n_frames)
    errors_2d_c = np.zeros(n_frames)

    for i in range(n_batches):
        logger.info(f'Calculating errors for batch {i + 1}/{n_batches}.')
        start_idx = i * args.batch_size
        end_idx = min(n_frames, (i + 1) * args.batch_size)
        points_batch = torch.from_numpy(points[start_idx:end_idx])
        points_2d_batch = torch.from_numpy(points_2d[start_idx:end_idx])
        X0_batch = torch.from_numpy(X0[start_idx:end_idx, 0])
        if args.clamp_X0:
            X0_batch = X0_batch.clamp(min=-0.5, max=0.5)

        # Reconstruct the 3D points from the parameters
        points_r_batch, tangents_r, M1_r = integrate_curvature(
            X0_batch,
            torch.from_numpy(T0[start_idx:end_idx, 0]),
            torch.from_numpy(lengths[start_idx:end_idx, 0]),
            smooth_parameter(torch.from_numpy(K[start_idx:end_idx].copy()), 15, mode='gaussian'),
            torch.from_numpy(M10[start_idx:end_idx, 0]),
        )
        points_r[start_idx:end_idx] = points_r_batch

        # Reproject the original 3D points to 2D
        points_2d_r1_batch = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs[start_idx:end_idx]),
            points_3d=points_batch,
            points_3d_base=torch.from_numpy(points_3d_base[start_idx:end_idx].astype(np.float32)),
            points_2d_base=torch.from_numpy(points_2d_base[start_idx:end_idx].astype(np.float32)),
        ).permute(0, 2, 1, 3)
        points_2d_r1[start_idx:end_idx] = points_2d_r1_batch

        # Reproject the reconstructed 3D points to 2D
        points_2d_r2_batch = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs[start_idx:end_idx]),
            points_3d=points_r_batch,
            points_3d_base=torch.from_numpy(points_3d_base[start_idx:end_idx].astype(np.float32)),
            points_2d_base=torch.from_numpy(points_2d_base[start_idx:end_idx].astype(np.float32)),
        ).permute(0, 2, 1, 3)
        points_2d_r2[start_idx:end_idx] = points_2d_r2_batch

        # Calculate errors as maximum distance between points
        errors_3d[start_idx:end_idx] = (points_r_batch - points_batch).norm(dim=-1).amax(dim=-1)
        errors_2d_a[start_idx:end_idx] = (points_2d_batch - points_2d_r1_batch).norm(dim=-1).amax(dim=(-1, -2))
        errors_2d_b[start_idx:end_idx] = (points_2d_batch - points_2d_r2_batch).norm(dim=-1).amax(dim=(-1, -2))
        errors_2d_c[start_idx:end_idx] = (points_2d_r1_batch - points_2d_r2_batch).norm(dim=-1).amax(dim=(-1, -2))

    # Plot the errors
    def _plot_errors(ax_, errors_):
        ax_.plot(np.arange(ts.n_frames), errors_)
        ax_.axhline(y=0, color='grey')
        ax_.axhline(y=errors_.mean(), color='purple', linestyle='--')

    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Trial={trial.id}.')
    for j, (title, errs) in enumerate(
            {
                'errors_3d': errors_3d,
                'errors_2d o-r1': errors_2d_a,
                'errors_2d o-r2': errors_2d_b,
                'errors_2d r1-r2': errors_2d_c,
            }.items()):
        ax = axes[0, j]
        ax.set_title(title)
        _plot_errors(ax, errs)
        ax = axes[1, j]
        _plot_errors(ax, errs)
        ax.set_xlabel('Frame')
        ax.set_yscale('log')

    fig.tight_layout()

    if save_plots:
        path = save_dir_n / f'errors.{img_extension}'
        logger.info(f'Saving errors plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    above_threshold_idxs = (errors_3d > args.plot_error_threshold).nonzero()[0]
    if args.plot_n_examples > 0 and len(above_threshold_idxs) > 0:
        idxs_at = np.random.choice(
            len(above_threshold_idxs),
            min(len(above_threshold_idxs), args.plot_n_examples),
            replace=False
        )
        idxs = above_threshold_idxs[idxs_at]

        for idx in idxs:
            # Prepare images with overlays
            img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{idx:06d}.npz'
            try:
                image_triplet = np.load(img_path)['images']
                if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
                    logger.warning('Prepared images are the wrong size, regeneration needed!')
                    raise RuntimeError()
            except Exception:
                logger.warning('Prepared images not available, skipping.')
                break

            images_original = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d[idx]).astype(np.int32)
            )
            images_r1 = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d_r1[idx]).astype(np.int32)
            )
            images_r2 = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d_r2[idx]).astype(np.int32)
            )

            NF_original = NaturalFrame(points[idx])
            NF_reconst = NaturalFrame(points_r[idx])

            fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(3, 2)
            fig.suptitle(f'Trial={trial.id}. Frame={idx}. '
                         f'Error 3D={errors_3d[idx]:.5f}. '
                         f'Errors 2D=[{errors_2d_a[idx]:.5f}, {errors_2d_b[idx]:.5f}, {errors_2d_c[idx]:.5f}].')

            # Plot 3D
            ax = fig.add_subplot(gs[:, 0], projection='3d')
            ax.set_title('Original + reconst')
            plot_natural_frame_3d(NF_original, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='autumn', use_centred_midline=False)
            plot_natural_frame_3d(NF_reconst, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='winter', use_centred_midline=False)

            # Plot reprojections
            ax = fig.add_subplot(gs[0, 1])
            ax.set_title('Original')
            ax.imshow(images_original, aspect='auto')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 1])
            ax.set_title('Reconst 1')
            ax.imshow(images_r1, aspect='auto')
            ax.axis('off')

            ax = fig.add_subplot(gs[2, 1])
            ax.set_title('Reconst 2')
            ax.imshow(images_r2, aspect='auto')
            ax.axis('off')

            fig.tight_layout()

            if save_plots:
                path = save_dir_n / f'{idx:05d}.{img_extension}'
                logger.info(f'Saving plot to {path}.')
                plt.savefig(path)

            if show_plots:
                plt.show()

            plt.close(fig)


def _set_valid_range(
        ts: TrialState,
        args: Namespace,
):
    """
    Trim the reconstruction to the given range.
    """
    reconstruction = ts.reconstruction
    start_frame = args.set_valid_range[0]
    end_frame = args.set_valid_range[1]
    if args.frame_idx_from == 'reconst':
        start_frame += reconstruction.start_frame
        end_frame += reconstruction.start_frame
    assert start_frame >= reconstruction.start_frame
    assert end_frame <= reconstruction.end_frame
    if args.dry_run:
        logger.info(f'(DRY RUN) NOT-Setting valid range for reconstruction as {start_frame}-{end_frame}.')
    else:
        logger.info(f'Setting valid range for reconstruction as {start_frame}-{end_frame}.')
        reconstruction.start_frame_valid = start_frame
        reconstruction.end_frame_valid = end_frame
        reconstruction.save()
        logger.info('Saved.')


def _verify_flipped_batch(
        trial: Trial,
        start_frame: int,
        clamp_X0: bool,

        X0: np.ndarray,
        T0: np.ndarray,
        M10: np.ndarray,
        K: np.ndarray,
        points: np.ndarray,
        points_2d: np.ndarray,
        points_3d_base: np.ndarray,
        points_2d_base: np.ndarray,

        lengths: np.ndarray,
        cam_coeffs: np.ndarray,

        X0_flipped: np.ndarray,
        T0_flipped: np.ndarray,
        M10_flipped: np.ndarray,
        K_flipped: np.ndarray,
        points_flipped: np.ndarray,
        points_2d_flipped: np.ndarray,

        save_dir: PosixPath,
        plot_error_threshold: float,
        plot_n_examples_per_batch: int,
) -> np.ndarray:
    """
    Verify that the flipped parameters are consistent.
    """
    prs = ProjectRenderScoreModel(image_size=trial.crop_size, clamp_X0=clamp_X0)

    # Reconstruct the 3D points from the parameters
    X0 = torch.from_numpy(X0)
    if clamp_X0:
        X0 = X0.clamp(min=-0.5, max=0.5)
    points_r, tangents_r, M1_r = integrate_curvature(
        X0,
        torch.from_numpy(T0),
        torch.from_numpy(lengths),
        smooth_parameter(torch.from_numpy(K.copy()), 15, mode='gaussian'),
        torch.from_numpy(M10),
    )

    # Reconstruct the 3D points from the flipped parameters
    X0_flipped = torch.from_numpy(X0_flipped)
    if clamp_X0:
        X0_flipped = X0_flipped.clamp(min=-0.5, max=0.5)
    points_flipped_r, tangents_flipped_r, M1_flipped_r = integrate_curvature(
        X0_flipped,
        torch.from_numpy(T0_flipped),
        torch.from_numpy(lengths),
        smooth_parameter(torch.from_numpy(K_flipped.copy()), 15, mode='gaussian'),
        torch.from_numpy(M10_flipped),
    )

    # Calculate errors as maximum distance between flipped points and reconstructed flipped points
    errors = (points_flipped_r - points_flipped).norm(dim=-1).amax(dim=-1)
    errors[errors.isnan()] = 0
    idxs_sorted = torch.argsort(errors, descending=True)

    # Plot examples above error threshold
    n_above_plot_threshold = (errors > plot_error_threshold).sum()
    if n_above_plot_threshold > 0 and plot_n_examples_per_batch > 0:
        logger.info(f'{n_above_plot_threshold} above plotting error threshold.')
        if save_plots:
            save_dir = save_dir / 'above_threshold_examples'
            os.makedirs(save_dir, exist_ok=True)

        # Reproject the 3D points to 2D
        points_2d_r = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs),
            points_3d=points_r,
            points_3d_base=torch.from_numpy(points_3d_base.astype(np.float32)),
            points_2d_base=torch.from_numpy(points_2d_base.astype(np.float32)),
        )
        points_2d_r = points_2d_r.numpy().transpose(0, 2, 1, 3)

        # Reproject the flipped 3D points to 2D
        points_2d_flipped_r = prs._project_to_2d(
            cam_coeffs=torch.from_numpy(cam_coeffs),
            points_3d=points_flipped_r,
            points_3d_base=torch.from_numpy(points_3d_base.astype(np.float32)),
            points_2d_base=torch.from_numpy(points_2d_base.astype(np.float32)),
        )
        points_2d_flipped_r = points_2d_flipped_r.numpy().transpose(0, 2, 1, 3)

        for idx in idxs_sorted[:plot_n_examples_per_batch]:
            n = start_frame + idx

            # Prepare images with overlays
            img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{n:06d}.npz'
            try:
                image_triplet = np.load(img_path)['images']
                if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
                    logger.warning('Prepared images are the wrong size, regeneration needed!')
                    raise RuntimeError()
            except Exception:
                logger.warning('Prepared images not available, stopping here.')

            images_original = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d[idx]).astype(np.int32)
            )
            images_original_r = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d_r[idx]).astype(np.int32)
            )
            images_flipped = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d_flipped[idx]).astype(np.int32)
            )
            images_flipped_r = generate_annotated_images(
                image_triplet=image_triplet,
                points_2d=np.round(points_2d_flipped_r[idx]).astype(np.int32)
            )

            NF_original = NaturalFrame(points[idx])
            NF_original_reconst = NaturalFrame(points_r[idx].numpy())
            NF_flipped = NaturalFrame(points_flipped[idx])
            NF_flipped_reconst = NaturalFrame(points_flipped_r[idx].numpy())

            fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(4, 2)
            fig.suptitle(f'Trial={trial.id}. Frame={n}. Error={errors[idx]:.5f}.')

            # Plot 3D
            ax = fig.add_subplot(gs[:2, 0], projection='3d')
            ax.set_title('Original + reconst')
            plot_natural_frame_3d(NF_original, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='autumn', use_centred_midline=False)
            plot_natural_frame_3d(NF_original_reconst, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='winter', use_centred_midline=False)
            ax = fig.add_subplot(gs[2:, 0], projection='3d')
            ax.set_title('Flipped + reconst')
            plot_natural_frame_3d(NF_flipped, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='autumn', use_centred_midline=False)
            plot_natural_frame_3d(NF_flipped_reconst, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                                  midline_cmap='winter', use_centred_midline=False)

            # Plot reprojections
            ax = fig.add_subplot(gs[0, 1])
            ax.set_title('Original')
            ax.imshow(images_original, aspect='auto')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 1])
            ax.set_title('Reconst')
            ax.imshow(images_original_r, aspect='auto')
            ax.axis('off')

            ax = fig.add_subplot(gs[2, 1])
            ax.set_title('Flipped')
            ax.imshow(images_flipped, aspect='auto')
            ax.axis('off')

            ax = fig.add_subplot(gs[3, 1])
            ax.set_title('Flipped + reconst')
            ax.imshow(images_flipped_r, aspect='auto')
            ax.axis('off')

            fig.tight_layout()

            if save_plots:
                path = save_dir / f'{n:05d}.{img_extension}'
                logger.info(f'Saving plot to {path}.')
                plt.savefig(path)

            if show_plots:
                plt.show()

            plt.close(fig)

    return errors


def _flip_frames(
        ts: TrialState,
        args: Namespace,
        save_dir: PosixPath,
):
    """
    Apply HT flips at given frames.
    """
    trial = ts.trial
    N = ts.parameters.n_points_total
    flip_frames: List[int] = args.flip_frames

    for n in flip_frames:
        if args.frame_idx_from == 'reconst':
            n += ts.reconstruction.start_frame
        logger.info(f'Flipping frames from {n} onwards.')

        # Make output directory
        save_dir_n = save_dir / f'{START_TIMESTAMP}_flip_{n:05d}'
        os.makedirs(save_dir_n, exist_ok=True)

        # Points just reverse
        points = ts.get('points', n)
        points_flipped = points.copy()[:, ::-1]

        # X0 shifts by 1
        X0 = ts.get('X0', n)
        X0_flipped = points.copy()[:, int(N / 2)][:, None]

        # Flip the tangent vectors
        T0 = ts.get('T0', n)
        T0_flipped = -(T0.copy())

        # M10 probably needs to change...
        M10 = ts.get('M10', n)
        M10_flipped = -(M10.copy())

        # Curvatures just reverse
        K = ts.get('curvatures', n)
        K_flipped = K.copy()[:, ::-1]
        K_flipped = np.stack([-K_flipped[..., 0], K_flipped[..., 1]], axis=-1)

        # 2D points just reverse
        points_2d = ts.get('points_2d', n)
        points_2d_flipped = points_2d.copy()[:, ::-1]

        # Scores just reverse
        scores = ts.get('scores', n)
        scores_flipped = scores.copy()[:, ::-1]

        # Load cam coeffs and base points for verification
        cam_coeffs = np.concatenate([
            ts.get(f'cam_{k}', n).copy()
            for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
        ], axis=2)
        points_3d_base = ts.get('points_3d_base', n)
        points_2d_base = ts.get('points_2d_base', n)
        lengths = ts.get('length', n)

        # Verify the flipped parameters are consistent
        n_frames = ts.n_frames - n + ts.reconstruction.start_frame
        n_batches = int(n_frames / args.batch_size) + 1
        errors = np.zeros(n_frames)
        for i in range(n_batches):
            logger.info(f'Verifying batch {i + 1}/{n_batches}.')
            start_idx = i * args.batch_size
            end_idx = min(n_frames, (i + 1) * args.batch_size)
            errors_batch = _verify_flipped_batch(
                trial=trial,
                start_frame=n + start_idx,
                clamp_X0=args.clamp_X0,
                X0=X0[start_idx:end_idx, 0],
                T0=T0[start_idx:end_idx, 0],
                M10=M10[start_idx:end_idx, 0],
                K=K[start_idx:end_idx],
                points=points[start_idx:end_idx],
                points_2d=points_2d[start_idx:end_idx],
                points_3d_base=points_3d_base[start_idx:end_idx],
                points_2d_base=points_2d_base[start_idx:end_idx],
                lengths=lengths[start_idx:end_idx, 0],
                cam_coeffs=cam_coeffs[start_idx:end_idx],
                X0_flipped=X0_flipped[start_idx:end_idx, 0],
                T0_flipped=T0_flipped[start_idx:end_idx, 0],
                M10_flipped=M10_flipped[start_idx:end_idx, 0],
                K_flipped=K_flipped[start_idx:end_idx],
                points_flipped=points_flipped[start_idx:end_idx],
                points_2d_flipped=points_2d_flipped[start_idx:end_idx],
                save_dir=save_dir_n,
                plot_error_threshold=args.plot_error_threshold,
                plot_n_examples_per_batch=args.plot_n_examples_per_batch,
            )
            errors[start_idx:end_idx] = errors_batch

        # Plot the errors
        def _plot_errors(ax_):
            ax_.plot(n + np.arange(n_frames), errors)
            ax_.axhline(y=0, color='grey')
            ax_.axhline(y=errors.mean(), color='purple', linestyle='--')
            ax_.axhline(y=args.error_threshold, color='red')

        fig, axes = plt.subplots(2, figsize=(10, 8), sharex=True)
        ax = axes[0]
        ax.set_title(f'Trial={trial.id}. Flipping from frame={n}. Average error={errors.mean():.5f}.')
        _plot_errors(ax)
        ax = axes[1]
        _plot_errors(ax)
        ax.set_xlabel('Frame')
        ax.set_yscale('log')

        fig.tight_layout()

        if save_plots:
            path = save_dir_n / f'flip_errors.{img_extension}'
            logger.info(f'Saving errors plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        if errors.max() < args.error_threshold:
            logger.info(f'Maximum error ({errors.max():.5f}) < Error threshold ({args.error_threshold:.4f}).')
            if args.dry_run:
                logger.info('(DRY RUN) NOT-Updating parameters.')
            else:
                logger.info('Updating parameters.')
                ts.states['points'][n:] = points_flipped
                ts.states['X0'][n:] = X0_flipped
                ts.states['T0'][n:] = T0_flipped
                ts.states['M10'][n:] = M10_flipped
                ts.states['curvatures'][n:] = K_flipped
                ts.states['points_2d'][n:] = points_2d_flipped
                ts.states['scores'][n:] = scores_flipped
                ts.save()
                logger.info('Saved.')
        else:
            logger.warning(
                f'Maximum error ({errors.max():.5f}) > Error threshold ({args.error_threshold:.4f})'
                f' - Aborting flips.'
            )
            break


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


def _init_cam_coeffs(
        ts: TrialState,
        args: Namespace,
        device,
        fixed: bool = False
):
    """
    Initialise camera coefficients.
    """
    f_range = {'start_frame': ts.reconstruction.start_frame, 'end_frame': ts.reconstruction.end_frame}
    params = {}
    for k in CAM_PARAMETER_NAMES:
        v = torch.from_numpy(ts.get(f'cam_{k}', **f_range).copy()).to(device)
        if fixed and hasattr(args, f'use_mean_{k}') and getattr(args, f'use_mean_{k}'):
            v[:] = v.mean(axis=0, keepdim=True)
        params[k] = v
    return params


def _get_cam_coeffs_for_frame(
        cam_coeffs: Dict[str, torch.Tensor],
        idx: int,
) -> torch.Tensor:
    """
    Build rotation matrices and collate all the camera coefficients together.
    """
    cc = []
    for k in CAM_PARAMETER_NAMES:
        v = cam_coeffs[k][idx]
        if k == 'rotation_preangles':
            Rs = []
            for i in range(3):
                pre = v[i]
                cos_phi, sin_phi = pre[0]
                cos_theta, sin_theta = pre[1]
                cos_psi, sin_psi = pre[2]
                Ri = make_rotation_matrix(cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi)
                Rs.append(Ri.flatten())
            v = torch.stack(Rs)
        cc.append(v)
    return torch.cat(cc, dim=1)


def _plot_camera_fix_examples(
        trial: Trial,
        batch: int,
        step: int,
        start_frame_num: int,
        plot_n: int,
        points: torch.Tensor,
        points_2d: torch.Tensor,
        points_f: torch.Tensor,
        points_2d_f: torch.Tensor,
        losses_p2d_batch: torch.Tensor,
        losses_reg_batch: torch.Tensor,
        save_dir: PosixPath
):
    """
    Plot some example frames comparing originals to updated.
    """
    idxs = torch.argsort(losses_p2d_batch, descending=True)[:plot_n]

    for i, idx in enumerate(idxs):
        frame_num = start_frame_num + idx

        # Prepare images with overlays
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{frame_num:06d}.npz'
        image_triplet = np.load(img_path)['images']
        images_original = generate_annotated_images(
            image_triplet=image_triplet,
            points_2d=np.round(to_numpy(points_2d[idx])).astype(np.int32)
        )
        images_r = generate_annotated_images(
            image_triplet=image_triplet,
            points_2d=np.round(to_numpy(points_2d_f[idx])).astype(np.int32)
        )
        NF_original = NaturalFrame(to_numpy(points[idx]))
        NF_reconst = NaturalFrame(to_numpy(points_f[idx]))

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(2, 2)
        fig.suptitle(f'Trial={trial.id}. Frame={frame_num}. '
                     f'P2D loss={losses_p2d_batch[idx]:.5f}. '
                     f'Reg loss={losses_reg_batch[idx]:.5f}. ')

        # Plot 3D
        ax = fig.add_subplot(gs[:, 0], projection='3d')
        ax.set_title('Original + reconst')
        plot_natural_frame_3d(NF_original, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                              midline_cmap='autumn', use_centred_midline=False)
        lims1 = np.array([getattr(ax, f'get_{xyz}lim')() for xyz in 'xyz'])
        plot_natural_frame_3d(NF_reconst, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                              midline_cmap='winter', use_centred_midline=False)
        lims2 = np.array([getattr(ax, f'get_{xyz}lim')() for xyz in 'xyz'])
        for j, xyz in enumerate('xyz'):
            getattr(ax, f'set_{xyz}lim')(min(lims1[j][0], lims2[j][0]), max(lims1[j][1], lims2[j][1]))

        # Plot reprojections
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Original')
        ax.imshow(images_original, aspect='auto')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('Fixed')
        ax.imshow(images_r, aspect='auto')
        ax.axis('off')

        fig.tight_layout()

        if save_plots:
            path = save_dir / f'{batch:04d}_{step:05d}_{idx:05d}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        plt.close(fig)


def _process_camfix_batch(
        prs: ProjectRenderScoreModel,
        clamp_X0: bool,
        X0: torch.Tensor,
        T0: torch.Tensor,
        M10: torch.Tensor,
        K: torch.Tensor,
        lengths: torch.Tensor,
        cam_coeffs: torch.Tensor,
        points_3d_base: torch.Tensor,
        points_2d_base: torch.Tensor,
        points_2d_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a batch of curves.
    """
    device = X0.device
    batch_size = len(X0)

    # Build the 3D points from the new parameters.
    if clamp_X0:
        X0 = X0.clamp(min=-0.5, max=0.5)
    p3d_batch, tangents_r, M1_r = integrate_curvature(
        X0,
        T0,
        lengths,
        smooth_parameter(K, 15, mode='gaussian'),
        M10,
    )

    # Build cam coefficients
    cam_coeffs_flat = torch.stack([
        _get_cam_coeffs_for_frame(cam_coeffs, idx)
        for idx in range(batch_size)
    ])

    # Project to 2D
    p2d_batch = prs._project_to_2d(
        points_3d=p3d_batch,
        cam_coeffs=cam_coeffs_flat,
        points_3d_base=points_3d_base,
        points_2d_base=points_2d_base,
        clamp=False
    ).permute(0, 2, 1, 3)

    # Calculate losses - average pixel distance
    losses_p2d = (points_2d_target - p2d_batch).norm(dim=-1).mean(dim=(1, 2))

    # Regularisation
    shifts_batch = cam_coeffs['shifts'][:, :, 0]
    reg_X0 = ((X0[1:] - X0[:-1])**2).sum(dim=-1)
    reg_T0 = ((T0[1:] - T0[:-1])**2).sum(dim=-1)
    reg_M10 = ((M10[1:] - M10[:-1])**2).sum(dim=-1)
    reg_K = ((K[1:] - K[:-1])**2).sum(dim=(-1, -2))
    reg_l = (lengths[1:] - lengths[:-1])**2
    reg_shifts = ((shifts_batch[1:] - shifts_batch[:-1])**2).sum(dim=-1)
    reg_loss_batch = reg_X0 + reg_T0 + reg_M10 + reg_K + reg_l + reg_shifts

    # Spread the regularisation across the batch
    reg_spread = torch.zeros(batch_size, device=device)
    reg_spread[1:] = reg_loss_batch
    reg_spread[:-1] = reg_spread[:-1] + reg_loss_batch
    reg_spread[1:-1] = reg_spread[1:-1] / 2

    # Check for tracking failures and just zero the losses where it happens
    tracking_failure_idxs = (points_3d_base.sum(dim=-1) == 0) | (points_2d_base.sum(dim=(-1, -2)) == 0)
    losses_p2d[tracking_failure_idxs] = 0

    return p3d_batch, p2d_batch, losses_p2d, reg_spread


def _fix_camera_positions(
        ts: TrialState,
        args: Namespace,
        save_dir: PosixPath,
):
    """
    Fix the camera positions.
    """
    logger.info(f'Fix camera positions.')
    trial = ts.trial
    device = _init_devices(args)
    prs = ProjectRenderScoreModel(image_size=trial.crop_size, clamp_X0=args.clamp_X0)
    prs = prs.to(device)

    # Make output directory
    save_dir_n = save_dir / f'{START_TIMESTAMP}_fix_camera_positions'
    os.makedirs(save_dir_n, exist_ok=True)

    # Fetch parameters
    f_range = {'start_frame': ts.reconstruction.start_frame, 'end_frame': ts.reconstruction.end_frame}
    cam_coeffs = _init_cam_coeffs(ts, args, device=device, fixed=False)
    points = torch.from_numpy(ts.get('points', **f_range).copy()).to(device)
    points_2d = torch.from_numpy(ts.get('points_2d', **f_range).copy()).to(device)
    points_3d_base = torch.from_numpy(ts.get('points_3d_base', **f_range).copy()).to(torch.float32).to(device)
    points_2d_base = torch.from_numpy(ts.get('points_2d_base', **f_range).copy()).to(torch.float32).to(device)
    X0 = torch.from_numpy(ts.get('X0', **f_range).copy())[:, 0].to(device)
    T0 = torch.from_numpy(ts.get('T0', **f_range).copy())[:, 0].to(device)
    M10 = torch.from_numpy(ts.get('M10', **f_range).copy())[:, 0].to(device)
    lengths = torch.from_numpy(ts.get('length', **f_range).copy())[:, 0].to(device)
    K = torch.from_numpy(ts.get('curvatures', **f_range).copy()).to(device)
    n_frames = len(points)

    # Initialise output placeholders
    X0f = torch.zeros_like(X0)
    T0f = torch.zeros_like(T0)
    M10f = torch.zeros_like(M10)
    Kf = torch.zeros_like(K)
    lengthsf = torch.zeros_like(lengths)
    cam_coeffs_f = _init_cam_coeffs(ts, args, device=device, fixed=True)

    # Initialise new 3D and 2D points
    points_f = torch.zeros_like(points, device=device)
    points_2d_f = torch.zeros_like(points_2d, device=device)

    # Initialise optimiser and outputs
    n_batches = int(n_frames / args.batch_size) + 1
    losses = np.zeros((n_batches, args.train_steps))
    lrs = np.zeros((n_batches, args.train_steps))
    train_steps = np.zeros(n_batches, dtype=np.int32)

    # Process batches at a time
    for i in range(n_batches):
        logger.info(f'--- Fixing camera parameters for batch {i + 1}/{n_batches}.')
        start_idx = i * args.batch_size
        end_idx = min(n_frames, (i + 1) * args.batch_size)
        idxs = np.arange(start_idx, end_idx)
        batch_size = end_idx - start_idx

        # Instantiate optimisable parameters
        if args.optimise_shifts and i > 0:
            csb = torch.ones_like(cam_coeffs['shifts'][idxs], device=device) \
                  * cam_shifts_batch.detach().clone().mean(dim=0, keepdim=True)
            cam_shifts_batch = nn.Parameter(csb, requires_grad=args.optimise_shifts)
        else:
            cam_shifts_batch = nn.Parameter(cam_coeffs['shifts'][idxs], requires_grad=args.optimise_shifts)
        if args.optimise_X0 and i > 0:
            xfb = torch.ones_like(X0[idxs], device=device) \
                  * X0f_batch.detach().clone().mean(dim=0, keepdim=True)
            X0f_batch = nn.Parameter(xfb, requires_grad=args.optimise_X0)
        else:
            X0f_batch = nn.Parameter(X0[idxs], requires_grad=args.optimise_X0)
        T0f_batch = nn.Parameter(T0[idxs], requires_grad=args.optimise_T0)
        M10f_batch = nn.Parameter(M10[idxs], requires_grad=False)
        Kf_batch = nn.Parameter(K[idxs], requires_grad=False)
        lengthsf_batch = nn.Parameter(lengths[idxs], requires_grad=args.optimise_lengths)

        optimiser = AdamW(
            params=[
                {'params': [cam_shifts_batch, X0f_batch, T0f_batch, M10f_batch, lengthsf_batch],
                 'lr': args.learning_rate},
                {'params': [Kf_batch], 'lr': args.learning_rate_K}
            ],
            amsgrad=True,
            weight_decay=0
        )
        scheduler = ReduceLROnPlateau(
            optimiser,
            mode='min',
            factor=args.learning_rate_decay,
            patience=10,
            cooldown=5,
            min_lr=args.learning_rate_min
        )

        # Build camera coefficients for these frames
        cam_coeffs_f_batch = {
            k: cam_coeffs_f[k][idxs]
            for k in CAM_PARAMETER_NAMES
        }
        cam_coeffs_f_batch['shifts'] = cam_shifts_batch

        for step in range(args.train_steps):
            optimiser.zero_grad()
            points_f_batch, points_2d_f_batch, losses_p2d_batch, losses_reg_batch = _process_camfix_batch(
                prs=prs,
                clamp_X0=args.clamp_X0,
                X0=X0f_batch,
                T0=T0f_batch,
                M10=M10f_batch,
                K=Kf_batch,
                lengths=lengthsf_batch,
                cam_coeffs=cam_coeffs_f_batch,
                points_3d_base=points_3d_base[idxs],
                points_2d_base=points_2d_base[idxs],
                points_2d_target=points_2d[idxs]
            )
            batch_loss = (losses_p2d_batch + args.reg_weight * losses_reg_batch).mean()
            batch_loss.backward()
            optimiser.step()
            lrs[i, step] = optimiser.param_groups[0]['lr']
            scheduler.step(batch_loss.item())
            losses[i, step] = batch_loss.item()

            # Enable curvature gradient only when the loss is small enough
            if batch_loss < args.optimise_M10_threshold and args.optimise_M10 and not M10f_batch.requires_grad:
                logger.info('Enabling M10 optimisation.')
                M10f_batch.requires_grad_(True)
            if batch_loss < args.optimise_K_threshold and args.optimise_K and not Kf_batch.requires_grad:
                logger.info('Enabling curvature optimisation.')
                Kf_batch.requires_grad_(True)

            # Clamp parameters
            with torch.no_grad():
                # X0 should be in range
                if args.clamp_X0:
                    eps = 1e-8
                    X0f_batch.data = X0f_batch.clamp(min=-0.5 + eps, max=0.5 - eps)

                # T0 should be normalised
                T0f_batch.data = normalise(T0f_batch)

                # M10 should be orthogonal to T0 and normalised
                M10f_batch2 = normalise(M10f_batch)
                M10f_batch2 = orthogonalise(M10f_batch2, T0f_batch)
                M10f_batch.data = normalise(M10f_batch2)

            if step > 0 and step % 10 == 0:
                logger.info(
                    f'Train step {step}/{args.train_steps}. '
                    f'\tLoss: {batch_loss.item():.6f}. '
                    f'\tP2D: {losses_p2d_batch.mean().item():.6f}. '
                    f'\tReg: {losses_reg_batch.mean().item():.6f}. '
                    f'\tlr: {optimiser.param_groups[0]["lr"]:.5f}. '
                )

            if batch_loss.item() < args.loss_batch_mean_threshold:
                break

        if not args.dry_run and batch_loss.item() > args.error_threshold:
            logger.warning(f'Failed to reach error threshold ({args.error_threshold:.2f}), aborting.')
            exit(1)

        # Get final output from updated parameters
        points_f_batch, points_2d_f_batch, losses_p2d_batch, losses_reg_batch = _process_camfix_batch(
            prs=prs,
            clamp_X0=args.clamp_X0,
            X0=X0f_batch,
            T0=T0f_batch,
            M10=M10f_batch,
            K=Kf_batch,
            lengths=lengthsf_batch,
            cam_coeffs=cam_coeffs_f_batch,
            points_3d_base=points_3d_base[idxs],
            points_2d_base=points_2d_base[idxs],
            points_2d_target=points_2d[idxs]
        )

        # Verify batch
        points_r, tangents_r, M1_r = integrate_curvature(
            X0f_batch,
            T0f_batch,
            lengthsf_batch,
            smooth_parameter(Kf_batch, 15, mode='gaussian'),
            M10f_batch,
        )

        assert torch.allclose(points_r, points_f_batch)

        # Update parameters
        for k in CAM_PARAMETER_NAMES:
            if k == 'shifts':
                cam_coeffs_f[k][idxs] = cam_shifts_batch
            else:
                cam_coeffs_f[k][idxs] = cam_coeffs_f_batch[k]
        X0f[idxs] = X0f_batch
        T0f[idxs] = T0f_batch
        M10f[idxs] = M10f_batch
        Kf[idxs] = Kf_batch
        lengthsf[idxs] = lengthsf_batch
        points_f[idxs] = points_f_batch
        points_2d_f[idxs] = points_2d_f_batch
        train_steps[i] = step

        if args.plot_n_examples_per_batch > 0:
            _plot_camera_fix_examples(
                trial=trial,
                batch=i,
                step=step,
                start_frame_num=start_idx,
                plot_n=min(batch_size, args.plot_n_examples_per_batch),
                points=points[idxs],
                points_2d=points_2d[idxs],
                points_f=points_f_batch,
                points_2d_f=points_2d_f_batch,
                losses_p2d_batch=losses_p2d_batch,
                losses_reg_batch=losses_reg_batch,
                save_dir=save_dir_n
            )

        # Plot loss and parameter changes across the batch
        if 1:
            fig = plt.figure(figsize=(16, 14))
            gs = GridSpec(4, 10)
            fig.suptitle(f'Batch {i + 1}/{n_batches}. Loss={batch_loss.item():.4f}. Steps={step}.')

            ax = fig.add_subplot(gs[0, 0:5])
            ax.set_title('Batch loss')
            ax.plot(np.arange(step), losses[i, :step])
            ax.axhline(y=args.loss_batch_mean_threshold, linestyle='--', color='red', alpha=0.7)
            ax.set_yscale('log')
            ax.grid()
            ax = fig.add_subplot(gs[0, 5:10])
            ax.set_title('Learning rate')
            ax.plot(np.arange(step), lrs[i, :step])
            ax.grid()

            for j in range(5):
                vo = to_numpy([X0, T0, M10, K, cam_coeffs['shifts']][j][idxs])
                vf = to_numpy([X0f, T0f, M10f, Kf, cam_coeffs_f['shifts']][j][idxs])
                for k in range(3):
                    ax = fig.add_subplot(gs[1 + k, j * 2:j * 2 + 2])
                    if k == 0:
                        ax.set_title(['X0', 'T0', 'M10', 'K/l', 'shifts'][j])
                    if j == 3:
                        ax.set_ylabel(['m1', 'm2', 'l'][k])
                        if k == 2:
                            vo = to_numpy(lengths[idxs])
                            vf = to_numpy(lengthsf[idxs])
                            ax.plot(vo, label='original')
                            ax.plot(vf, label='fixed')
                        else:
                            ax.plot(np.abs(vo[:, :, k]).sum(axis=-1), label='original')
                            ax.plot(np.abs(vf[:, :, k]).sum(axis=-1), label='fixed')
                    else:
                        ax.set_ylabel(['x', 'y', 'z'][k])
                        ax.plot(vo[:, k], label='original')
                        ax.plot(vf[:, k], label='fixed')
                    ax.legend()

            fig.tight_layout()

            if save_plots:
                path = save_dir_n / f'batch_{i:04d}_{step:05d}.{img_extension}'
                logger.info(f'Saving plot to {path}.')
                plt.savefig(path)

            if show_plots:
                plt.show()

            plt.close(fig)

    # Collect the final losses for each batch
    final_losses = losses[np.arange(n_batches), train_steps]

    # Plot training errors
    fig, axes = plt.subplots(3, figsize=(10, 10), sharex=True)
    ax = axes[0]
    ax.set_title('Losses')
    ax.plot(final_losses, marker='x')
    ax.axhline(y=args.error_threshold, linestyle='--', color='red', alpha=0.7)
    ax.grid()
    ax = axes[1]
    ax.plot(final_losses, marker='x')
    ax.set_yscale('log')
    ax.axhline(y=args.error_threshold, linestyle='--', color='red', alpha=0.7)
    ax.grid()
    ax = axes[2]
    ax.set_title('Training steps')
    ax.plot(train_steps, marker='x')
    ax.axhline(y=args.train_steps, linestyle='--', color='red', alpha=0.7)
    ax.set_xlabel('Batch')
    ax.grid()

    fig.tight_layout()

    if save_plots:
        path = save_dir_n / f'losses_{i:04d}_{step:05d}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Check how a stationary focal point drifts with the cameras
    logger.info('Checking stationary point drift.')
    p3d_base_mean = points_3d_base.mean(axis=0)
    p2d_base_mean = points_2d_base.mean(axis=0)
    p2d_target = torch.ones((1, 3, 2), device=device) * trial.crop_size / 2
    p2d = torch.zeros((n_frames, 3, 2), device=device)
    p2d_f = torch.zeros((n_frames, 3, 2), device=device)

    for i in range(n_batches):
        if i > 0 and i % 10 == 0:
            logger.info(f'Checking drift for batch {i + 1}/{n_batches}.')
        start_idx = i * args.batch_size
        end_idx = min(n_frames, (i + 1) * args.batch_size)
        idxs = np.arange(start_idx, end_idx)
        batch_size = end_idx - start_idx
        p3d_batch = torch.zeros((batch_size, 1, 3), device=device)
        cam_coeffs_batch = torch.stack([
            _get_cam_coeffs_for_frame(cam_coeffs, idx)
            for idx in idxs
        ])
        cam_coeffs_f_batch = torch.stack([
            _get_cam_coeffs_for_frame(cam_coeffs_f, idx)
            for idx in idxs
        ])
        projection_parameters = dict(
            points_3d=p3d_batch,
            points_3d_base=p3d_base_mean[None, :].repeat_interleave(batch_size, 0),
            points_2d_base=p2d_base_mean[None, :].repeat_interleave(batch_size, 0),
            clamp=False
        )

        p2d_batch = prs._project_to_2d(cam_coeffs=cam_coeffs_batch, **projection_parameters)[:, :, 0]
        p2d[start_idx:end_idx] = p2d_batch.detach()

        p2d_f_batch = prs._project_to_2d(cam_coeffs=cam_coeffs_f_batch, **projection_parameters)[:, :, 0]
        p2d_f[start_idx:end_idx] = p2d_f_batch.detach()

    # Calculate errors as distance from 2d points to centre of images
    errors = to_numpy(torch.sum((p2d - p2d_target)**2, dim=-1))
    errors_f = to_numpy(torch.sum((p2d_f - p2d_target)**2, dim=-1))

    # Plot the drift
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 5)
    recency = np.linspace(0, 1, n_frames)
    cmap_original = plt.get_cmap('Blues')
    cmap_fixed = plt.get_cmap('Reds')

    def _plot_2d_drift(ax_, px_, py_, cmap_):
        Z = np.stack([px_, py_], axis=1)
        points = Z[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=recency, cmap=cmap_, alpha=0.6)
        ax_.add_collection(lc)

    for i in range(3):
        # 2D drift, original vs fixed
        px = to_numpy(p2d[:, i, 0])
        py = to_numpy(p2d[:, i, 1])
        pxf = to_numpy(p2d_f[:, i, 0])
        pyf = to_numpy(p2d_f[:, i, 1])

        ax = fig.add_subplot(gs[i, 2])
        _plot_2d_drift(ax, px, py, cmap_original)
        _plot_2d_drift(ax, pxf, pyf, cmap_fixed)
        ax.set_xlim(left=0, right=trial.crop_size)
        ax.set_ylim(bottom=0, top=trial.crop_size)
        ax.set_xticks([])
        ax.set_yticks([])

        # Drift over time
        ax = fig.add_subplot(gs[i, 0:2])
        ax.set_title(f'Camera {i}')
        ax.plot(px, label='x', color='blue', alpha=0.7)
        ax.plot(pxf, label='xf', color='darkblue', linestyle='--', alpha=0.7)
        ax.plot(py, label='y', color='green', alpha=0.7)
        ax.plot(pyf, label='yf', color='darkgreen', linestyle='--', alpha=0.7)
        ax.legend()

        # Errors
        ax = fig.add_subplot(gs[i, 3:5])
        ax.set_title('Errors')
        ax.plot(errors[:, i], label='Before fix')
        ax.plot(errors_f[:, i], label='After fix')
        ax.set_yscale('log')
        ax.legend()

    fig.tight_layout()

    if save_plots:
        path = save_dir_n / f'drift.{img_extension}'
        logger.info(f'Saving drift plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Plot trajectory comparisons
    X_tracking = get_trajectory(trial_id=trial.id, tracking_only=True, **f_range)[0][:, 0]
    X_original = to_numpy(points.mean(dim=1) + points_3d_base)
    X_fixed = to_numpy(points_f.mean(dim=1) + points_3d_base)
    if len(X_tracking) < len(X_original):
        X_original = X_original[:len(X_tracking)]
        X_fixed = X_fixed[:len(X_tracking)]
    elif len(X_tracking) > len(X_original):
        X_tracking = X_tracking[:len(X_original)]

    # Centre the trajectories
    X_tracking -= X_tracking.mean(axis=0, keepdims=True)
    X_original -= X_original.mean(axis=0, keepdims=True)
    X_fixed -= X_fixed.mean(axis=0, keepdims=True)

    # Plot 3D and 2D views
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 3)
    ax = fig.add_subplot(gs[:2, :], projection='3d')
    ax.set_title('blue=tracking, green=original, red=fixed')
    ax.scatter(*X_tracking.T, c='blue', alpha=0.6, s=1)
    ax.scatter(*X_original.T, c='green', alpha=0.6, s=1)
    ax.scatter(*X_fixed.T, c='red', alpha=0.6, s=1)
    projections = ['xy', 'yz', 'xz']
    for i, p in enumerate(projections):
        if p == 'xy':
            X_tracking_p = np.delete(X_tracking, 2, 1)
            X_original_p = np.delete(X_original, 2, 1)
            X_fixed_p = np.delete(X_fixed, 2, 1)
        elif p == 'yz':
            X_tracking_p = np.delete(X_tracking, 0, 1)
            X_original_p = np.delete(X_original, 0, 1)
            X_fixed_p = np.delete(X_fixed, 0, 1)
        elif p == 'xz':
            X_tracking_p = np.delete(X_tracking, 1, 1)
            X_original_p = np.delete(X_original, 1, 1)
            X_fixed_p = np.delete(X_fixed, 1, 1)
        ax = fig.add_subplot(gs[2, i])
        ax.set_title(p)
        ax.scatter(X_tracking_p[:, 0], X_tracking_p[:, 1], c='blue', alpha=0.6, s=1)
        ax.scatter(X_original_p[:, 0], X_original_p[:, 1], c='green', alpha=0.6, s=1)
        ax.scatter(X_fixed_p[:, 0], X_fixed_p[:, 1], c='red', alpha=0.6, s=1)
    fig.tight_layout()

    if save_plots:
        path = save_dir_n / f'trajectory_comparisons.{img_extension}'
        logger.info(f'Saving trajectory comparisons plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Updating parameters
    if final_losses.max() < args.error_threshold:
        logger.info(f'Maximum error ({final_losses.max():.5f}) < Error threshold ({args.error_threshold:.4f}).')
        if args.dry_run:
            logger.info('(DRY RUN) NOT-Updating parameters.')
        else:
            logger.info('Updating parameters.')
            start_idx = f_range['start_frame']
            end_idx = f_range['end_frame']

            # Update the camera parameters
            for k in CAM_PARAMETER_NAMES:
                ts.states[f'cam_{k}'][start_idx:end_idx] = to_numpy(cam_coeffs_f[k])

            # Update the curve parameters
            ts.states['X0'][start_idx:end_idx] = to_numpy(X0f)[:, None, :]
            ts.states['T0'][start_idx:end_idx] = to_numpy(T0f)[:, None, :]
            ts.states['M10'][start_idx:end_idx] = to_numpy(M10f)[:, None, :]
            ts.states['curvatures'][start_idx:end_idx] = to_numpy(Kf)
            ts.states['length'][start_idx:end_idx] = to_numpy(lengthsf)[:, None]

            # Update the points
            ts.states['points'][start_idx:end_idx] = to_numpy(points_f)
            ts.states['points_2d'][start_idx:end_idx] = to_numpy(points_2d_f)

            ts.save()
            logger.info('Saved.')
    else:
        logger.warning(
            f'Maximum error ({final_losses.max():.5f}) > Error threshold ({args.error_threshold:.4f})'
            f' - Not updating parameters.'
        )


def _plot_curvature_fix_examples(
        trial: Trial,
        batch: int,
        step: int,
        start_frame_num: int,
        plot_n: int,
        points: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        points_f: torch.Tensor,
        points_2d_f: torch.Tensor,
        K_f: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        losses_combined: torch.Tensor,
        regs: Dict[str, torch.Tensor],
        save_dir: PosixPath
):
    """
    Plot some example frames comparing originals to updated.
    """
    idxs = torch.argsort(losses_combined, descending=True)[:plot_n]

    for i, idx in enumerate(idxs):
        frame_num = start_frame_num + idx
        Xi = to_numpy(points[idx])
        Xfi = to_numpy(points_f[idx])
        X2i = to_numpy(points_2d[idx])
        X2fi = to_numpy(points_2d_f[idx])

        # Prepare images with overlays
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{frame_num:06d}.npz'
        image_triplet = np.load(img_path)['images']
        images_original = generate_annotated_images(
            image_triplet=image_triplet,
            points_2d=np.round(X2i).astype(np.int32)
        )
        images_r = generate_annotated_images(
            image_triplet=image_triplet,
            points_2d=np.round(X2fi).astype(np.int32)
        )
        NF_original = NaturalFrame(Xi)
        NF_reconst = NaturalFrame(Xfi)

        fig = plt.figure(figsize=(10, 9))
        gs = GridSpec(4, 2)
        title = f'Trial={trial.id}. Frame={frame_num}. Loss={losses_combined[idx]:.4E}.\n' \
                f'Losses=[' + ', '.join([f'{k}: {v[idx].item():.3E}' for k, v in losses.items()]) + f']\n ' + \
                f'Regs=[' + ', '.join([f'{k}: {v[idx].item():.3E}' for k, v in regs.items()]) + ']\n'
        fig.suptitle(title)

        # Plot 3D
        ax = fig.add_subplot(gs[0:2, 0], projection='3d')
        ax.set_title('Original + fixed')
        plot_natural_frame_3d(NF_original, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                              midline_cmap='autumn', use_centred_midline=False)
        lims1 = np.array([getattr(ax, f'get_{xyz}lim')() for xyz in 'xyz'])
        plot_natural_frame_3d(NF_reconst, azim=60, show_frame_arrows=False, show_pca_arrows=False, ax=ax,
                              midline_cmap='winter', use_centred_midline=False)
        lims2 = np.array([getattr(ax, f'get_{xyz}lim')() for xyz in 'xyz'])
        for j, xyz in enumerate('xyz'):
            getattr(ax, f'set_{xyz}lim')(min(lims1[j][0], lims2[j][0]), max(lims1[j][1], lims2[j][1]))

        # Plot reprojections
        ax = fig.add_subplot(gs[2, 0])
        ax.set_title('Original')
        ax.imshow(images_original, aspect='auto')
        ax.axis('off')

        ax = fig.add_subplot(gs[3, 0])
        ax.set_title('Fixed')
        ax.imshow(images_r, aspect='auto')
        ax.axis('off')

        # Plot distances from original
        d3d = np.linalg.norm(Xi - Xfi, axis=-1)
        d2d = np.linalg.norm(X2i - X2fi, axis=-1)
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Point distances from originals')
        ax.set_xlabel('u')
        ax.set_ylabel('2D')
        for j in range(3):
            ax.plot(d2d[:, j], label=f'cam{j}', alpha=0.7)
        ax.legend()
        ax = ax.twinx()
        ax.set_ylabel('3D')
        ax.plot(d3d, color='purple', linestyle='--', linewidth=3, alpha=0.5)

        # Plot curvature
        Ki = to_numpy(K[idx])
        Ki_f = to_numpy(K_f[idx])
        N2 = K.shape[1] / 2
        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('$|\kappa|$')
        ax.plot(np.linalg.norm(Ki, axis=-1), label='Original')
        ax.plot(np.linalg.norm(Ki_f, axis=-1), label='Fixed')
        ax.axvline(x=N2, linestyle=':', color='pink', alpha=0.6)
        ax.legend()

        ax = fig.add_subplot(gs[2, 1])
        ax.set_title('$m_1$')
        ax.plot(Ki[:, 0], label='Original')
        ax.plot(Ki_f[:, 0], label='Fixed')
        ax.axvline(x=N2, linestyle=':', color='pink', alpha=0.6)

        ax = fig.add_subplot(gs[3, 1])
        ax.set_title('$m_2$')
        ax.plot(Ki[:, 1], label='Original')
        ax.plot(Ki_f[:, 1], label='Fixed')
        ax.axvline(x=N2, linestyle=':', color='pink', alpha=0.6)

        fig.tight_layout()

        if save_plots:
            path = save_dir / f'{batch:04d}_{step:05d}_{idx:05d}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        plt.close(fig)


@torch.jit.script
def _process_curvature_batch(
        X0: torch.Tensor,
        T0: torch.Tensor,
        M10: torch.Tensor,
        K: torch.Tensor,
        lengths: torch.Tensor,
        points_3d_base: torch.Tensor,
        points_2d_base: torch.Tensor,
        X_original: torch.Tensor,
        parabola_baseline: float,
        parabola_gradient: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Process a batch of curves.
    """
    device = X0.device
    batch_size = len(X0)
    N = K.shape[1]
    # smooth_parameter(K, 15, mode='gaussian'),  # todo: parameterise this?

    # Build the 3D curve from the updated parameters
    X_f, T_f, M1_f = integrate_curvature(X0, T0, lengths, K, M10)

    # Rebuild it again from the head and the tail
    X_h, T_h, M1_h = integrate_curvature(X_f[:, 1], T_f[:, 1], lengths, K, M1_f[:, 1], start_idx=1)
    X_t, T_t, M1_t = integrate_curvature(X_f[:, -2], T_f[:, -2], lengths, K, M1_f[:, -2], start_idx=N - 2)

    # Calculate losses - L2 distance between curves to ensure consistency
    L_fh = (X_f - X_h).norm(dim=-1, p=2).mean(dim=-1)
    L_ft = (X_f - X_t).norm(dim=-1, p=2).mean(dim=-1)
    L_ht = (X_h - X_t).norm(dim=-1, p=2).mean(dim=-1)

    # Match original curve, but weighted towards ends
    w = parabola_baseline + parabola_gradient * torch.linspace(-1, 1, N, device=device)**2
    L_of = (w * (X_original - X_f).norm(dim=-1, p=2)).sum(dim=-1)
    L_oh = (w * (X_original - X_h).norm(dim=-1, p=2)).sum(dim=-1)
    L_ot = (w * (X_original - X_t).norm(dim=-1, p=2)).sum(dim=-1)

    losses = {
        'fh': L_fh,
        'ft': L_ft,
        'ht': L_ht,
        'of': L_of,
        'oh': L_oh,
        'ot': L_ot,
    }

    # Regularisation 1 - smoothness in time
    reg_X0 = ((X0[1:] - X0[:-1])**2).sum(dim=-1)
    reg_T0 = ((T0[1:] - T0[:-1])**2).sum(dim=-1)
    reg_M10 = ((M10[1:] - M10[:-1])**2).sum(dim=-1)
    reg_K = ((K[1:] - K[:-1])**2).sum(dim=(-1, -2))
    reg_l = (lengths[1:] - lengths[:-1])**2
    reg1_batch = reg_X0 + reg_T0 + reg_M10 + reg_K + reg_l

    # Spread the regularisation across the batch
    reg_dt = torch.zeros(batch_size, device=device)
    reg_dt[1:] = reg1_batch
    reg_dt[:-1] = reg_dt[:-1] + reg1_batch
    reg_dt[1:-1] = reg_dt[1:-1] / 2

    # Regularisation 2 - smoothness along body
    reg_ds = ((K[:, 1:] - K[:, :-1])**2).sum(dim=(-1, -2))
    reg_ds2 = ((K[:, 2:] - K[:, :-2])**2).sum(dim=(-1, -2)) / 2

    regs = {
        'dt': reg_dt,
        'ds': reg_ds + reg_ds2,
    }

    # Check for tracking failures and just zero the losses where it happens
    tracking_failure_idxs = (points_3d_base.sum(dim=-1) == 0) | (points_2d_base.sum(dim=(-1, -2)) == 0)
    if tracking_failure_idxs.sum() > 0:
        for k, l in losses.items():
            l[tracking_failure_idxs] = 0

    return losses, regs


def _project_curvature_batch(
        prs: ProjectRenderScoreModel,
        X0: torch.Tensor,
        T0: torch.Tensor,
        M10: torch.Tensor,
        K: torch.Tensor,
        lengths: torch.Tensor,
        cam_coeffs: torch.Tensor,
        points_3d_base: torch.Tensor,
        points_2d_base: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a batch of curves.
    """
    batch_size = len(X0)

    # Build the 3D curve from the updated parameters
    p3d_batch, T_f, M1_f = integrate_curvature(X0, T0, lengths, K, M10)

    # Build cam coefficients
    cam_coeffs_flat = torch.stack([
        _get_cam_coeffs_for_frame(cam_coeffs, idx)
        for idx in range(batch_size)
    ])

    # Project to 2D
    p2d_batch = prs._project_to_2d(
        points_3d=p3d_batch,
        cam_coeffs=cam_coeffs_flat,
        points_3d_base=points_3d_base,
        points_2d_base=points_2d_base,
        clamp=False
    ).permute(0, 2, 1, 3)

    return p3d_batch, p2d_batch


def _fix_curvature(
        ts: TrialState,
        args: Namespace,
        save_dir: PosixPath,
):
    """
    Fix the curvature.
    """
    logger.info(f'Fix curvature.')
    trial = ts.trial
    device = _init_devices(args)
    prs = ProjectRenderScoreModel(image_size=trial.crop_size, clamp_X0=args.clamp_X0)
    prs = torch.jit.script(prs)
    prs = prs.to(device)
    batch_prefix_size = 2

    loss_weightings = {
        'fh': args.loss_w_fh,
        'ft': args.loss_w_ft,
        'ht': args.loss_w_ht,
        'of': args.loss_w_of,
        'oh': args.loss_w_oh,
        'ot': args.loss_w_ot,
    }
    reg_weightings = {
        'ds': args.reg_w_ds,
        'dt': args.reg_w_dt,
    }

    # Make output directory
    save_dir_n = save_dir / f'{START_TIMESTAMP}_fix_curvature'
    os.makedirs(save_dir_n, exist_ok=True)

    # Fetch parameters
    f_range = {'start_frame': ts.reconstruction.start_frame, 'end_frame': ts.reconstruction.end_frame}
    cam_coeffs = _init_cam_coeffs(ts, args, device=device, fixed=False)
    points = torch.from_numpy(ts.get('points', **f_range).copy()).to(device)
    points_2d = torch.from_numpy(ts.get('points_2d', **f_range).copy()).to(device)
    points_3d_base = torch.from_numpy(ts.get('points_3d_base', **f_range).copy()).to(torch.float32).to(device)
    points_2d_base = torch.from_numpy(ts.get('points_2d_base', **f_range).copy()).to(torch.float32).to(device)
    X0 = torch.from_numpy(ts.get('X0', **f_range).copy())[:, 0].to(device)
    T0 = torch.from_numpy(ts.get('T0', **f_range).copy())[:, 0].to(device)
    M10 = torch.from_numpy(ts.get('M10', **f_range).copy())[:, 0].to(device)
    lengths = torch.from_numpy(ts.get('length', **f_range).copy())[:, 0].to(device)
    K = torch.from_numpy(ts.get('curvatures', **f_range).copy()).to(device)
    n_frames = len(points)

    # Initialise output placeholders
    X0f = torch.zeros_like(X0)
    T0f = torch.zeros_like(T0)
    M10f = torch.zeros_like(M10)
    Kf = torch.zeros_like(K)
    lengthsf = torch.zeros_like(lengths)

    # Initialise new 3D and 2D points
    points_f = torch.zeros_like(points, device=device)
    points_2d_f = torch.zeros_like(points_2d, device=device)

    # Calculate number of batches accounting for overlaps
    n_batches = int(n_frames / (args.batch_size - batch_prefix_size)) + 1

    # Initialise optimiser and outputs
    losses = {
        k: np.zeros((n_batches, args.train_steps))
        for k in loss_weightings.keys()
    }
    losses_c = np.zeros((n_batches, args.train_steps))
    regs = {
        k: np.zeros((n_batches, args.train_steps))
        for k in reg_weightings.keys()
    }
    lrs = np.zeros((n_batches, args.train_steps))
    train_steps = np.zeros(n_batches, dtype=np.int32)

    # Process batches at a time
    for i in range(n_batches):
        logger.info(f'--- Fixing curvature for batch {i + 1}/{n_batches}.')

        # Build batch idxs
        if i == 0:
            start_idx = 0
            start_idx_prefix = 0
            end_idx = min(n_frames, args.batch_size)
        else:
            start_idx = i * (args.batch_size - batch_prefix_size) + batch_prefix_size
            start_idx_prefix = start_idx - batch_prefix_size
            end_idx = min(n_frames, start_idx + args.batch_size - batch_prefix_size)

        idxs_all = np.arange(start_idx_prefix, end_idx)
        idxs_opt = np.arange(start_idx, end_idx)
        idxs_prefix = np.arange(start_idx_prefix, start_idx)
        batch_size = end_idx - start_idx_prefix

        if len(idxs_opt) == len(idxs_prefix):
            logger.info('Empty batch!')
            continue

        # Instantiate optimisable parameters
        X0o_batch = nn.Parameter(X0[idxs_opt], requires_grad=args.optimise_X0)
        T0o_batch = nn.Parameter(T0[idxs_opt], requires_grad=args.optimise_T0)
        M10o_batch = nn.Parameter(M10[idxs_opt], requires_grad=args.optimise_M10)
        Ko_batch = nn.Parameter(K[idxs_opt], requires_grad=args.optimise_K)
        lengthso_batch = nn.Parameter(lengths[idxs_opt], requires_grad=args.optimise_lengths)
        X0f_batch, T0f_batch, M10f_batch, Kf_batch, lengthsf_batch = None, None, None, None, None

        def prefix_batch():
            # Prefix the previous batch overlap
            nonlocal X0f_batch, T0f_batch, M10f_batch, Kf_batch, lengthsf_batch
            X0f_batch = torch.cat([X0f[idxs_prefix].detach(), X0o_batch])
            T0f_batch = torch.cat([T0f[idxs_prefix].detach(), T0o_batch])
            M10f_batch = torch.cat([M10f[idxs_prefix].detach(), M10o_batch])
            Kf_batch = torch.cat([Kf[idxs_prefix].detach(), Ko_batch])
            lengthsf_batch = torch.cat([lengthsf[idxs_prefix].detach(), lengthso_batch])

        # Align frames
        logger.info('Aligning frames.')
        prefix_batch()
        obs = len(idxs_opt) - batch_prefix_size
        alignment_angles = np.zeros(obs)
        alignment_angles_post = np.zeros(obs)
        M10_dists = np.zeros(obs)
        alignment_errors = np.zeros(obs)

        with torch.no_grad():
            for k in range(obs):
                k2 = k + batch_prefix_size

                # Calculate optimal alignment rotation angle between frames
                mc_prev = Kf_batch[k2 - 1, :, 0] + 1j * Kf_batch[k2 - 1, :, 1]
                mc_next = Kf_batch[k2, :, 0] + 1j * Kf_batch[k2, :, 1]
                opt_angle = -torch.angle(torch.dot(mc_prev, mc_next.conj()))
                alignment_angles[k] = opt_angle.item()

                # Rotate the curvature components
                mc_aligned = torch.exp(-1j * opt_angle) * mc_next
                K_aligned = torch.stack([mc_aligned.real, mc_aligned.imag], axis=-1)
                ver_angle = -torch.angle(torch.dot(mc_prev, mc_aligned.conj()))
                alignment_angles_post[k] = ver_angle.item()
                assert ver_angle.abs() <= opt_angle.abs()

                # Rotate the M10 vector around the tangent
                T0k = T0f_batch[k2]
                M10k = M10f_batch[k2]
                I = torch.eye(3, device=device)
                cosA = torch.cos(opt_angle)
                sinA = torch.sin(opt_angle)
                outer = torch.einsum('i,j->ij', T0k, T0k)
                cross = torch.cross(T0k[..., None].repeat(1, 3), I)
                R = cosA * I \
                    + sinA * cross \
                    + (1 - cosA) * outer
                M10_aligned = torch.einsum('ij,j->i', R, M10k)
                M10_dists[k] = torch.norm(M10k - M10_aligned).item()

                # Verify that the rotated frame gives the same curve
                points_o, tangents_o, M1_o = integrate_curvature(
                    X0f_batch[k2].unsqueeze(0),
                    T0f_batch[k2].unsqueeze(0),
                    lengthsf_batch[k2].unsqueeze(0),
                    Kf_batch[k2].unsqueeze(0),
                    M10f_batch[k2].unsqueeze(0),
                )
                points_a, tangents_a, M1_a = integrate_curvature(
                    X0f_batch[k2].unsqueeze(0),
                    T0f_batch[k2].unsqueeze(0),
                    lengthsf_batch[k2].unsqueeze(0),
                    K_aligned.unsqueeze(0),
                    M10_aligned.unsqueeze(0),
                )
                err = ((points_o - points_a)**2).sum()
                alignment_errors[k] = err.item()
                Kf_batch[k2] = K_aligned
                M10f_batch[k2] = M10_aligned
                Ko_batch.data[k] = K_aligned
                M10o_batch.data[k] = M10_aligned

        # Plot alignment results
        fig, axes = plt.subplots(3, figsize=(6, 6))
        fig.suptitle(f'Alignment results. Trial {trial.id}. Batch {i + 1}/{n_batches}.')

        ax = axes[0]
        ax.set_title('Alignment angles')
        ax.plot(alignment_angles, label='Angle')
        ax.plot(alignment_angles_post, label='Verification')
        ax.legend()

        ax = axes[1]
        ax.set_title('M10 adjustment distance')
        ax.plot(M10_dists)

        ax = axes[2]
        ax.set_title('Curve reconstruction errors')
        ax.plot(alignment_errors)

        fig.tight_layout()

        if save_plots:
            path = save_dir_n / f'batch_{i:04d}_alignment.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        plt.close(fig)

        # Initialise optimiser and lr scheduler
        optimiser = AdamW(
            params=[
                {'params': [X0o_batch, T0o_batch, M10o_batch, lengthso_batch],
                 'lr': args.learning_rate},
                {'params': [Ko_batch], 'lr': args.learning_rate_K}
            ],
            amsgrad=True,
            weight_decay=0
        )
        scheduler = ReduceLROnPlateau(
            optimiser,
            mode='min',
            factor=args.learning_rate_decay,
            patience=10,
            cooldown=5,
            min_lr=args.learning_rate_min
        )

        for step in range(args.train_steps):
            optimiser.zero_grad()
            prefix_batch()

            # Process the batch and calculate errors
            losses_batch, regs_batch = _process_curvature_batch(
                X0=X0f_batch,
                T0=T0f_batch,
                M10=M10f_batch,
                K=Kf_batch,
                lengths=lengthsf_batch,
                points_3d_base=points_3d_base[idxs_all],
                points_2d_base=points_2d_base[idxs_all],
                X_original=points[idxs_all],
                parabola_baseline=args.parabola_baseline,
                parabola_gradient=args.parabola_gradient,
            )

            # Sum the losses
            losses_c_batch = torch.zeros(batch_size, device=device)
            for k, l in losses_batch.items():
                losses_c_batch += loss_weightings[k] * l
                losses[k][i, step] = l.mean().item()
            for k, r in regs_batch.items():
                losses_c_batch += reg_weightings[k] * r
                regs[k][i, step] = r.mean().item()
            losses_c[i, step] = losses_c_batch.mean()
            batch_loss = losses_c_batch.mean()

            if step > args.min_steps and batch_loss.item() < args.loss_batch_mean_threshold:
                logger.info('Batch loss better than threshold, breaking.')
                break

            # Train step
            batch_loss.backward()
            optimiser.step()
            lrs[i, step] = optimiser.param_groups[0]['lr']
            scheduler.step(batch_loss.item())

            # Clamp parameters
            with torch.no_grad():
                # T0 should be normalised
                T0f_batch.data = normalise(T0f_batch)

                # M10 should be orthogonal to T0 and normalised
                M10f_batch2 = normalise(M10f_batch)
                M10f_batch2 = orthogonalise(M10f_batch2, T0f_batch)
                M10f_batch.data = normalise(M10f_batch2)

            if step > 0 and step % 10 == 0:
                log = f'Train step {step}/{args.train_steps}.' \
                      f'\tLoss: {batch_loss.item():.4E}. '
                for k, l in losses_batch.items():
                    log += f'\t{k}: {l.mean().item():.4E}. '
                for k, r in regs.items():
                    log += f'\t{k}: {r.mean().item():.4E}. '
                log += f'\tlr: {optimiser.param_groups[0]["lr"]:.4f}.'
                logger.info(log)

            if step > args.min_steps \
                    and step > args.convergence_patience \
                    and (lrs[i, step - args.convergence_patience:step] == args.learning_rate_min).all():
                logger.info('No longer improving at minimum learning rate, breaking.')
                break

        if not args.dry_run and batch_loss.item() > args.error_threshold:
            logger.warning(f'Failed to reach error threshold ({args.error_threshold:.2f}), aborting.')
            exit(1)

        # Build camera coefficients for these frames
        cam_coeffs_batch = {
            k: cam_coeffs[k][idxs_opt]
            for k in CAM_PARAMETER_NAMES
        }

        # Get final output from updated parameters
        points_o_batch, points_2d_o_batch = _project_curvature_batch(
            prs=prs,
            X0=X0o_batch,
            T0=T0o_batch,
            M10=M10o_batch,
            K=Ko_batch,
            lengths=lengthso_batch,
            cam_coeffs=cam_coeffs_batch,
            points_3d_base=points_3d_base[idxs_opt],
            points_2d_base=points_2d_base[idxs_opt],
        )

        # Verify batch
        points_r, tangents_r, M1_r = integrate_curvature(
            X0o_batch,
            T0o_batch,
            lengthso_batch,
            Ko_batch,
            M10o_batch,
        )

        diff = (points_o_batch - points_r).norm(dim=-1)
        assert diff.max() < 1e-6

        # Update parameters
        X0f[idxs_opt] = X0o_batch
        T0f[idxs_opt] = T0o_batch
        M10f[idxs_opt] = M10o_batch
        Kf[idxs_opt] = Ko_batch
        lengthsf[idxs_opt] = lengthso_batch
        points_f[idxs_opt] = points_o_batch
        points_2d_f[idxs_opt] = points_2d_o_batch
        train_steps[i] = step

        if args.plot_n_examples_per_batch > 0:
            losses_o_batch = {k: l[len(idxs_prefix):] for k, l in losses_batch.items()}
            regs_o_batch = {k: l[len(idxs_prefix):] for k, l in regs_batch.items()}
            _plot_curvature_fix_examples(
                trial=trial,
                batch=i,
                step=step,
                start_frame_num=start_idx,
                plot_n=min(batch_size - len(idxs_prefix), args.plot_n_examples_per_batch),
                points=points[idxs_opt],
                points_2d=points_2d[idxs_opt],
                K=K[idxs_opt],
                points_f=points_o_batch,
                points_2d_f=points_2d_o_batch,
                K_f=Ko_batch,
                losses=losses_o_batch,
                losses_combined=losses_c_batch[len(idxs_prefix):],
                regs=regs_o_batch,
                save_dir=save_dir_n
            )

        # Plot loss and parameter changes across the batch
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(5, 12)
        fig.suptitle(f'Batch {i + 1}/{n_batches}. Loss={batch_loss.item():.4f}. Steps={step}.')

        ax = fig.add_subplot(gs[0, 0:6])
        ax.set_title('Batch loss')
        ax.plot(np.arange(step), losses_c[i, :step])
        ax.axhline(y=args.loss_batch_mean_threshold, linestyle='--', color='red', alpha=0.7)
        ax.set_yscale('log')
        ax.grid()
        ax = fig.add_subplot(gs[0, 6:12])
        ax.set_title('Learning rate')
        ax.plot(np.arange(step), lrs[i, :step])
        ax.grid()

        ax = fig.add_subplot(gs[1, 0:4])
        ax.set_title('Losses - consistency')
        for k in ['fh', 'ft', 'ht']:
            ax.plot(np.arange(step), losses[k][i, :step], label=k)
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        ax = fig.add_subplot(gs[1, 4:8])
        ax.set_title('Losses - matching')
        for k in ['of', 'oh', 'ot']:
            ax.plot(np.arange(step), losses[k][i, :step], label=k)
        ax.set_yscale('log')
        ax.legend()
        ax.grid()
        ax = fig.add_subplot(gs[1, 8:12])
        ax.set_title('Regs')
        for k in ['ds', 'dt']:
            ax.plot(np.arange(step), regs[k][i, :step], label=k)
        ax.set_yscale('log')
        ax.legend()
        ax.grid()

        for j in range(4):
            vo = to_numpy([X0, T0, M10, K][j][idxs_opt])
            vf = to_numpy([X0f, T0f, M10f, Kf][j][idxs_opt])
            for k in range(3):
                ax = fig.add_subplot(gs[2 + k, j * 3:j * 3 + 3])
                if k == 0:
                    ax.set_title(['X0', 'T0', 'M10', 'K/l'][j])
                if j == 3:
                    ax.set_ylabel(['m1', 'm2', 'l'][k])
                    if k == 2:
                        vo = to_numpy(lengths[idxs_opt])
                        vf = to_numpy(lengthsf[idxs_opt])
                        ax.plot(vo, label='original')
                        ax.plot(vf, label='fixed')
                    else:
                        ax.plot(np.abs(vo[:, :, k]).sum(axis=-1), label='original')
                        ax.plot(np.abs(vf[:, :, k]).sum(axis=-1), label='fixed')
                else:
                    ax.set_ylabel(['x', 'y', 'z'][k])
                    ax.plot(vo[:, k], label='original')
                    ax.plot(vf[:, k], label='fixed')
                ax.legend()

        fig.tight_layout()

        if save_plots:
            path = save_dir_n / f'batch_{i:04d}_{step:05d}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        plt.close(fig)

    # Collect the final losses for each batch
    final_losses = losses_c[np.arange(n_batches), train_steps]

    # Plot training errors
    fig, axes = plt.subplots(3, figsize=(10, 10), sharex=True)
    ax = axes[0]
    ax.set_title('Losses')
    ax.plot(final_losses, marker='x')
    ax.axhline(y=args.error_threshold, linestyle='--', color='red', alpha=0.7)
    ax.grid()
    ax = axes[1]
    ax.plot(final_losses, marker='x')
    ax.set_yscale('log')
    ax.axhline(y=args.error_threshold, linestyle='--', color='red', alpha=0.7)
    ax.grid()
    ax = axes[2]
    ax.set_title('Training steps')
    ax.plot(train_steps, marker='x')
    ax.axhline(y=args.train_steps, linestyle='--', color='red', alpha=0.7)
    ax.set_xlabel('Batch')
    ax.grid()

    fig.tight_layout()

    if save_plots:
        path = save_dir_n / f'losses_{i:04d}_{step:05d}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Updating parameters
    if final_losses.max() < args.error_threshold:
        logger.info(f'Maximum error ({final_losses.max():.5f}) < Error threshold ({args.error_threshold:.4f}).')
        if args.dry_run:
            logger.info('(DRY RUN) NOT-Updating parameters.')
        else:
            logger.info('Updating parameters.')
            start_idx = f_range['start_frame']
            end_idx = f_range['end_frame']

            # Update the curve parameters
            ts.states['X0'][start_idx:end_idx] = to_numpy(X0f)[:, None, :]
            ts.states['T0'][start_idx:end_idx] = to_numpy(T0f)[:, None, :]
            ts.states['M10'][start_idx:end_idx] = to_numpy(M10f)[:, None, :]
            ts.states['curvatures'][start_idx:end_idx] = to_numpy(Kf)
            ts.states['length'][start_idx:end_idx] = to_numpy(lengthsf)[:, None]

            # Update the points
            ts.states['points'][start_idx:end_idx] = to_numpy(points_f)
            ts.states['points_2d'][start_idx:end_idx] = to_numpy(points_2d_f)

            ts.save()
            logger.info('Saved.')
    else:
        logger.warning(
            f'Maximum error ({final_losses.max():.5f}) > Error threshold ({args.error_threshold:.4f})'
            f' - Not updating parameters.'
        )


def fix():
    """
    Apply fixes to a MF result.
    """
    # from simple_worm.plot3d import interactive
    # interactive()

    args = get_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    ts = TrialState(reconstruction, read_only=args.dry_run, partial_load_ok=True)

    # Make output directory
    save_dir = LOGS_PATH / f'trial_{reconstruction.trial.id:03d}_r={reconstruction.id}'
    os.makedirs(save_dir, exist_ok=True)

    # Write meta data
    meta = to_dict(args)
    meta['date'] = START_TIMESTAMP
    with open(save_dir / f'args_{START_TIMESTAMP}.meta', 'w') as f:
        json.dump(meta, f, indent=2, separators=(',', ': '))

    # Check parameters
    if args.check_parameters:
        _check_bad_parameters(ts, args, save_dir)

    # Set valid range
    if args.set_valid_range is not None:
        assert len(args.set_valid_range) == 2, 'Start and end frames needed for setting a valid range.'
        _set_valid_range(ts, args)

    # Flip frames
    if args.flip_frames is not None:
        _flip_frames(ts, args, save_dir)

    # Fix camera positions
    if args.fix_camera_positions:
        _fix_camera_positions(ts, args, save_dir)

    # Fix curvature
    if args.fix_curvature:
        _fix_curvature(ts, args, save_dir)


if __name__ == '__main__':
    fix()
