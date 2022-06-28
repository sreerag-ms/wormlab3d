import json
import os
from argparse import Namespace, ArgumentParser
from pathlib import PosixPath
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from wormlab3d import PREPARED_IMAGES_PATH, logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Trial
from wormlab3d.midlines3d.mf_methods import integrate_curvature, smooth_parameter
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d
from wormlab3d.toolkit.util import print_args, to_dict, str2bool

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
    parser.add_argument('--check-parameters', type=str2bool, default=True,
                        help='Check the parameters for any which don\'t reproduce the points.')
    parser.add_argument('--set-valid-range', type=lambda s: [int(item) for item in s.split(',')],
                        help='Start and end frame numbers to trim reconstruction to.')
    parser.add_argument('--flip-frames', type=lambda s: [int(item) for item in s.split(',')],
                        help='Flip HT at these frames (and subsequent).')
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
    prs = ProjectRenderScoreModel(image_size=trial.crop_size)

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

        # Reconstruct the 3D points from the parameters
        points_r_batch, tangents_r, M1_r = integrate_curvature(
            torch.from_numpy(X0[start_idx:end_idx, 0]).clamp(min=-0.5, max=0.5),
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
        idxs_at = np.random.choice(len(above_threshold_idxs), args.plot_n_examples, replace=False)
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
    prs = ProjectRenderScoreModel(image_size=trial.crop_size)

    # Reconstruct the 3D points from the parameters
    points_r, tangents_r, M1_r = integrate_curvature(
        torch.from_numpy(X0).clamp(min=-0.5, max=0.5),
        torch.from_numpy(T0),
        torch.from_numpy(lengths),
        smooth_parameter(torch.from_numpy(K.copy()), 15, mode='gaussian'),
        torch.from_numpy(M10),
    )

    # Reconstruct the 3D points from the flipped parameters
    points_flipped_r, tangents_flipped_r, M1_flipped_r = integrate_curvature(
        torch.from_numpy(X0_flipped).clamp(min=-0.5, max=0.5),
        torch.from_numpy(T0_flipped),
        torch.from_numpy(lengths),
        smooth_parameter(torch.from_numpy(K_flipped.copy()), 15, mode='gaussian'),
        torch.from_numpy(M10_flipped),
    )

    # Calculate errors as maximum distance between flipped points and reconstructed flipped points
    errors = (points_flipped_r - points_flipped).norm(dim=-1).amax(dim=-1)
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


def fix():
    """
    Apply fixes to a MF result.
    """
    # from simple_worm.plot3d import interactive
    # interactive()

    args = get_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    ts = TrialState(reconstruction, read_only=True, partial_load_ok=True)

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
        _set_valid_range(ts, args, save_dir)

    # Flip frames
    if args.flip_frames is not None:
        _flip_frames(ts, args, save_dir)


if __name__ == '__main__':
    fix()
