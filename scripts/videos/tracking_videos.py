import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import PosixPath
from typing import List, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import torch
import yaml

from wormlab3d import LOGS_PATH, PREPARED_IMAGES_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.toolkit.util import to_dict
from wormlab3d.trajectories.util import smooth_trajectory


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate the tracking videos.')
    parser.add_argument('--dataset', type=str, help='Dataset by id.', required=True)
    parser.add_argument('--smoothing-window-postures', type=int, default=0, help='Smoothing window for the postures.')
    args = parser.parse_args()
    return args


def _prepare_image(
        image: np.ndarray,
) -> np.ndarray:
    """
    Prepare image for display.
    """
    z = ((1 - image) * 255).astype(np.uint8)
    z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
    return z


def _prepare_image_triplet(
        image_triplet: np.ndarray,
        points_2d: np.ndarray = None,
        overlay_midlines: bool = False
) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    if overlay_midlines:
        images = generate_annotated_images(image_triplet, points_2d)
        images = images.transpose(1, 0, 2)
    else:
        images = []
        for img in image_triplet:
            z = _prepare_image(img)
            images.append(z)
        images = np.fliplr(np.concatenate(images))

    # Add 1px border between images
    images[int(images.shape[0] / 3)] = 0
    images[int(images.shape[0] / 3 * 2)] = 0
    panel = np.concatenate(images, axis=1)

    return panel


def prepare_views(
        args: Namespace,
        trial: Trial,
        reconstruction: Optional[Reconstruction],
):
    """
    Prepare the camera views for the given trial.
    """
    logger.info(f'Preparing camera views for trial {trial.id}.')
    frame_nums = np.arange(trial.n_frames_min + 1)

    # Load the reconstruction data if there is any
    if reconstruction is not None:
        assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions supported!'
        logger.info('Loading reconstruction data.')
        r_start_frame = reconstruction.start_frame_valid
        r_end_frame = reconstruction.end_frame_valid
        ts = TrialState(reconstruction)
        points_3d = ts.get('points')
        if args.smoothing_window_postures > 1:
            points_3d = smooth_trajectory(points_3d, args.smoothing_window_postures)
        points_3d_base = ts.get('points_3d_base')
        points_2d_base = ts.get('points_2d_base')
        cam_coeffs = np.concatenate([
            ts.get(f'cam_{k}')
            for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
        ], axis=2)
        prs = ProjectRenderScoreModel(image_size=trial.crop_size)

    def prepare_frame_triplet(i: int) -> Tuple[List[np.ndarray], np.ndarray]:
        n = frame_nums[i]

        # Check images are present
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{n:06d}.npz'
        tracked_images = np.load(img_path)['images']
        if tracked_images.shape != (3, trial.crop_size, trial.crop_size):
            raise RuntimeError('Prepared images are the wrong size, regeneration needed!')

        # Prepare the combined frame with overlaid midlines if available
        if reconstruction is not None and r_start_frame <= n < r_end_frame:
            points_2d = prs._project_to_2d(
                cam_coeffs=torch.from_numpy(cam_coeffs[n][None, ...]),
                points_3d=torch.from_numpy(points_3d[n][None, ...]),
                points_3d_base=torch.from_numpy(points_3d_base[n][None, ...].astype(np.float32)),
                points_2d_base=torch.from_numpy(points_2d_base[n][None, ...].astype(np.float32)),
            )
            points_2d = points_2d[0].numpy().transpose(1, 0, 2)
            points_2d = np.round(points_2d).astype(np.int32)
            images_combined = _prepare_image_triplet(
                image_triplet=tracked_images,
                points_2d=points_2d,
                overlay_midlines=True
            )
        else:
            images_combined = _prepare_image_triplet(image_triplet=tracked_images)

        # Prepare the individual views
        tracked_images = [_prepare_image(img) for img in tracked_images]

        return tracked_images, images_combined

    return prepare_frame_triplet, frame_nums


def generate_trial_videos(
        ds: Dataset,
        trial: Trial,
        args: Namespace,
        output_dir: PosixPath,
):
    """
    Generate a tracking video showing the camera images with overlaid 2D midline reprojections where available.
    """
    r_id = ds.get_reconstruction_id_for_trial(trial)
    reconstruction = None
    if r_id is not None:
        reconstruction = Reconstruction.objects.get(id=r_id)
    else:
        return

    update_fn, frame_nums = prepare_views(
        args=args,
        trial=trial,
        reconstruction=reconstruction,
    )

    def make_ffmpeg_process(combined: bool = False, camera_idx: int = 0):
        title = f'Trial {trial.id}.'
        if combined:
            output_path = output_dir / f'trial={trial.id:03d}_combined_r={r_id}'
            title += ' Combined views.'
            if r_id is not None:
                title += f' Reconstruction {r_id}.'
            size = f'{int(trial.crop_size * 3)}x{trial.crop_size}'
        else:
            output_path = output_dir / f'trial={trial.id:03d}_camera={camera_idx}'
            title += f' Camera {camera_idx}.'
            size = f'{trial.crop_size}x{trial.crop_size}'
        output_args = {
            'pix_fmt': 'yuv444p',
            'vcodec': 'libx264',
            'r': trial.fps,
            'metadata:g:0': f'title={title}',
            'metadata:g:1': 'artist=Leeds Wormlab',
            'metadata:g:2': f'year={time.strftime("%Y")}',
        }
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=size)
            .output(str(output_path) + '.mp4', **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        return process

    processes = [make_ffmpeg_process(combined=True)]
    for camera_idx in range(3):
        processes.append(make_ffmpeg_process(combined=False, camera_idx=camera_idx))

    logger.info('Rendering frames.')
    n_frames = len(frame_nums)
    for i in range(n_frames):
        if i > 0 and i % 50 == 0:
            logger.info(f'Rendering frame {i}/{n_frames}.')

        # Update and write to streams
        try:
            tracked_images, images_combined = update_fn(i)
        except Exception as e:
            logger.warning(f'Error rendering frame {i}: {e}. Stopping here')
            break
        processes[0].stdin.write(images_combined.tobytes())
        for j, img in enumerate(tracked_images):
            processes[j + 1].stdin.write(img.tobytes())

    # Flush video
    for p in processes:
        p.stdin.close()
        p.wait()

    logger.info(f'Generated videos.')


def generate_tracking_videos():
    """
    Generate tracking videos for all trials in a dataset.
    """
    args = get_args()

    # Copy the spec with final args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / 'spec.yml', 'w') as f:
        spec = {
            'script': 'videos/tracking_videos',
            'created': START_TIMESTAMP,
            'args': to_dict(args),
        }
        yaml.dump(spec, f)

    # Fetch dataset
    ds = Dataset.objects.get(id=args.dataset)

    # Generate clips for each trial
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Generating tracking videos for trial={trial.id} ({i + 1}/{len(ds.include_trials)}).')
        trial_dir = output_dir / f'trial={trial.id:03d}'
        os.makedirs(trial_dir, exist_ok=True)
        generate_trial_videos(ds, trial, args, output_dir=trial_dir)


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_tracking_videos()
