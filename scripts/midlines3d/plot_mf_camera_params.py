import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Dataset
from wormlab3d.data.model.dataset import DatasetMidline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.mf_render_wrapper import RenderWrapper
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import to_numpy
from wormlab3d.trajectories.util import fetch_reconstruction

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'

CAM_PARAMETER_KEYS = [
    'Intrinsics',
    'Rotations',
    'Translations',
    'Distortions',
    'Shifts',
]


def _get_targets() -> Union[DatasetMidline3D, Reconstruction]:
    """
    Resolve the target.
    """
    parser = ArgumentParser(description='Wormlab3D script to plot the dynamic camera parameters.')
    parser.add_argument('--dataset', type=str, help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    args = parser.parse_args()

    # Check for dataset
    if args.dataset is not None:
        ds = Dataset.objects.get(id=args.dataset)
        assert type(ds) == DatasetMidline3D, 'Only DatasetMidline3D datasets work here!'
        return ds

    # Fetch reconstruction
    assert args.reconstruction is not None, 'Reconstruction or dataset must be specified!'
    reconstruction = fetch_reconstruction(reconstruction_id=args.reconstruction)
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions work for this!'

    return reconstruction


def _get_camera_parameters(
        ts: TrialState,
        key: str
) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Fetch particular camera parameters from the trial state.
    """
    assert key in CAM_PARAMETER_KEYS

    if key == 'Intrinsics':
        p = ts.get('cam_intrinsics')  # (T, 3, 4)
        titles = ['fx', 'fy', 'cx', 'cy']

    elif key == 'Rotations':
        p = ts.get('cam_rotation_preangles')  # (T, 3, 3, 2)
        p = np.arctan2(p[:, :, :, 0], p[:, :, :, 1])
        titles = ['theta', 'phi', 'psi']

    elif key == 'Translations':
        p = ts.get('cam_translations')  # (T, 3, 3)
        titles = ['x', 'y', 'z']

    elif key == 'Distortions':
        p = ts.get('cam_distortions')  # (T, 3, 5)
        titles = ['k1', 'k2', 'p1', 'p2', 'k3']

    elif key == 'Shifts':
        p = ts.get('cam_shifts')  # (T, 3, 1)
        titles = ['ds']

    p = p.transpose(2, 1, 0)  # (4, 3, T)

    return {
        'data': p,
        'titles': titles
    }


def plot_ds_camera_params(
        dataset: DatasetMidline3D,
        save_dir: Path,
        x_label: str = 'time',
):
    """
    Plot the camera parameters for all reconstructions in a dataset.
    """
    logger.info(f'Plotting camera parameters for all reconstructions in dataset {dataset.id}.')
    for i, rec in enumerate(dataset.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(dataset.reconstructions)}.')
        save_dir_rec = save_dir / f'trial={rec.trial.id:03d}_{rec.id}'
        if save_plots:
            os.makedirs(save_dir_rec, exist_ok=True)
        plot_rec_camera_params(rec, save_dir_rec, x_label)


def scan_ds_camera_params(
        dataset: DatasetMidline3D,
):
    """
    Scan reconstructions in a dataset and report the ones with highest variances.
    """
    logger.info(f'Scanning camera parameters for all reconstructions in dataset {dataset.id}.')

    variances = {}

    for i, rec in enumerate(dataset.reconstructions):
        logger.info(f'Reconstruction {i + 1}/{len(dataset.reconstructions)}.')
        ts = TrialState(
            reconstruction=rec,
            start_frame=rec.start_frame_valid,
            end_frame=rec.end_frame_valid
        )
        for k in CAM_PARAMETER_KEYS:
            p = _get_camera_parameters(ts, k)
            data_k, titles_k = p['data'], p['titles']
            n_plots = len(p['titles'])

            for j in range(n_plots):
                kj = f'{k}_{titles_k[j]}'
                if kj not in variances:
                    variances[kj] = {}

                x = data_k[j].astype(np.float64)
                variances[kj][rec.trial.id] = x.var(axis=-1).mean()

    # Report
    for k, v in variances.items():
        # Sort in descending order
        res = {k2: v2 for k2, v2 in sorted(v.items(), key=lambda item: item[1], reverse=True)}
        print(f'\n\n ==== {k} ==== \n')
        for trial_id, variance in res.items():
            print(f'{trial_id:03d}: {variance:.5E}')
            if variance == 0:
                break


def plot_rec_camera_params(
        reconstruction: Reconstruction,
        save_dir: Path = None,
        x_label: str = 'time',
):
    """
    Plot the camera parameters for a reconstruction.
    """
    assert reconstruction.source == M3D_SOURCE_MF, 'Only MF reconstructions work for this!'
    logger.info(f'Plotting camera parameters for trial {reconstruction.trial.id} (rec={reconstruction.id}).')

    if save_dir is None:
        save_dir = LOGS_PATH / (f'{START_TIMESTAMP}'
                                f'_trial={target.trial.id:03d}'
                                f'_{target.id}_params')
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)

    ts = TrialState(
        reconstruction=reconstruction,
        start_frame=reconstruction.start_frame_valid,
        end_frame=reconstruction.end_frame_valid + 1
    )

    N = len(ts)
    if x_label == 'time':
        x = np.linspace(0, N / ts.trial.fps, N)
    else:
        x = np.arange(N) + ts.start_frame

    def _make_plot(axes_: np.ndarray, data: np.ndarray, title: str):
        for c in range(3):
            ax = axes_[c]
            ax.set_title(f'{title} - cam{c}')
            ax.plot(x, data[c])
            if x_label == 'time':
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('Frame #')

    for k in CAM_PARAMETER_KEYS:
        p = _get_camera_parameters(ts, k)
        data_k, titles_k = p['data'], p['titles']
        n_plots = len(p['titles'])

        fig, axes = plt.subplots(n_plots, 3, figsize=(10, n_plots * 3), sharex=True, squeeze=False)
        fig.suptitle(k)

        for i in range(n_plots):
            _make_plot(axes[i], data_k[i], titles_k[i])

        fig.tight_layout()

        if save_plots:
            path = save_dir / f'{k}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()


def plot_rec_projection_examples(
        reconstruction: Reconstruction,
        frame_nums: List[int],
        save_dir: Path = None,
        midline_width: int = 1,
        crop_size: int = -1
):
    """
    Plot projected midlines using camera parameters from different points.
    """
    if save_dir is None:
        save_dir = LOGS_PATH / (f'{START_TIMESTAMP}'
                                f'_trial={target.trial.id:03d}'
                                f'_{target.id}_examples')
        if save_plots:
            os.makedirs(save_dir, exist_ok=True)

    N = reconstruction.mf_parameters.n_points_total
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = cmap(np.linspace(0, 1, N))
    colours = np.round(colours * 255).astype(np.uint8)

    ts = TrialState(reconstruction)
    points_3d = ts.get('points')
    points_3d_base = ts.get('points_3d_base')
    points_2d_base = ts.get('points_2d_base')
    cam_coeffs = np.concatenate([
        ts.get(f'cam_{k}')
        for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
    ], axis=2)
    shifts = ts.get('cam_shifts')
    distortions = ts.get('cam_distortions').astype(np.float64)
    prs = ProjectRenderScoreModel(image_size=reconstruction.trial.crop_size)

    if save_plots:
        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            midline_width=midline_width,
            crop_size=crop_size
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for frame_num in frame_nums:
        # Fetch the images and midline for the frame
        frame = reconstruction.trial.get_frame(frame_num)
        rw = RenderWrapper(reconstruction, frame)
        dm = rw.get_detection_masks()
        masked_images = (frame.images * dm).clip(min=0, max=1)
        p3d = torch.from_numpy(points_3d[frame_num][None, ...])
        p3d_base = torch.from_numpy(points_3d_base[frame_num][None, ...].astype(np.float32))
        p2d_base = torch.from_numpy(points_2d_base[frame_num][None, ...].astype(np.float32))

        # Project the midline using the camera models from all other frames
        for n in frame_nums:
            cc = torch.from_numpy(cam_coeffs[n][None, ...])
            cc[:, :, -1] = torch.from_numpy(shifts[frame_num][None, :, 0])
            dist = torch.from_numpy(distortions[n][None, :])
            dist = torch.from_numpy(
                np.random.normal(
                    # loc=distortions.mean(axis=0).mean(axis=0),
                    loc=dist[0],
                    scale=10 * np.sqrt(distortions.var(axis=0).mean(axis=0)),
                    size=(3, 5)
                )
            ).to(torch.float32)
            cc[:, :, -6:-1] = dist

            points_2d = prs._project_to_2d(
                # cam_coeffs=torch.from_numpy(cam_coeffs[n][None, ...]),
                cam_coeffs=cc,
                points_3d=p3d,
                points_3d_base=p3d_base,
                points_2d_base=p2d_base,
            )
            points_2d = np.round(to_numpy(points_2d[0])).astype(np.int32)

            for c, img_array in enumerate(masked_images):
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

                # Convert to PIL image
                img = Image.fromarray(z, 'RGBA')

                # Crop
                if -1 < crop_size < img.size[0]:
                    m = int(img.size[0] - crop_size) / 2  # margin to remove
                    img = img.crop(box=(m, m, img.size[0] - m, img.size[1] - m))

                if save_plots:
                    save_path = save_dir / f'midline_{frame_num:05d}_params_{n:05d}_c{c}.png'
                    logger.info(f'Saving image to {save_path}.')
                    img.save(save_path)

                if show_plots:
                    img.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    target = _get_targets()

    if type(target) == Reconstruction:
        # plot_rec_camera_params(
        #     reconstruction=target,
        #     x_label='time'
        # )

        plot_rec_projection_examples(
            reconstruction=target,
            frame_nums=[0, 5000, 10000],
            midline_width=2,
            crop_size=-1
        )

    else:
        save_dir_ = LOGS_PATH / (f'{START_TIMESTAMP}'
                                 f'_ds={target.id}')
        if save_plots:
            os.makedirs(save_dir_, exist_ok=True)
        # plot_ds_camera_params(
        #     dataset=target,
        #     save_dir=save_dir_,
        #     x_label='time'
        # )

        scan_ds_camera_params(target)
