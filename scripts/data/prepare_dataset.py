import json
import shutil
from argparse import ArgumentParser, Namespace
from csv import DictWriter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from wormlab3d import LOGS_PATH, ROOT_PATH, SCRIPT_PATH, START_TIMESTAMP, logger
from wormlab3d.data.annex import fetch_from_annex, is_annexed_file
from wormlab3d.data.model import Cameras, Dataset, Eigenworms, Reconstruction, Trial
from wormlab3d.data.util import fix_path
from wormlab3d.midlines3d.mf_methods import make_rotation_matrix
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.particles.tumble_run import generate_or_load_ds_statistics
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.util import print_args, str2bool, to_dict
from wormlab3d.trajectories.cache import get_trajectory


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to prepare a dataset for publication/sharing.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset by id.')
    parser.add_argument('--eigenworms', type=str, help='Eigenworms by id.')
    parser.add_argument('--include-videos', type=str2bool, default=True, help='Include videos.')
    parser.add_argument('--tracking-videos-path', type=Path, help='Path to the tracking videos.')

    args = parser.parse_args()

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            setattr(args, k, v)

    print_args(args)

    return args


def _init() -> Tuple[Namespace, Path, Dataset, Eigenworms]:
    """
    Initialise the arguments, save dir and load the dataset statistics.
    """
    args = get_args()

    # Fetch dataset
    ds = Dataset.objects.get(id=args.dataset)

    # Load eigenworms
    ew = generate_or_load_eigenworms(eigenworms_id=args.eigenworms, dataset_id=ds.id)

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_ds={ds.id}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    arg_dict = to_dict(args)
    arg_dict['created'] = START_TIMESTAMP
    arg_dict['eigenworms'] = str(ew.id)
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(arg_dict, f)

    return args, save_dir, ds, ew


def _get_camera_parameters(
        trial: Trial,
        rec: Reconstruction = None,
) -> Dict[str, list]:
    """
    Fetch particular camera parameters from the trial state.
    """
    cam_params = {}

    if rec is None:
        cameras: Cameras = trial.get_cameras()
        cam_params = {
            'pose': np.stack(cameras.pose).tolist(),  # (3, 4, 4)
            'matrix': np.stack(cameras.matrix).tolist(),  # (3, 3, 3)
            'distortion': np.stack(cameras.distortion).tolist(),  # (3, 5)
            'shifts': []
        }

    else:
        ts = TrialState(reconstruction=rec, partial_load_ok=True)

        # Build the pose matrices: (3, 4, 4)
        pose = np.zeros((3, 4, 4))
        pose[:, 3, 3] = 1.0  # Set the last row to [0, 0, 0, 1]

        # Construct the rotation matrices from the angles
        p = ts.get('cam_rotation_preangles')  # (T, 3, 3, 2)
        p = p.copy()[rec.start_frame_valid:rec.end_frame_valid].astype(np.float64)
        assert p.var(axis=0).max() < 1e-6, f'Camera rotation angles are not constant across time!'
        p = torch.from_numpy(p[0])  # (3, 3, 2)
        pose[:, :3, :3] = make_rotation_matrix(
            cos_phi=p[:, 0, 0],
            sin_phi=p[:, 0, 1],
            cos_theta=p[:, 1, 0],
            sin_theta=p[:, 1, 1],
            cos_psi=p[:, 2, 0],
            sin_psi=p[:, 2, 1],
        ).numpy().transpose(2, 0, 1)

        # Set the translation vectors
        p = ts.get('cam_translations')  # (T, 3, 3)
        p = p.copy()[rec.start_frame_valid:rec.end_frame_valid].astype(np.float64)
        assert p.var(axis=0).max() < 1e-5, f'Camera rotation angles are not constant across time!'
        pose[:, :3, 3] = p[0]  # (3, 3)
        cam_params['pose'] = pose.tolist()

        # Set the camera matrices: (3, 3, 3)
        matrix = np.zeros((3, 3, 3))
        intrinsics = ts.get('cam_intrinsics')  # (T, 3, 4)
        intrinsics = intrinsics.copy()[rec.start_frame_valid:rec.end_frame_valid].astype(np.float64)
        assert intrinsics.var(axis=0).max() < 1e-6, f'Camera intrinsics are not constant across time!'
        intrinsics = intrinsics[0]  # (3, 4)
        matrix[:, 0, 0] = intrinsics[:, 0]  # fx
        matrix[:, 1, 1] = intrinsics[:, 1]  # fy
        matrix[:, 0, 2] = intrinsics[:, 2]  # cx
        matrix[:, 1, 2] = intrinsics[:, 3]  # cy
        cam_params['matrix'] = matrix.tolist()

        # Set the distortion coefficients
        distortions = ts.get('cam_distortions')  # (T, 3, 5)
        distortions = distortions.copy()[rec.start_frame_valid:rec.end_frame_valid].astype(np.float64)
        if distortions.var(axis=0).max() > 1e-5:
            logger.warning('Camera distortion coefficients are not constant across time! Using the median.')
            distortions = np.median(distortions, axis=0)  # (3, 5)
        else:
            distortions = distortions[0]  # (3, 5)
        cam_params['distortion'] = distortions.tolist()

        # Set the shifts
        shifts = ts.get('cam_shifts')  # (T, 3, 1)
        shifts = shifts.copy().squeeze().astype(np.float64)
        shifts_valid = shifts[rec.start_frame_valid:rec.end_frame_valid]
        if rec.start_frame_valid > 0:
            shifts[:rec.start_frame_valid] = shifts_valid[0]
        if rec.end_frame_valid < len(shifts):
            shifts[rec.end_frame_valid:] = shifts_valid[-1]
        cam_params['shifts'] = shifts.tolist()

    return cam_params


def prepare_dataset():
    """
    Collate and prepare a dataset.
    """
    args, save_dir, ds, ew = _init()
    records = []

    # Make output dirs
    video_dir = save_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = save_dir / 'tracking'
    tracking_dir.mkdir(parents=True, exist_ok=True)
    cameras_dir = save_dir / 'cameras'
    cameras_dir.mkdir(parents=True, exist_ok=True)
    reconst_dir = save_dir / 'reconstruction_xyz'
    reconst_dir.mkdir(parents=True, exist_ok=True)
    eigenworms_dir = save_dir / 'reconstruction_eigenworms'
    eigenworms_dir.mkdir(parents=True, exist_ok=True)
    runtumble_dir = save_dir / 'run_tumble_approximations'
    runtumble_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = save_dir / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Save the eigenworms (means and components)
    shutil.copy(ew.components_path, eigenworms_dir / f'eigenworms_{ew.id}.npz')

    # Get the run and tumble approximations
    ds_stats = generate_or_load_ds_statistics(
        ds=ds,
        approx_method=args.approx_method,
        error_limits=[args.approx_error_limit],
        planarity_window=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.smoothing_window_K,
        smooth_K_first=args.smoothing_window_K,
        use_euler_angles=args.approx_use_euler_angles,
        min_run_speed_duration=(0, 10000),
    )
    trajectory_lengths = ds_stats[0]
    run_durations = ds_stats[1][0]
    run_speeds = ds_stats[2][0]
    planar_angles = ds_stats[3][0]
    nonplanar_angles = ds_stats[4][0]
    tumble_idxs = ds_stats[6][0]
    runs_start_idx = tumbles_start_idx = 0

    # Prepare the data for each trial
    for i, trial in enumerate(ds.include_trials):
        logger.info(f'Preparing data for trial={trial.id}.')
        experiment = trial.experiment
        rec_id = ds.get_reconstruction_id_for_trial(trial)
        rec = None if rec_id is None else Reconstruction.objects.get(id=rec_id)

        # Copy the videos
        if args.include_videos:
            logger.info('Copying videos.')
            video_dir_trial = video_dir / f'trial={trial.id:03d}'
            video_dir_trial.mkdir(parents=True, exist_ok=True)
            for j, video_path in enumerate(trial.videos):
                video_path = fix_path(video_path)
                if is_annexed_file(video_path):
                    fetch_from_annex(video_path, quiet=True)
                video_path = Path(video_path)
                assert video_path.exists(), f'Video not found: {video_path}.'
                shutil.copy(video_path, video_dir_trial / f'trial={trial.id:03d}_camera={j}{video_path.suffix}')

            # Copy the tracking videos
            if args.tracking_videos_path is not None:
                tracking_videos_dir = args.tracking_videos_path / f'trial={trial.id:03d}'
                assert tracking_videos_dir.exists(), f'Tracking videos not found: {tracking_videos_dir}.'
                for video_path in tracking_videos_dir.iterdir():
                    shutil.copy(video_path, video_dir_trial / video_path.name.replace(f'_camera', '_tracking_camera'))

        # Save the tracking data
        Xt, _ = get_trajectory(trial_id=trial.id, tracking_only=True)
        if Xt.ndim == 3:
            Xt = Xt.mean(axis=1)
        np.savez_compressed(tracking_dir / f'trial={trial.id:03d}_tracking.npz', X=Xt)

        # Save the camera configurations
        cam_params = _get_camera_parameters(trial, rec)
        with open(cameras_dir / f'trial={trial.id:03d}_camera_parameters.json', 'w') as f:
            json.dump(cam_params, f, indent=4)

        # Save the reconstruction data
        if rec_id is not None:
            Xr, _ = get_trajectory(reconstruction_id=rec_id)
            if rec.n_frames_valid != len(Xr):
                if rec.n_frames_valid == len(Xr) - 1:
                    rec.end_frame_valid += 1
                elif rec.n_frames_valid == len(Xr) + 1:
                    rec.end_frame_valid -= 1
                else:
                    raise RuntimeError(f'Expected {rec.n_frames_valid} reconstruction frames, got {len(Xr)}.')
            if Xr.ndim != 3:
                raise RuntimeError(f'Expected 3D trajectory, got shape={Xr.shape}.')
            np.savez_compressed(reconst_dir / f'trial={trial.id:03d}_reconstruction={rec_id}.npz', X=Xr)

            # Save the eigenworms embeddings
            Z, _ = get_trajectory(reconstruction_id=rec_id, natural_frame=True)
            Xe = ew.transform(np.array(Z))
            np.savez_compressed(eigenworms_dir / f'trial={trial.id:03d}_reconstruction={rec_id}_eigenworms={ew.id}.npz',
                                X=Xe)

        # Save the run and tumble approximation
        assert trajectory_lengths[i] == len(Xt), f'Expected {len(Xt)} frames, got {trajectory_lengths[i]}.'
        runs_end_idx = runs_start_idx + len(tumble_idxs[i]) - 1
        tumbles_end_idx = tumbles_start_idx + len(tumble_idxs[i])
        approx = {
            'tumble_idxs': tumble_idxs[i],
            'run_durations': run_durations[runs_start_idx:runs_end_idx].tolist(),
            'run_speeds': run_speeds[runs_start_idx:runs_end_idx].tolist(),
            'planar_angles': planar_angles[tumbles_start_idx:tumbles_end_idx].tolist(),
            'nonplanar_angles': nonplanar_angles[tumbles_start_idx:tumbles_end_idx].tolist(),
        }
        runs_start_idx = runs_end_idx
        tumbles_start_idx = tumbles_end_idx
        with open(runtumble_dir / f'trial={trial.id:03d}_approximation.json', 'w') as f:
            json.dump(approx, f, indent=4)

        # Add record
        record = {
            'trial': trial.id,
            'date': f'{trial.date:%Y-%m-%d}',
            'strain': experiment.strain,
            'sex': experiment.sex,
            'age': experiment.age,
            'concentration': experiment.concentration,
            # 'temperature': trial.temperature,
            'n_frames': len(Xt),
            'fps': f'{trial.fps:.2f}',
            'duration': f'{len(Xt) / trial.fps:.2f}',
            # 'tracking': True,
            'reconstruction': rec_id if rec_id is not None else '-',
            'r_start_frame': rec.start_frame_valid if rec is not None else '-',
            'r_end_frame': rec.end_frame_valid if rec is not None else '-',
            'n_reconstruction_frames': len(Xr) if rec is not None else '-',
        }
        records.append(record)

    # Save the records
    csv_file = save_dir / 'dataset.csv'
    with open(csv_file, 'w') as file:
        fieldnames = records[0].keys()  # Assuming all dictionaries have the same keys
        writer = DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    # Copy over the README, cpca class and the example scripts
    assets_dir = SCRIPT_PATH / 'dataset_assets'
    shutil.copy(assets_dir / 'README', save_dir)
    shutil.copy(ROOT_PATH / 'wormlab3d' / 'postures' / 'cpca.py', examples_dir / 'cpca.py')
    for record in records:
        if record['reconstruction'] == '-':
            continue
        break
    for script_name in ['plot_approximation.py', 'plot_eigenworms.py', 'plot_projection.py', 'plot_reconstruction.py',
                        'plot_tracking.py']:
        script_src = assets_dir / script_name
        with open(script_src, 'r') as f:
            content = f.read()
            content = content.replace('%%EIGENWORMS_ID%%', str(ew.id))
            content = content.replace('%%TRIAL_ID%%', f'{record["trial"]:03d}')
            content = content.replace('%%RECONSTRUCTION_ID%%', record['reconstruction'])
        with open(examples_dir / script_name, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    prepare_dataset()
