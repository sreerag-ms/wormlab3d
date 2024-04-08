import shutil
from argparse import ArgumentParser, Namespace
from csv import DictWriter
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

from wormlab3d import LOGS_PATH, ROOT_PATH, SCRIPT_PATH, START_TIMESTAMP, logger
from wormlab3d.data.annex import fetch_from_annex, is_annexed_file
from wormlab3d.data.model import Dataset, Eigenworms, Reconstruction
from wormlab3d.data.util import fix_path
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

    args = parser.parse_args()
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
    reconst_dir = save_dir / 'reconstruction_xyz'
    reconst_dir.mkdir(parents=True, exist_ok=True)
    eigenworms_dir = save_dir / 'reconstruction_eigenworms'
    eigenworms_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = save_dir / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Save the eigenworms (means and components)
    shutil.copy(ew.components_path, eigenworms_dir / f'eigenworms_{ew.id}.npz')

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

        # Save the tracking data
        Xt, _ = get_trajectory(trial_id=trial.id, tracking_only=True)
        if Xt.ndim == 3:
            Xt = Xt.mean(axis=1)
        np.savez_compressed(tracking_dir / f'trial={trial.id:03d}_tracking.npz', X=Xt)

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

        # Add record
        record = {
            'trial': trial.id,
            'date': f'{trial.date:%Y-%m-%d}',
            'strain': experiment.strain,
            'sex': experiment.sex,
            'age': experiment.age,
            'concentration': experiment.concentration,
            'temperature': trial.temperature,
            'n_frames': len(Xt),
            'fps': f'{trial.fps:.2f}',
            'duration': f'{len(Xt) / trial.fps:.2f}',
            'tracking': True,
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

    # Copy over the example scripts and replace placeholders
    shutil.copytree(SCRIPT_PATH / 'dataset_examples', examples_dir, dirs_exist_ok=True)
    shutil.copy(ROOT_PATH / 'wormlab3d' / 'postures' / 'cpca.py', examples_dir / 'cpca.py')
    plot_tracking_script = examples_dir / 'plot_tracking.py'
    plot_reconstruction_script = examples_dir / 'plot_reconstruction.py'
    plot_eigenworms_script = examples_dir / 'plot_eigenworms.py'
    for record in records:
        if record['reconstruction'] == '-':
            continue
        break
    for script in [plot_tracking_script, plot_reconstruction_script, plot_eigenworms_script]:
        with open(script, 'r') as f:
            content = f.read()
            content = content.replace('%%EIGENWORMS_ID%%', str(ew.id))
            content = content.replace('%%TRIAL_ID%%', f'{record["trial"]:03d}')
            content = content.replace('%%RECONSTRUCTION_ID%%', record['reconstruction'])
        with open(script, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    prepare_dataset()
