import csv
import datetime
import os
from typing import List

import numpy as np
import scipy.io as sio
from mongoengine import DoesNotExist

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.midline3d import Midline3D
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trial import Trial

HOME_DIR = os.path.expanduser('~')
DATA_DIR = HOME_DIR + '/projects/worm_data'
VIDEO_DIR = 'video'
BACKGROUND_IMAGES_DIR = 'background'
MIDLINES_2D_DIR = DATA_DIR + '/midlines'
TAGS_MAT_PATH = '../../data/Behavior_Dictionary.mat'

fields = [
    '#id',
    'legacy_id',
    'quality',
    'num_frames',
    'calibration_id',
    'time_sync',
    'exp_id',
    'trial_id',
    'concentration',
    'worm_length',
    'fps',
    'strain',
    'sex',
    'age',
    'temp',
    'magnification',
    'user',
    'comments'
]

values = {k: [] for k in fields}


def print_runinfo_data():
    with open(DATA_DIR + '/run_info.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in fields:
                if row[k] not in values[k]:
                    values[k].append(row[k])
    for k in fields:
        print('\n\n=== ' + k)
        print(values[k])


def clear_db():
    Tag.drop_collection()
    Experiment.drop_collection()
    Trial.drop_collection()
    Frame.drop_collection()
    Midline2D.drop_collection()
    Midline3D.drop_collection()


def find_or_create_experiment(row: dict) -> Experiment:
    id_old = int(row['exp_id'])
    try:
        # Try to find existing experiment
        experiment = Experiment.objects.get(legacy_id=id_old)
        return experiment
    except DoesNotExist:
        pass

    # Create new experiment
    experiment = Experiment()
    experiment.user = row['user'].strip()
    experiment.sex = row['sex'].strip()
    if row['age'] not in ['?', '', ' ']:
        experiment.age = row['age']
    if row['concentration'] not in ['?', '']:
        experiment.concentration = float(row['concentration'])
    if row['worm_length'] not in ['?', '']:
        experiment.worm_length = float(row['worm_length'])
    if row['strain'] not in ['?', '']:
        experiment.strain = row['strain'].strip()
    experiment.legacy_id = id_old
    experiment.save()

    return experiment


def find_or_create_trial(row: dict, experiment: Experiment) -> Trial:
    id_old = int(row['#id'])
    try:
        # Try to find existing trial
        trial = Trial.objects.get(legacy_id=id_old)
        return trial
    except DoesNotExist:
        pass

    # Create new trial
    trial = Trial()
    long_id = row['legacy_id']
    trial.date = datetime.datetime(year=int(long_id[:4]), month=int(long_id[4:6]), day=int(long_id[6:8]))
    trial.experiment = experiment
    trial.legacy_id = id_old
    if row['num_frames'] not in ['', '?']:
        trial.num_frames = int(row['num_frames'])
    if row['fps'] not in ['', '?', '??']:
        trial.fps = float(row['fps'])
    if row['quality'] not in ['', '?']:
        trial.quality = float(row['quality'])
    if row['temp'] not in ['', '?']:
        trial.temperature = float(row['temp'])
    trial.comments = row['comments'].strip()
    trial.legacy_data = {
        'calibration_id': row['calibration_id'],
        'time_sync': row['time_sync'],
        'magnification': row['magnification'],
        'trial_id': row['trial_id'],
        'legacy_id': row['legacy_id']
    }

    # Look for video files like 025_1.avi
    for cam_num in range(3):
        location = f'{VIDEO_DIR}/{int(row["#id"]):03d}_{cam_num}.avi'
        vid_path = DATA_DIR + '/' + location
        if os.path.exists(vid_path) or os.path.lexists(vid_path):
            setattr(trial, f'camera_{cam_num + 1}_avi', f'$WORM_DATA$/{location}')
        else:
            raise RuntimeError(f'Video file not present "{vid_path}"')

    # Look for background image files like 025_1.png
    for cam_num in range(3):
        location = f'{BACKGROUND_IMAGES_DIR}/{int(row["#id"]):03d}_{cam_num}.avi'
        bg_path = DATA_DIR + '/' + location
        if os.path.exists(bg_path) or os.path.lexists(bg_path):
            setattr(trial, f'camera_{cam_num + 1}_background', f'$WORM_DATA$/{location}')

    trial.save()

    return trial


def find_or_create_frames(trial: Trial) -> List[Frame]:
    if trial.num_frames == 0:
        return

    frames = []
    for i in range(trial.num_frames):
        frame = Frame()
        frame.trial = trial
        frame.experiment = trial.experiment
        frame.frame_num = i
        frames.append(frame)
    Frame.objects.insert(frames)

    return frames


def find_or_create_midline() -> Midline3D:
    # Create new midline
    midline = Midline3D()

    midline.base_3d = np.array([-1, 5, 2.], dtype=np.float32)

    midline.save()

    return midline


def migrate_tags():
    # Map Omer's ontology matlab table
    mat = sio.loadmat(TAGS_MAT_PATH)

    key_map = {
        'ID': 'id',
        'Name': 'name',
        'Tag': 'short_name',
        'Symbol': 'symbol',
        'Definition': 'description',
    }

    for row in mat['Behavior_Dictionary'][0]:
        tag = Tag()
        for key_from, key_to in key_map.items():
            val = row[key_from].squeeze()
            if key_from == 'ID':
                val = int(val)
            else:
                val = str(val).strip()
            setattr(tag, key_to, val)
        tag.save()


def migrate_runinfo():
    skipped_rows = []

    with open(DATA_DIR + '/run_info.csv') as f:
        reader = csv.DictReader(f)

        for row in reader:
            print(row)

            # Skip dummy entries
            if row['comments'] == 'ignore':
                skipped_rows.append((row, 'comment set to "ignore"'))
                continue
            elif row['exp_id'] in ['', '  ']:
                skipped_rows.append((row, 'missing exp_id'))
                continue

            try:
                experiment = find_or_create_experiment(row)
                trial = find_or_create_trial(row, experiment)
                frames = find_or_create_frames(trial)
            except RuntimeError as e:
                skipped_rows.append((row, str(e)))

    print(f'\n\n==== skipped_rows ({len(skipped_rows)}) ====')
    for r, e in skipped_rows:
        print(f'\n{e}:')
        print(r)


def migrate_midlines2d():
    midlines = []
    files = os.listdir(MIDLINES_2D_DIR)
    failed = []
    print(f'{len(files)} files found.')
    for i, filename in enumerate(files):
        print(f'Processing file {i + 1}/{len(files)}: {filename}')
        if not filename.endswith('.csv'):
            continue
        midline = Midline2D()

        # Get frame
        clip_num, cam_num, frame_num = (int(p) for p in filename.strip('.csv').split('_'))
        try:
            trial = Trial.objects.get(legacy_id=clip_num)
            midline.frame = trial.get_frame(frame_num)
        except DoesNotExist:
            failed.append(filename)
            continue
        midline.camera = cam_num + 1

        # Parse annotated midlines
        with open(MIDLINES_2D_DIR + '/' + filename) as f:
            X = []
            for line in f.readlines():
                if '# author:' in line:
                    midline.user = line[9:].strip()
                if line[0] == '#':
                    continue
                coords = np.array(list(float(c) for c in line.split(',')), dtype=np.float32)
                if len(coords) == 2:
                    X.append(coords)
            X = np.stack(X)
            midline.X = X
            midline.validate()
            midlines.append(midline)

    # Bulk insert
    Midline2D.objects.insert(midlines)

    # Show any failures
    if len(failed):
        print('\n\n=== FAILED ===')
        print(failed)


if __name__ == '__main__':
    clear_db()
    # print_runinfo_data()
    migrate_tags()
    migrate_runinfo()
    migrate_midlines2d()
