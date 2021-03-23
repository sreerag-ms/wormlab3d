import csv
import datetime
import os
from typing import List

import cv2
import dateutil
import numpy as np
import scipy.io as sio
from mongoengine import DoesNotExist

from wormlab3d import WT3D_PATH, logger
from wormlab3d.data.model.cameras import Cameras
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.midline3d import Midline3D
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trial import Trial, CAMERA_IDXS
from wormlab3d.data.util import ANNEX_PATH_PLACEHOLDER

HOME_DIR = os.path.expanduser('~')
DATA_DIR = HOME_DIR + '/projects/worm_data'
VIDEO_DIR = 'video'
CALIB_DIR = 'calib'
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
        logger.info('\n\n=== ' + k)
        logger.info(values[k])


def clear_db():
    Tag.drop_collection()
    Experiment.drop_collection()
    Trial.drop_collection()
    Frame.drop_collection()
    Midline2D.drop_collection()
    Midline3D.drop_collection()
    Cameras.drop_collection()


def find_or_create_experiment(row: dict) -> Experiment:
    id_old = int(row['exp_id'])
    try:
        # Try to find existing experiment
        experiment = Experiment.objects.get(legacy_id=id_old)
        return experiment
    except DoesNotExist:
        pass

    # Create new experiment
    logger.info(f'Creating experiment (legacy_id={id_old})')
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

    # Look for calibration file like 025.xml
    calib_path = f'{DATA_DIR}/{CALIB_DIR}/{int(row["#id"]):03d}.xml'
    if os.path.exists(calib_path) or os.path.lexists(calib_path):
        logger.info(f'Found calibration file: {calib_path}')
        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        cams = Cameras()
        cams.experiment = experiment

        try:
            cams.timestamp = dateutil.parser.parse(fs.getNode('calibration_time').string())
            cams.wormcv_version = fs.getNode('WormCV_Version').string()
            cams.opencv_version = fs.getNode('OpenCV_Version').string()
            cams.opencv_contrib_hash = fs.getNode('OpenCV_contrib_hash').string()
            cams.total_calib_images = int(fs.getNode('total_filenames').real())
            if cams.total_calib_images == 0 and fs.getNode('image_filenames').size() > 0:
                cams.total_calib_images = fs.getNode('image_filenames').size()
            cams.pattern_height = float(fs.getNode('pattern_height').real())
            cams.pattern_width = float(fs.getNode('pattern_width').real())
            cams.square_size = float(fs.getNode('square_size').real())
            cams.flag_value = int(fs.getNode('flag_value').real())
            cams.n_mini_matches = int(fs.getNode('n_mini_matches').real())
            cams.n_cameras = int(fs.getNode('nCameras').real())
            cams.camera_type = int(fs.getNode('camera_type').real())
            cams.reprojection_error = float(fs.getNode('reprojection_error').real())
            cams.n_images_used = [int(fs.getNode(f'images_used_{c}').real()) for c in CAMERA_IDXS]
            cams.pose = [fs.getNode(f'camera_pose_{c}').mat() for c in CAMERA_IDXS]
            cams.matrix = [fs.getNode(f'camera_matrix_{c}').mat() for c in CAMERA_IDXS]
            cams.distortion = [fs.getNode(f'camera_distortion_{c}').mat() for c in CAMERA_IDXS]
            cams.save()
        except Exception:
            logger.error(f'Could not parse calibration file: {calib_path}')

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
    for c in CAMERA_IDXS:
        location = f'{VIDEO_DIR}/{int(row["#id"]):03d}_{c}.avi'
        vid_path = DATA_DIR + '/' + location
        if os.path.exists(vid_path) or os.path.lexists(vid_path):
            setattr(trial, f'camera_{c}_avi', f'{ANNEX_PATH_PLACEHOLDER}/{location}')
        else:
            raise RuntimeError(f'Video file not present "{vid_path}"')

    # Look for background image files like 025_1.png
    for c in CAMERA_IDXS:
        location = f'{BACKGROUND_IMAGES_DIR}/{int(row["#id"]):03d}_{c}.avi'
        bg_path = DATA_DIR + '/' + location
        if os.path.exists(bg_path) or os.path.lexists(bg_path):
            setattr(trial, f'camera_{c}_background', f'{ANNEX_PATH_PLACEHOLDER}/{location}')

    trial.save()

    return trial


def find_or_create_frames(trial: Trial) -> List[Frame]:
    if trial.num_frames == 0:
        return

    # Check for existing frames
    existing = trial.get_frames()
    if len(existing) > 0:
        assert len(existing) == trial.num_frames
        return

    # Create new empty frames
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
            logger.debug(row)

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

    logger.info(f'\n\n==== skipped_rows ({len(skipped_rows)}) ====')
    for r, e in skipped_rows:
        logger.info(f'\n{e}:')
        logger.info(r)


def migrate_midlines2d():
    midlines = []
    files = os.listdir(MIDLINES_2D_DIR)
    failed = []
    logger.info(f'{len(files)} files found.')
    for i, filename in enumerate(files):
        logger.info(f'Processing file {i + 1}/{len(files)}: {filename}')
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
    if len(midlines) > 0:
        logger.info(f'Inserting {len(midlines)} 2D midlines')
        Midline2D.objects.insert(midlines)
    else:
        logger.error('No 2D midlines could be migrated!')

    # Show any failures
    if len(failed):
        logger.error('\n\n=== FAILED ===')
        logger.error(failed)



def migrate_WT3D():
    # Project *.mat files
    path = WT3D_PATH + '/Project_Files'

    files = os.listdir(path)
    for i, filename in enumerate(files):
        logger.info(f'Processing file {i + 1}/{len(files)}: {filename}')
        if not filename.endswith('.mat'):
            continue
        mat = sio.loadmat(path + '/' + filename)
        mat = mat['Project_3DWT']

        # Top level keys: 'Camera_Parameters', 'Tracing_Parameters', 'Trace', 'Behavior', 'Info'

        # Find trial and experiment
        filename_parts = filename.split('_')
        legacy_id = filename_parts[2] + '_' + filename_parts[3][:-4]
        trial = Trial.objects.get(legacy_data__legacy_id=legacy_id)
        experiment = trial.experiment

        # todo: verify other trial info all matches up

        # Tags (Behavior)
        if 0:
            tag_entries = mat['Behavior'][0][0]
            n_entries = tag_entries.shape[1]
            if n_entries > 0:
                for j in range(n_entries):
                    # Get tag objects
                    tags = []
                    tag_ids = tag_entries[0]['Tags'][j][0]
                    for tag_id in tag_ids:
                        tags.append(Tag.objects.get(id=tag_id))

                    # Attach tags to all frames in range
                    first_frame = tag_entries[0]['First_Frame'][j][0][0]
                    last_frame = tag_entries[0]['Last_Frame'][j][0][0]
                    logger.info(f'Migrating {j + 1}/{n_entries} tag entries. '
                                f'Adding {len(tags)} tags to frames {first_frame} - {last_frame}.')
                    Frame.objects(
                        trial=trial,
                        frame_num__gte=first_frame,
                        frame_num__lte=last_frame
                    ).update(
                        set__tags=tags
                    )

        # 2D midline annotations are stored in the camera parameters as annotations
        if 0:
            annotations = mat['Camera_Parameters'][0][0]['Calib'][0][0]['Annotations']

            logger.info(
                mat['Camera_Parameters'].shape,
                mat['Camera_Parameters'][0].shape,
                mat['Camera_Parameters'][0][0].shape,
                mat['Camera_Parameters'][0][0]['Calib'].shape,
                mat['Camera_Parameters'][0][0]['Calib'][0].shape,
                mat['Camera_Parameters'][0][0]['Calib'][0][0].shape,
                mat['Camera_Parameters'][0][0]['Calib'][0][0]['Annotations'].shape,
            )
            n_midlines = annotations.shape[-1]
            logger.info(f'n_midlines={n_midlines}')

            # Annotation keys: 'Frame_Number', 'X', 'Y', 'Intrinsic_Matrix', 'Radial_Distortion_Vector', 'Tangential_Distortion_Vector', 'Extrinsic_Matrix'
            for n in range(n_midlines):
                annotation = annotations[0][n]
                x = annotation['X'][0][0][0]
                y = annotation['Y'][0][0][0]
                X = np.stack([x, y]).T
                # logger.info(X.shape)
                # logger.info(np.isnan(X.any()))

                # Get frame
                frame_num = annotation['Frame_Number'][0][0][0][0]
                frame = trial.get_frame(frame_num)

                # Get any existing annotations
                exists = False
                existing = frame.get_midlines2d(manual_only=True)
                for mid in existing:
                    if np.allclose(X, mid.X):
                        # Midline matches existing record
                        exists = True
                        break

                if not exists:
                    midline = Midline2D()
                    midline.frame = frame
                    midline.camera = '?'
                    midline.user = 'YO'
                    midline.X = X
                    # midline.save()


if __name__ == '__main__':
    clear_db()
    # print_runinfo_data()
    migrate_tags()
    migrate_runinfo()
    migrate_midlines2d()

    # migrate_WT3D()
