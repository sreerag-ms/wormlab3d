import csv
import datetime
import os

import cv2
import dateutil
import h5py
import numpy as np
import scipy.io as sio
from mongoengine import DoesNotExist

from wormlab3d import WT3D_PATH, logger, CAMERA_IDXS, ANNEX_PATH, DATA_PATH
from wormlab3d.data.model import *
from wormlab3d.data.model.cameras import CAM_SOURCE_ANNEX, CAM_SOURCE_WT3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_RECONST, M3D_SOURCE_WT3D
from wormlab3d.data.util import ANNEX_PATH_PLACEHOLDER

VIDEO_DIR = 'video'
CALIB_DIR = 'calib'
CAMERA_SHIFTS_DIR = ANNEX_PATH + '/calib/shifts'
BACKGROUND_IMAGES_DIR = 'background'
MIDLINES_2D_DIR = ANNEX_PATH + '/midlines'
MIDLINES_3D_DIR = ANNEX_PATH + '/reconst'
TAGS_MAT_PATH = DATA_PATH + '/Behavior_Dictionary.mat'

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
    with open(ANNEX_PATH + '/run_info.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in fields:
                if row[k] not in values[k]:
                    values[k].append(row[k])
    for k in fields:
        logger.info('\n\n=== ' + k)
        logger.info(values[k])


def clear_db():
    # Drop collections
    Cameras.drop_collection()
    Experiment.drop_collection()
    Frame.drop_collection()
    Midline2D.drop_collection()
    Midline3D.drop_collection()
    Model.drop_collection()
    Tag.drop_collection()
    FrameSequence.drop_collection()
    Dataset.drop_collection()
    Trial.drop_collection()

    # Reset id sequences / counters
    Experiment.id.set_next_value(0)
    Model.id.set_next_value(0)
    Trial.id.set_next_value(0)


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
    trial.trial_num = int(row['trial_id'])
    trial.legacy_id = id_old
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
        'legacy_id': row['legacy_id'],
        'num_frames': row['num_frames'],
    }

    # Look for video files like 025_1.avi
    videos = []
    for c in CAMERA_IDXS:
        location = f'{VIDEO_DIR}/{int(row["#id"]):03d}_{c}.avi'
        vid_path = ANNEX_PATH + '/' + location
        if os.path.exists(vid_path) or os.path.lexists(vid_path):
            videos.append(f'{ANNEX_PATH_PLACEHOLDER}/{location}')
        else:
            raise RuntimeError(f'Video file not present "{vid_path}"')
    trial.videos = videos

    # Look for background image files like 025_1.png
    bgs = []
    for c in CAMERA_IDXS:
        location = f'{BACKGROUND_IMAGES_DIR}/{int(row["#id"]):03d}_{c}.png'
        bg_path = ANNEX_PATH + '/' + location
        if os.path.exists(bg_path) or os.path.lexists(bg_path):
            bgs.append(f'{ANNEX_PATH_PLACEHOLDER}/{location}')
    if len(bgs) == 3:
        trial.backgrounds = bgs
    elif len(bgs) > 0:
        logger.error(f'Missing some backgrounds, found {len(bgs)}')
    else:
        logger.warning('No backgrounds found')

    trial.save()

    return trial


def find_or_create_cameras(row: dict, experiment: Experiment) -> Cameras:
    # Look for calibration file like 025.xml
    calib_path = f'{ANNEX_PATH}/{CALIB_DIR}/{int(row["#id"]):03d}.xml'
    if os.path.exists(calib_path) or os.path.lexists(calib_path):
        logger.info(f'Found calibration file: {calib_path}')
        try:
            fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
            timestamp = dateutil.parser.parse(fs.getNode('calibration_time').string())
        except Exception:
            logger.error(f'Could not parse calibration file: {calib_path}')
            return

        trial_id_old = int(row['#id'])
        trial = None
        try:
            trial = Trial.objects.get(legacy_id=trial_id_old)
        except DoesNotExist:
            pass

        try:
            # Try to find existing cameras
            if trial is None:
                cams = Cameras.objects.get(experiment=experiment, timestamp=timestamp)
            else:
                try:
                    cams = Cameras.objects.get(experiment=experiment, trial=trial, timestamp=timestamp)
                except DoesNotExist:
                    cams = Cameras.objects.get(experiment=experiment, timestamp=timestamp)
                    cams.trial = trial
            cams.source = CAM_SOURCE_ANNEX
            cams.source_file = f'{int(row["#id"]):03d}.xml'
            cams.save()

            logger.debug(f'Found existing cameras, id={cams.id}')
            return cams
        except DoesNotExist:
            pass

        cams = Cameras()
        cams.source = CAM_SOURCE_ANNEX
        cams.source_file = f'{int(row["#id"]):03d}.xml'
        cams.experiment = experiment
        if trial is not None:
            cams.trial = trial

        try:
            cams.timestamp = timestamp
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
            cams.reprojection_error = float(fs.getNode('meanReprojectError').real())
            cams.n_images_used = [int(fs.getNode(f'images_used_{c}').real()) for c in CAMERA_IDXS]
            cams.pose = [fs.getNode(f'camera_pose_{c}').mat() for c in CAMERA_IDXS]
            cams.matrix = [fs.getNode(f'camera_matrix_{c}').mat() for c in CAMERA_IDXS]
            cams.distortion = [fs.getNode(f'camera_distortion_{c}').mat()[0] for c in CAMERA_IDXS]
            cams.save()
            logger.debug('Saved cameras')
            return cams

        except Exception:
            logger.error(f'Could not parse calibration file: {calib_path}')


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

    with open(ANNEX_PATH + '/run_info.csv') as f:
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
                cams = find_or_create_cameras(row, experiment)
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
        midline.camera = cam_num

        # Parse annotated midlines
        with open(MIDLINES_2D_DIR + '/' + filename) as f:
            X = []
            for line in f.readlines():
                if '# author:' in line:
                    midline.user = line[9:].strip()
                if line[0] == '#':
                    continue
                coords = np.array(list(float(c) for c in line.split(',')), dtype=np.float32)
                if len(coords) == 2 and not np.allclose(coords, np.array([0, 0])):
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
        logger.error('\n=== FAILED:' + '\n'.join(failed) + '\n')


def migrate_midlines3d(drop_collection=False):
    if drop_collection:
        logger.warning('Dropping Midline3D collection.')
        Midline3D.drop_collection()
    files = os.listdir(MIDLINES_3D_DIR)
    failed = []
    logger.info(f'{len(files)} files found.')
    for i, filename in enumerate(files):
        if not filename.endswith('.hdf5'):
            continue
        logger.info(f'Processing file {i + 1}/{len(files)}: {filename}')

        # Get trial
        trial_id = filename[:3]
        try:
            trial = Trial.objects.get(legacy_id=trial_id)
        except DoesNotExist:
            failed.append(filename)
            continue

        midlines = []

        try:
            f = h5py.File(MIDLINES_3D_DIR + '/' + filename, 'r')
        except (FileNotFoundError, OSError) as e:
            logger.error(f'Failed to open file: {e}')
            failed.append(filename)
            continue

        X = f.get('X')
        E = f.get('E')
        base_3d = f.get('base_3d')
        n_frames = X.shape[0]

        if n_frames > trial.n_frames_min:
            logger.error('More frames than possible!')
            failed.append(filename)
            continue

        if E.shape[0] != n_frames or base_3d.shape[0] != n_frames:
            logger.error('Numbers of frames inconsistent!')
            failed.append(filename)
            continue

        # Check for existing
        existing = Midline3D.objects(source_file=filename)
        check_missing = False
        if existing.count() > 0:
            logger.debug(f'Found {existing.count()} existing midlines.')
            if existing.count() != n_frames:
                logger.warning('Numbers from file don\'t match database, adding any missing.')
                check_missing = True
            else:
                continue

        for frame_num in range(n_frames):
            frame = trial.get_frame(frame_num)
            if check_missing:
                try:
                    Midline3D.objects.get(source_file=filename, frame=frame)
                    continue
                except DoesNotExist:
                    pass
            midline3d = Midline3D()
            midline3d.frame = frame
            midline3d.X = X[frame_num]
            midline3d.base_3d = base_3d[frame_num]
            midline3d.error = E[frame_num]
            midline3d.source = M3D_SOURCE_RECONST
            midline3d.source_file = filename
            # midline3d.validate()
            midlines.append(midline3d)

        f.close()

        # Bulk insert
        if len(midlines) > 0:
            logger.info(f'Inserting {len(midlines)} 3D midlines.')
            Midline3D.objects.insert(midlines)
        else:
            logger.error('No 3D midlines could be migrated!')

    # Show any failures
    if len(failed):
        logger.error('\n=== FAILED:' + '\n'.join(failed) + '\n')


def migrate_shifts(drop_collection=False):
    if drop_collection:
        logger.warning('Dropping CameraShifts collection.')
        CameraShifts.drop_collection()
    files = os.listdir(CAMERA_SHIFTS_DIR)
    failed = []
    batch_size = 10  # shifts are calculated in batches of 10
    logger.info(f'{len(files)} files found.')
    for i, filename in enumerate(files):
        if not filename.endswith('_shift.out'):
            continue
        logger.info(f'Processing file {i + 1}/{len(files)}: {filename}')

        # Get trial
        try:
            trial_id = int(filename[:3])
        except ValueError:
            failed.append(filename)
            continue
        try:
            trial = Trial.objects.get(legacy_id=trial_id)
        except DoesNotExist:
            failed.append(filename)
            continue

        shifts = []

        # Parse shifts
        with open(CAMERA_SHIFTS_DIR + '/' + filename) as f:
            for line in f.readlines():
                if line[0] == '#':
                    continue

                start_frame, dx, dy, dz = line.split(' ')
                start_frame = int(start_frame)
                for frame_num in range(start_frame, start_frame + batch_size):
                    try:
                        frame = trial.get_frame(frame_num)
                    except DoesNotExist:
                        continue
                    try:
                        CameraShifts.objects.get(frame=frame)
                        continue  # unique
                    except DoesNotExist:
                        pass

                    cam_shifts = CameraShifts()
                    cam_shifts.frame = frame
                    cam_shifts.dx = float(dx)
                    cam_shifts.dy = float(dy)
                    cam_shifts.dz = float(dz)
                    # cam_shifts.validate()
                    shifts.append(cam_shifts)

        # Bulk insert
        if len(shifts) > 0:
            logger.info(f'Inserting {len(shifts)} camera shifts.')
            CameraShifts.objects.insert(shifts)
        else:
            logger.error('No shifts could be migrated!')

    # Show any failures
    if len(failed):
        logger.error('\n=== FAILED:' + '\n'.join(failed) + '\n')


def migrate_WT3D(
        update_tags: bool = False,
        update_midlines2d: bool = False,
        update_midlines3d: bool = False,
        update_cameras: bool = False
):
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
        if update_tags:
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
                    first_frame = tag_entries[0]['First_Frame'][j][0][0] - 1
                    last_frame = tag_entries[0]['Last_Frame'][j][0][0] - 1
                    logger.info(f'Migrating {j + 1}/{n_entries} tag entries. '
                                f'Adding {len(tags)} tags to frames {first_frame} - {last_frame}.')
                    Frame.objects(
                        trial=trial,
                        frame_num__gte=first_frame,
                        frame_num__lte=last_frame
                    ).update(
                        set__tags=tags
                    )

        if update_midlines2d:
            # 2D midline annotations are stored in the camera parameters as annotations
            annotations = mat['Camera_Parameters'][0][0]['Calib'][0][0]['Annotations']

            # Annotation keys: 'Frame_Number', 'X', 'Y', 'Intrinsic_Matrix', 'Radial_Distortion_Vector', 'Tangential_Distortion_Vector', 'Extrinsic_Matrix'

            n_annotations = annotations[0][0].shape[-1]
            logger.info(f'n_midlines={n_annotations}')
            for n in range(n_annotations):
                midlines = []

                for c in CAMERA_IDXS:
                    annotation = annotations[0][c]
                    x = annotation['X'][0][n][0]
                    y = annotation['Y'][0][n][0]
                    X = np.stack([x, y]).T

                    # Get frame
                    frame_num = annotation['Frame_Number'][0][n][0][0] - 1
                    try:
                        frame = trial.get_frame(frame_num)
                    except DoesNotExist:
                        logger.error('Could not find frame in database.')
                        continue

                    # Get any existing annotations
                    exists = False
                    existing = frame.get_midlines2d(manual_only=True, filters={'camera': c})
                    for mid in existing:
                        if X.shape == mid.X.shape and np.allclose(X, mid.X):
                            # Midline matches existing record so skip it
                            logger.debug('Midline already exists, skipping.')
                            exists = True
                            break
                    if exists:
                        continue

                    midline = Midline2D()
                    midline.frame = frame
                    midline.camera = c
                    midline.user = 'YO'
                    midline.X = X
                    midlines.append(midline)

                if len(midlines) > 0:
                    # Midline annotations should come in triplets, one for each camera view
                    assert len(midlines) == 3
                    for m1, m2 in zip(midlines, midlines):
                        assert m1.frame.frame_num == m2.frame.frame_num
                    for m in midlines:
                        m.save()

        # 3D midline reconstructions
        if update_midlines3d:
            trace = mat['Trace'][0][0]
            if len(trace) == 0:
                logger.debug('No 3D midlines found.')
                continue
            n_midlines = trace.shape[1]
            logger.debug(f'{n_midlines} midlines found.')
            midlines = []

            for j in range(n_midlines):
                trace_j = trace[0][j]
                frame_num = trace_j['Frame_Number'][0][0] - 1

                # Check curve
                X = trace_j['Curve_3D']

                # Ignore head-tail annotations
                if np.isnan(X[1:-1]).all():
                    continue

                # Check that we don't have any infs or nans
                try:
                    assert not np.isinf(X).any(), 'Curve contains infs!'
                    assert not np.isnan(X).any(), 'Curve contains nans!'
                except AssertionError as e:
                    logger.error(f'Invalid curve for frame={frame_num}: {e}')
                    continue

                # Get frame
                try:
                    frame = trial.get_frame(frame_num)
                except DoesNotExist:
                    logger.error(f'Could not find frame {frame_num} in database.')
                    continue

                # Get any existing midlines
                exists = False
                existing = frame.get_midlines3d(filters={'source': M3D_SOURCE_WT3D, 'source_file': filename})
                for mid in existing:
                    if X.shape == mid.X.shape and np.allclose(X, mid.X):
                        # Midline matches existing record so skip it
                        logger.debug('Midline already exists, skipping.')
                        exists = True
                        break
                if exists:
                    continue

                midline = Midline3D()
                midline.frame = trial.get_frame(frame_num)
                midline.X = X
                midline.source = M3D_SOURCE_WT3D
                midline.source_file = filename
                # midline.validate()
                midlines.append(midline)

            if len(midlines) > 0:
                Midline3D.objects.insert(midlines)
                logger.debug(f'{len(midlines)} inserted')
            else:
                logger.debug('0 valid midlines found')

        # Camera models
        if update_cameras:
            cameras_inserted = 0
            cameras_updated = 0

            def create_or_update_WT3D_camera(cam_data, frame=None):
                nonlocal cameras_inserted, cameras_updated

                # Check that camera model parameters exist
                try:
                    intrinsic_matrix = cam_data['Intrinsic_Matrix'][0]
                    assert len(intrinsic_matrix) == 3
                    assert all([intrinsic_matrix[c].shape == (3, 3) for c in CAMERA_IDXS])

                    extrinsic_matrix = cam_data['Extrinsic_Matrix'][0]
                    assert len(extrinsic_matrix) == 3
                    assert all([extrinsic_matrix[c].shape == (3, 4) for c in CAMERA_IDXS])

                    radial_distortion = cam_data['Radial_Distortion_Vector'][0]
                    assert len(radial_distortion) == 3
                    assert all([radial_distortion[c].shape[0] == 1 for c in CAMERA_IDXS])
                    assert all([radial_distortion[c].shape[1] >= 3 for c in CAMERA_IDXS])

                    tangential_distortion = cam_data['Tangential_Distortion_Vector'][0]
                    assert len(tangential_distortion) == 3
                    assert all([tangential_distortion[c].shape[0] == 1 for c in CAMERA_IDXS])
                    assert all([tangential_distortion[c].shape[1] >= 2 for c in CAMERA_IDXS])

                except ValueError:
                    return
                except AssertionError:
                    return

                existing_cams = Cameras.objects(
                    source=CAM_SOURCE_WT3D,
                    source_file=filename,
                    frame=frame
                )

                if existing_cams.count() == 1:
                    logger.debug('Checking/updating existing camera.')
                    cams = existing_cams[0]
                    is_new = False
                elif existing_cams.count() > 1:
                    logger.error('Multiple cameras found! (This shouldn\'t happen!)')
                    return
                else:
                    logger.debug('Existing camera not found, creating new.')
                    cams = Cameras()
                    is_new = True

                cams.source = CAM_SOURCE_WT3D
                cams.source_file = filename
                cams.experiment = experiment
                cams.trial = trial
                pose = list(extrinsic_matrix)
                cams.pose = [np.r_[pose[c], np.zeros((1, 4))] for c in CAMERA_IDXS]
                cams.matrix = list(intrinsic_matrix)

                # Reshape the distortion parameters into [k1, k2, p1, p2, k3]
                cams.distortion = [
                    np.array([
                        radial_distortion[c][0][0],
                        radial_distortion[c][0][1],
                        tangential_distortion[c][0][0],
                        tangential_distortion[c][0][1],
                        radial_distortion[c][0][2],
                    ])
                    for c in CAMERA_IDXS
                ]

                cams.save()
                if is_new:
                    cameras_inserted += 1
                else:
                    cameras_updated += 1

            # Create top-level cameras
            calib = mat['Camera_Parameters'][0][0]['Calib'][0][0]
            create_or_update_WT3D_camera(calib)

            # Other camera models may exist tuned for a specific frame
            annotations = calib['Annotations']
            n_annotations = annotations[0][0].shape[-1]
            logger.info(f'n_annotations={n_annotations}')
            for n in range(n_annotations):
                frame_num = annotations[0][0]['Frame_Number'][0][n] - 1
                try:
                    frame = trial.get_frame(frame_num)
                except DoesNotExist:
                    logger.error('Could not find frame in database.')
                    continue

                try:
                    cam_data = {
                        'Intrinsic_Matrix': [[annotations[0][c][0][n]['Intrinsic_Matrix'] for c in CAMERA_IDXS]],
                        'Extrinsic_Matrix': [[annotations[0][c][0][n]['Extrinsic_Matrix'] for c in CAMERA_IDXS]],
                        'Radial_Distortion_Vector': [
                            [annotations[0][c][0][n]['Radial_Distortion_Vector'] for c in CAMERA_IDXS]],
                        'Tangential_Distortion_Vector': [
                            [annotations[0][c][0][n]['Tangential_Distortion_Vector'] for c in CAMERA_IDXS]],
                    }
                except ValueError:
                    continue

                create_or_update_WT3D_camera(cam_data, frame)

            logger.info(f'{cameras_inserted} new cameras inserted.')
            logger.info(f'{cameras_updated} existing cameras updated.')


if __name__ == '__main__':
    # print_runinfo_data()
    # clear_db()
    # migrate_tags()
    # migrate_runinfo()
    # migrate_midlines2d()
    # migrate_midlines3d(drop_collection=False)
    # migrate_shifts(drop_collection=True)
    # exit()
    migrate_WT3D(
        update_tags=False,
        update_midlines2d=False,
        update_midlines3d=False,
        update_cameras=True,
    )
