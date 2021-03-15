import csv
import datetime
import os

from mongoengine import DoesNotExist

from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.trial import Trial

HOME_DIR = os.path.expanduser('~')
DATA_DIR = HOME_DIR + '/projects/worm_data'

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
    id_old = row['legacy_id']
    try:
        # Try to find existing trial
        trial = Trial.objects.get(legacy_id=id_old)
        return trial
    except DoesNotExist:
        pass

    # Create new trial
    trial = Trial()
    trial.date = datetime.datetime(year=int(id_old[:4]), month=int(id_old[4:6]), day=int(id_old[6:8]))
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
        '#id': row['#id']
    }
    # todo: files = ListField(EmbeddedDocumentField(File))

    trial.save()

    return trial


def migrate_runinfo():
    Experiment.drop_collection()
    Trial.drop_collection()
    skipped_rows = []

    with open(DATA_DIR + '/run_info.csv') as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f)

        for row in reader:
            print(row)

            if row['comments'] == 'ignore' or row['exp_id'] in ['', '  ']:
                # skip these dummy experiments
                skipped_rows.append(row)
                continue

            experiment = find_or_create_experiment(row)
            trial = find_or_create_trial(row, experiment)


if __name__ == '__main__':
    # print_runinfo_data()
    migrate_runinfo()
