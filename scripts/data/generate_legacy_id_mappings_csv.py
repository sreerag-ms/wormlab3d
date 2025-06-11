from wormlab3d import LOGS_PATH
from wormlab3d.data.model import Trial


def main():
    data = []

    for trial in Trial.objects:
        data.append((trial.id, trial.legacy_id))

    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    with open(LOGS_PATH / 'legacy_id_mappings.csv', 'w') as f:
        f.write('trial_id,legacy_id\n')
        for trial_id, legacy_id in data:
            f.write(f'{trial_id},{legacy_id}\n')



if __name__ == '__main__':
    main()
