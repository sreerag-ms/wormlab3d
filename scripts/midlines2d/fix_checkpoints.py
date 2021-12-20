import os

from mongoengine import DoesNotExist
from wormlab3d import ROOT_PATH, logger
from wormlab3d.data.model import Checkpoint

base_logs_path = ROOT_PATH + '/logs/scripts'

logs_paths = [
    ROOT_PATH + '/logs/scripts/midlines2d/train',
    ROOT_PATH + '/logs/scripts/midlines2d/train_coordinates',
    ROOT_PATH + '/logs/scripts/midlines2d/train_progressive_thinning',
    ROOT_PATH + '/logs/scripts/midlines3d/train',
    ROOT_PATH + '/logs/scripts/midlines3d/train_rotae',
]

for logs_path in logs_paths:
    for root, dirs, files in os.walk(logs_path):
        path = root.split(os.sep)
        if path[-1] == 'checkpoints':
            for file in files:
                cp_id = file[:-6]
                try:
                    cp = Checkpoint.objects.get(id=cp_id)
                except DoesNotExist:
                    logger.warning(f'Checkpoint id={cp_id} not found in database.')
                    continue

                cp_path = root[len(ROOT_PATH) + 1:] + '/' + file
                if cp.parameters_file == cp_path:
                    logger.info(f'Checkpoint parameters path "{cp_path}" matches database record.')
                else:
                    logger.warning(f'Checkpoint parameters path "{cp_path}" not matching database record.')
                    cp.parameters_file = cp_path
                    cp.save()
