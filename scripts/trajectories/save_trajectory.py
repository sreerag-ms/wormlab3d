import os

import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.eigenworms import fetch_eigenworms
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args


def save_trajectory():
    args = get_args()
    X, meta = get_trajectory_from_args(args, return_meta=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    reconstruction = Reconstruction.objects.get(id=meta['reconstruction'])

    fn = f'{START_TIMESTAMP}' \
         f'_trial={reconstruction.trial.id}' \
         f'_frames={meta["start_frame"]}-{meta["end_frame"]}' \
         f'_{reconstruction.source}' \
         f'_{reconstruction.id}'

    np.save(LOGS_PATH / fn, X)

    if args.eigenworms is not None:
        ew = fetch_eigenworms(args.eigenworms)
        args.natural_frame = True
        Z = get_trajectory_from_args(args, natural_frame=True)
        np.save(LOGS_PATH / (fn + '_nf'), Z)
        X_ew = ew.transform(Z)
        np.save(LOGS_PATH / (fn + '_ew'), X_ew)


def load_trajectory():
    X = np.load(LOGS_PATH / '20220812_1522_trial=164_frames=0-500_reconst_61a8ee924ae7c48a65d0a352_nf.npy')
    return


if __name__ == '__main__':
    # save_trajectory()
    load_trajectory()
