import os

import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args


def save_trajectory():
    args = get_args()
    X = get_trajectory_from_args(args)
    os.makedirs(LOGS_PATH, exist_ok=True)
    np.save(LOGS_PATH / f'{START_TIMESTAMP}_trial={args.trial}_frames={args.start_frame}-{args.end_frame}_{args.midline3d_source}', X)


def load_trajectory():
    X = np.load(LOGS_PATH / '20220502_1650_37_5335-5583_reconst.npy')
    return



if __name__=='__main__':
    save_trajectory()
    # load_trajectory()
