import math
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.material_parameters import MP_DEFAULT_K
from simple_worm.util import estimate_K_from_x
from wormlab3d import logger
from wormlab3d.data.model import Trial, FrameSequence
from wormlab3d.simple_worm.args import FrameSequenceArgs
from wormlab3d.toolkit.util import build_target_arguments_parser


def generate_FS(trial: Trial, args: Namespace, save: bool = False):
    f0 = args.frame_num
    n_frames = math.ceil(args.duration * trial.fps)
    fn = f0 + n_frames
    frame_nums = range(f0, fn)
    seq = []
    for frame_num in frame_nums:
        frame = trial.get_frame(frame_num)
        filters = {'source': args.midline3d_source}
        if args.midline3d_source_file is not None:
            filters['source_file'] = args.midline3d_source_file
        midlines = frame.get_midlines3d(filters)
        if len(midlines) > 1:
            logger.info(
                f'Found {len(midlines)} 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}. Picking with lowest error..')
            midline = midlines[0]
        elif len(midlines) == 1:
            midline = midlines[0]
        else:
            raise RuntimeError(f'Could not find any 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}.')
        seq.append(midline)

    FS_db = FrameSequence()
    FS_db.set_from_sequence(seq)
    if save:
        FS_db.save()
        logger.info('Saved FS to database')

    return FS_db


def load_or_generate_FS(trial: Trial, args: Namespace, save: bool = False) -> FrameSequence:
    n_frames = math.ceil(args.duration * trial.fps)

    # Look for existing FS
    fsa = FrameSequenceArgs(
        trial=args.trial,
        start_frame=args.frame_num,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
    )
    FS_db = FrameSequence.find_from_args(fsa, n_frames)
    if FS_db.count() > 0:
        logger.info(f'Found {len(FS_db)} matching frame sequences in database, using most recent.')
        FS_db = FS_db[0]
    else:
        # Create a new one
        logger.info(f'Generating FS for frames {args.frame_num}-{args.frame_num + n_frames}.')
        FS_db = generate_FS(trial, args, save)
    return FS_db


def get_data(args: Namespace):
    assert args.trial is not None, 'Trial must be specified.'
    assert args.duration is not None, 'Duration must be specified.'
    trial = Trial.objects.get(id=args.trial)
    n_sample_frames = math.ceil(args.duration * trial.fps)

    def get_FS_numpy(save):
        FS_db = load_or_generate_FS(trial, args, save=save)
        x = FS_db.X.transpose(0, 2, 1)  # x.shape == (T, 3, N)
        psi = np.zeros((x.shape[0], x.shape[2]))
        FS = FrameSequenceNumpy(x=x, psi=psi, calculate_components=True)
        return FS

    FSs = []
    if args.frame_num is not None:
        FSs.append(get_FS_numpy(save=True))
    else:
        for i in range(args.n_samples):
            args.frame_num = np.random.choice(trial.n_frames_min - n_sample_frames)
            FSs.append(get_FS_numpy(save=False))

    return FSs


def do_estimation(FSs: List[FrameSequenceNumpy], args: Namespace):
    res = []
    for FS in FSs:
        K_est = estimate_K_from_x(FS, args.K0, verbosity=1)
        res.append(K_est)
    return np.array(res)


def plot_results(res, args):
    plt.hist(res, bins=20)
    plt.title(f'Trial={args.trial}. Num samples={args.n_samples}. Sample duration={args.duration:.2f}s. K0={args.K0}.')
    plt.ylabel('frequency')
    plt.xlabel('K_est')
    plt.show()


def estimate_K():
    """
    Estimate the K parameter for various settings.
    """
    parser = build_target_arguments_parser()
    parser.add_argument('--n-samples', type=int, default=1, help='How many random samples to draw.')
    parser.add_argument('--K0', type=float, default=MP_DEFAULT_K, help='Initial value of K for the optimiser.')
    args = parser.parse_args()

    FSs = get_data(args)
    res = do_estimation(FSs, args)
    plot_results(res, args)


if __name__ == '__main__':
    estimate_K()
