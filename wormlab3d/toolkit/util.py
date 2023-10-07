import distutils.util
import hashlib
import json
from argparse import ArgumentParser, Namespace
from json import JSONEncoder
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
import torch
from numpy.linalg import norm

from wormlab3d import logger, CAMERA_IDXS

if TYPE_CHECKING:
    from wormlab3d.data.model import Trial


def build_target_arguments_parser() -> ArgumentParser:
    """
    Generic command line parser for multiple scripts to avoid repetition.
    """
    from wormlab3d.data.model.midline3d import M3D_SOURCE_RECONST, M3D_SOURCES
    parser = ArgumentParser(description='Wormlab3D script.')
    parser.add_argument('--experiment', type=int,
                        help='Experiment by id.')
    parser.add_argument('--trial', type=int,
                        help='Trial by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--camera', type=int,
                        help='Camera by index.')
    parser.add_argument('--frame-num', type=int,
                        help='Frame number.')
    parser.add_argument('--duration', type=float,
                        help='Duration in seconds.')
    parser.add_argument('--order-trials', type=str,
                        help='Specify some ordering for multiple trials.')
    parser.add_argument('--midline2d', type=str,
                        help='Midline2D id.')
    parser.add_argument('--midline3d', type=str,
                        help='Midline3D id.')
    parser.add_argument('--midline3d-source', type=str, default=M3D_SOURCE_RECONST, choices=M3D_SOURCES,
                        help='Midline3D source.')
    parser.add_argument('--midline3d-source-file', type=str,
                        help='Midline3D source file.')
    parser.add_argument('--frame-sequence', type=str,
                        help='FrameSequence id.')
    parser.add_argument('--sw-run', type=str,
                        help='SwRun id.')
    parser.add_argument('--sw-checkpoint', type=str,
                        help='SwCheckpoint id.')
    return parser


def parse_target_arguments() -> Namespace:
    """
    Parse command line arguments and build parameter holders.
    This is used for multiple scripts to avoid repetition.
    """
    parser = build_target_arguments_parser()
    args = parser.parse_args()

    return args


def resolve_targets(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None
) -> Tuple[List['Trial'], List[int]]:
    """
    Resolves any combination of passed experiments, trials and cameras.
    Function arguments take priority over command-line arguments so this method can be called multiple times.
    """
    from wormlab3d.data.model import Trial
    args = parse_target_arguments()

    # Override command line args with function arguments
    if experiment_id is not None:
        args.experiment = experiment_id
    if trial_id is not None:
        args.trial = trial_id
    if camera_idx is not None:
        args.camera = camera_idx
    if frame_num is not None:
        args.frame_num = frame_num
    print_args(args)

    # Get trials
    if args.trial is not None:
        trial = Trial.objects.get(id=args.trial)
        if args.experiment is not None:
            assert trial.experiment.id == args.experiment, 'Trial is not part of target experiment!'
        trials = [trial]
    else:
        if args.experiment is not None:
            trials = Trial.objects(experiment=args.experiment)
        else:
            trials = Trial.objects
        if args.order_trials is not None:
            trials = trials.order_by(args.order_trials)

    # Camera indices
    cam_idxs = CAMERA_IDXS
    if args.camera is not None:
        assert trial_id is not None
        assert args.camera in cam_idxs
        cam_idxs = [args.camera]

    # Frame number - only makes sense in combination with a trial_id
    if args.frame_num is not None:
        assert args.trial is not None
        assert 0 <= args.frame_num < trial.n_frames_max

    return trials, cam_idxs


def print_args(args: Namespace):
    """Logs all the keys and values present in an argument namespace."""
    log = '--- Arguments ---\n'
    for arg_name, arg_value in vars(args).items():
        log += f'{arg_name}: {arg_value}\n'
    log += '-----------------\n'
    logger.info(log)


class NumpyCompatibleJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic) or isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def hash_data(data) -> str:
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(
        json.dumps(data, sort_keys=True, cls=NumpyCompatibleJSONEncoder).encode('utf-8')
    ).hexdigest()


def is_bad(t: torch.Tensor):
    """Checks if any of the elements in the tensor are infinite or nan."""
    if torch.isinf(t).any():
        return True
    if torch.isnan(t).any():
        return True
    return False


def to_numpy(t: torch.Tensor):
    """Returns a numpy version of a given tensor, accounting for grads and devices."""
    return t.detach().cpu().numpy()


def to_dict(obj) -> dict:
    """Returns a dictionary representation of an object"""
    # Get any attrs defined on the instance (self)
    inst_vars = vars(obj)
    # Filter out private attributes
    public_vars = {k: v for k, v in inst_vars.items() if not k.startswith('_')}
    return public_vars


def str2bool(v: str) -> bool:
    """Converts truthy and falsey strings to their boolean values."""
    return bool(distutils.util.strtobool(v))


def normalise(v: np.ndarray) -> np.ndarray:
    """Normalise an array along its final dimension."""
    return v / norm(v, axis=-1, keepdims=True)


def orthogonalise(source: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Orthogonalise a vector or set of vectors against another."""
    assert source.ndim == ref.ndim
    if source.ndim == 1:
        return source - np.dot(source, ref) / norm(ref, axis=-1, keepdims=True) * ref
    return source - (np.einsum('bs,bs->b', source, ref) / norm(ref, axis=-1, keepdims=False))[:, None] * ref
