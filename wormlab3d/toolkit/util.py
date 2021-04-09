import argparse
import hashlib
import json
from typing import Tuple, List

import torch

from wormlab3d import CAMERA_IDXS
from wormlab3d import logger
from wormlab3d.data.model.trial import Trial


def resolve_targets(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None,
) -> Tuple[List[Trial], List[int]]:
    """
    Resolves any combination of passed experiments, trials and cameras.
    """

    # Get trials
    if trial_id is not None:
        trial = Trial.objects.get(id=trial_id)
        if experiment_id is not None:
            assert trial.experiment.id == experiment_id, 'Trial is not part of target experiment!'
        trials = [trial]
    else:
        if experiment_id is not None:
            trials = Trial.objects(experiment_id=experiment_id)
        else:
            trials = Trial.objects

    # Camera indices
    cam_idxs = CAMERA_IDXS
    if camera_idx is not None:
        assert trial_id is not None
        assert camera_idx in cam_idxs
        cam_idxs = [camera_idx]

    # Frame number - only makes sense in combination with a trial_id
    if frame_num is not None:
        assert trial_id is not None
        assert 0 <= frame_num < trial.num_frames

    return trials, cam_idxs


def print_args(args: argparse.Namespace):
    """Logs all the keys and values present in an argument namespace."""
    log = '--- Arguments ---\n'
    for arg_name, arg_value in vars(args).items():
        log += f'{arg_name}: {arg_value}\n'
    log += '-----------------\n'
    logger.info(log)


def hash_data(data):
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


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
