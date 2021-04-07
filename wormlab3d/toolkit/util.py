import argparse
import hashlib
import json

import torch

from wormlab3d import logger


def hash_data(data):
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def print_args(args: argparse.Namespace):
    """Logs all the keys and values present in an argument namespace."""
    log = '--- Arguments ---\n'
    for arg_name, arg_value in vars(args).items():
        log += f'{arg_name}: {arg_value}\n'
    log += '-----------------\n'
    logger.info(log)


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
