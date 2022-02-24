import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

from wormlab3d import PREPARED_IMAGE_SIZE, DATA_PATH
from wormlab3d import logger
from wormlab3d.data.model import MFParameters, MFCheckpoint, Reconstruction
from wormlab3d.midlines3d.dynamic_cameras import N_CAM_COEFFICIENTS
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, BINARY_DATA_KEYS
from wormlab3d.toolkit.util import to_numpy

TRIAL_STATES_PATH = DATA_PATH / 'MF_outputs'


class TrialState:
    def __init__(
            self,
            reconstruction: Reconstruction,
            start_frame: int = None,
            end_frame: int = None,
            read_only: bool = True
    ):
        self.reconstruction = reconstruction
        self.trial = reconstruction.trial
        if start_frame is None:
            start_frame = reconstruction.start_frame
        if end_frame is None:
            end_frame = reconstruction.end_frame
        self.start_frame = start_frame
        self.end_frame = self.trial.n_frames_min if end_frame == -1 else end_frame
        self.parameters: MFParameters = reconstruction.mf_parameters
        self.states = {}
        self.stats = {}
        self.checkpoint: MFCheckpoint = None

        # Load frame numbers
        self.n_frames = self.end_frame - self.start_frame + 1
        self.frame_nums = []
        for i in range(self.start_frame, self.end_frame + 1):
            self.frame_nums.append(i)
        assert len(self.frame_nums) == self.n_frames

        loaded = self._load_state(read_only)
        if not loaded and not read_only:
            self._init_state()
            self.save()

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            'reconstruction': str(self.reconstruction.id),
            'trial': int(self.trial.id),
            'parameters': str(self.parameters.id),
        }

    @property
    def path(self) -> Path:
        return TRIAL_STATES_PATH / f'trial_{self.trial.id}' / str(self.reconstruction.id)

    def _load_state(self, read_only: bool = True) -> bool:
        """
        Try to load the state.
        """

        # Check for metadata first
        path_meta = self.path / 'metadata.json'
        if not os.path.exists(path_meta):
            return False
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f'Could not load from {path_meta}. {e}')
            return False

        # If metadata exists, use the shapes to load the other state files.
        states = {}
        mode = 'r' if read_only else 'r+'
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            path_state = self.path / f'{k}.npz'
            try:
                state = np.memmap(path_state, dtype=np.float32, mode=mode, shape=tuple(meta['shapes'][k]))
                states[k] = state
                logger.info(f'Loaded data from {path_state}.')
            except Exception as e:
                logger.warning(f'Could not load from {path_state}. {e}')
                return False

        # Load statistics
        path_stats = self.path / 'stats.json'
        try:
            with open(path_stats, 'r') as f:
                stats = json.load(f)
        except Exception as e:
            logger.warning(f'Could not load from {path_stats}. {e}')
            return False

        self.states = states
        self.shapes = meta['shapes']
        self.stats = stats

        return True

    def _init_state(self):
        """
        Initialise empty state.
        """
        logger.info(f'Initialising state in {self.path}.')
        os.makedirs(self.path, exist_ok=True)
        mp = self.parameters
        T = self.trial.n_frames_min
        N = mp.n_points_total
        D = mp.depth
        states = {}
        shapes = {}

        for k in BUFFER_NAMES + PARAMETER_NAMES:
            path_state = self.path / f'{k}.npz'

            if k in ['images', 'masks_target', 'masks_target_residuals', 'masks_curve']:
                shape = (T, 3, *PREPARED_IMAGE_SIZE)
            elif k in ['cam_coeffs_db', 'cam_coeffs']:
                shape = (T, 3, N_CAM_COEFFICIENTS)
            elif k == 'cam_intrinsics':
                shape = (T, 3, 4)
            elif k == 'cam_rotations':
                shape = (T, 3, 9)
            elif k == 'cam_rotation_preangles':
                shape = (T, 3, 3, 2)
            elif k == 'cam_translations':
                shape = (T, 3, 3)
            elif k == 'cam_distortions':
                shape = (T, 3, 5)
            elif k == 'cam_shifts':
                shape = (T, 3, 1)
            elif k == 'points_3d_base':
                shape = (T, 3)
            elif k == 'points_2d_base':
                shape = (T, 3, 2)
            elif k == 'points':
                shape = (T, N, 3)
            elif k == 'points_2d':
                shape = (T, N, 3, 2)
            elif k == 'curve_lengths':
                shape = (T, D)
            elif k in ['sigmas', 'intensities', 'scores']:
                shape = (T, N)
            elif k in ['camera_sigmas', 'camera_intensities']:
                shape = (T, 3)
            else:
                raise RuntimeError(f'Unknown shape for buffer/parameter key: {k}')

            shape = np.array(shape)
            shape = shape.clip(min=1)
            shape = tuple(int(s) for s in shape)
            if any(s == 0 for s in shape):
                logger.debug(f'Empty shape for {k}, skipping.')
                continue
            shapes[k] = shape
            dtype = np.float32 if k not in BINARY_DATA_KEYS else np.bool
            states[k] = np.memmap(path_state, dtype=dtype, mode='w+', shape=shape)

        self.states = states
        self.shapes = shapes

    def save(self):
        """
        Save the states to the hard drive
        """
        logger.debug(f'Saving trial state to {self.path}.')
        for n in BUFFER_NAMES + PARAMETER_NAMES:
            self.states[n].flush()

        # Save the meta data
        meta = {**self.meta, 'shapes': self.shapes}
        if self.checkpoint is not None and self.checkpoint.id is not None:
            meta['checkpoint'] = self.checkpoint.id
        with open(self.path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, separators=(',', ': '))

        # Save the stats
        with open(self.path / 'stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2, separators=(',', ': '))

    def update_frame_state(self, frame_num: int, frame_state: FrameState):
        """
        Update the state of a single frame.
        """
        assert self.start_frame <= frame_num <= self.end_frame, 'Requested frame is not available.'
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            # Collate outputs generated at each depth
            if k in ['points', 'points_2d', 'sigmas', 'intensities', 'scores', 'masks_target', 'masks_target_residuals',
                     'masks_curve']:
                p_ms = frame_state.get_state(k)

                # Only store the deepest target and output masks
                if k in ['masks_target', 'masks_target_residuals', 'masks_curve']:
                    p = to_numpy(p_ms[-1])

                # Vectorise everything else
                else:
                    p = np.concatenate([to_numpy(p) for p in p_ms], axis=0)

                self.states[k][frame_num] = p
            else:
                self.states[k][frame_num] = to_numpy(frame_state.get_state(k))

        # Add the stats into the json dictionary
        for k, v in frame_state.stats.items():
            if k not in self.stats:
                self.stats[k] = [0. for _ in range(self.n_frames)]
            self.stats[k][frame_num] = float(v)

    def get(self, k: str, start_frame: int = None, end_frame: int = None) -> np.ndarray:
        """
        Return a slice of data for a given buffer/parameter key.
        """
        assert k in BUFFER_NAMES + PARAMETER_NAMES
        state = self.states[k]

        if start_frame is None and end_frame is None:
            return state

        if start_frame is not None:
            assert start_frame >= self.frame_nums[0]
        else:
            start_frame = self.start_frame
        if end_frame is not None:
            assert end_frame >= self.frame_nums[-1]
        else:
            end_frame = self.end_frame + 1

        return state[start_frame:end_frame]

    def __len__(self):
        return self.n_frames
