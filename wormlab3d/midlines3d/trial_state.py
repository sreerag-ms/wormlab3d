import json
import os
from typing import Dict, Any

import numpy as np

from wormlab3d import PREPARED_IMAGE_SIZE, DATA_PATH
from wormlab3d import logger
from wormlab3d.data.model import Trial, MFModelParameters, MFCheckpoint
from wormlab3d.midlines3d.args.network_args import ENCODING_MODE_DELTA_VECTORS, ENCODING_MODE_DELTA_ANGLES, \
    ENCODING_MODE_DELTA_ANGLES_BASIS, ENCODING_MODE_POINTS, ENCODING_MODE_MSC
from wormlab3d.midlines3d.args_finder import OptimiserArgs
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES
from wormlab3d.toolkit.util import hash_data, to_dict, to_numpy

TRIAL_STATES_PATH = DATA_PATH + '/MF_outputs'


class TrialState:
    def __init__(
            self,
            trial: Trial,
            start_frame: int,
            end_frame: int,
            model_params: MFModelParameters,
            optimiser_args: OptimiserArgs
    ):
        self.trial = trial
        self.start_frame = start_frame
        self.end_frame = trial.n_frames_min if end_frame == -1 else end_frame
        self.model_params = model_params
        self.optimiser_args = optimiser_args
        self.states = {}
        self.stats = {}
        self.checkpoint: MFCheckpoint = None

        # Load frame numbers
        self.n_frames = self.end_frame - self.start_frame + 1
        self.frame_nums = []
        for i in range(self.start_frame, self.end_frame + 1):
            self.frame_nums.append(i)
        assert len(self.frame_nums) == self.n_frames

        loaded = self._load_state()
        if not loaded:
            self._init_state()
            self.save()

    @property
    def meta(self) -> Dict[str, Any]:
        if self.model_params.curve_mode == ENCODING_MODE_MSC:
            return {
                'trial': self.trial.id,
                'model_params': str(self.model_params.id),
            }

        return {
            'trial': self.trial.id,
            # 'start_frame': self.start_frame,
            # 'end_frame': self.end_frame,
            'model_params': str(self.model_params.id),
            **to_dict(self.optimiser_args)
        }

    @property
    def path(self):
        meta_hash = hash_data(self.meta)
        return f'{TRIAL_STATES_PATH}/trial_{self.trial.id}/{meta_hash}'

    def _load_state(self) -> bool:
        """
        Try to load the state.
        """

        # Check for metadata first
        path_meta = self.path + '/metadata.json'
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
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            path_state = self.path + f'/{k}.npz'
            try:
                state = np.memmap(path_state, dtype=np.float32, mode='r+', shape=tuple(meta['shapes'][k]))
                states[k] = state
                logger.info(f'Loaded data from {path_state}.')
            except Exception as e:
                logger.warning(f'Could not load from {path_state}. {e}')
                return False

        # Load statistics
        path_stats = self.path + '/stats.json'
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
        mp = self.model_params
        N = self.trial.n_frames_min
        states = {}
        shapes = {}

        for k in BUFFER_NAMES + PARAMETER_NAMES:
            path_state = self.path + f'/{k}.npz'

            if k in ['images', 'masks_target', 'masks_cloud', 'masks_curve']:
                shape = (N, 3, *PREPARED_IMAGE_SIZE)
            elif k in ['cam_coeffs_db', 'cam_coeffs']:
                shape = (N, 3, 22)
            elif k == 'cam_intrinsics':
                shape = (N, 3, 4)
            elif k == 'cam_rotations':
                shape = (N, 3, 9)
            elif k == 'cam_rotation_preangles':
                shape = (N, 3, 3, 2)
            elif k == 'cam_translations':
                shape = (N, 3, 3)
            elif k == 'cam_distortions':
                shape = (N, 3, 5)
            elif k == 'cam_shifts':
                shape = (N, 3, 1)
            elif k == 'points_3d_base':
                shape = (N, 3)
            elif k == 'points_2d_base':
                shape = (N, 3, 2)
            elif k == 'points_2d_base':
                shape = (N, 3, 2)
            elif k == 'cloud_points':
                shape = (N, mp.n_cloud_points, 3)
            elif k == 'curve_parameters':
                if mp.curve_mode == ENCODING_MODE_POINTS:
                    shape = (N, mp.n_curve_points, 3)
                elif mp.curve_mode == ENCODING_MODE_DELTA_VECTORS:
                    shape = (N, 1 + mp.n_curve_points, 3)
                elif mp.curve_mode == ENCODING_MODE_DELTA_ANGLES:
                    shape = (N, 3 + 4 + 2 * (mp.n_curve_points - 1))
                elif mp.curve_mode == ENCODING_MODE_DELTA_ANGLES_BASIS:
                    shape = (N, 3 + 4 + 4 * mp.n_curve_basis_fns)
                elif mp.curve_mode == ENCODING_MODE_MSC:
                    shape = (N, mp.n_curve_points, 3)
            elif k in ['worm_length_db', 'curve_length']:
                shape = (N,)
            elif k in ['blur_sigmas_cloud', 'cloud_points_scores']:
                shape = (N, mp.n_cloud_points)
            elif k in ['blur_sigmas_curve', 'blur_intensities_curve', 'curve_points_scores']:
                shape = (N, mp.n_curve_points)
            elif k in ['blur_sigmas_cameras_sfs', 'blur_intensities_cameras_sfs']:
                shape = (N, 3)
            elif k == 'curve_points':
                shape = (N, mp.n_curve_points, 3)
            else:
                raise RuntimeError(f'Unknown shape for buffer/parameter key: {k}')

            shape = np.array(shape)
            shape = shape.clip(min=1)
            shape = tuple(int(s) for s in shape)
            if any(s == 0 for s in shape):
                logger.debug(f'Empty shape for {k}, skipping.')
                continue
            shapes[k] = shape
            states[k] = np.memmap(path_state, dtype=np.float32, mode='w+', shape=shape)

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
        with open(self.path + '/metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, separators=(',', ': '))

        # Save the stats
        with open(self.path + '/stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2, separators=(',', ': '))

    def update_frame_state(self, frame_num: int, frame_state: FrameState):
        """
        Update the state of a single frame.
        """
        assert self.start_frame <= frame_num <= self.end_frame, 'Requested frame is not available.'
        i = frame_num - self.start_frame
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            # print(k)
            # v = to_numpy(frame_state.get_state(k))
            # print(v.shape)
            # print(self.states[k][i].shape)
            #'blur_sigmas_cameras_sfs', 'blur_intensities_cameras_sfs',

            if self.model_params.curve_mode == ENCODING_MODE_MSC \
                    and k in ['curve_parameters', 'blur_sigmas_curve', 'blur_intensities_curve','curve_points_scores', 'masks_target', 'masks_curve']:
                p_ms = frame_state.get_state(k)
                if k in ['masks_target', 'masks_curve']:
                    p = to_numpy(p_ms[-1])
                else:
                    p = np.concatenate([to_numpy(p) for p in p_ms], axis=0)
                # print(k)
                self.states[k][i] = p
            else:
                # print(k)
                # print(frame_state.get_state(k).shape)
                self.states[k][i] = to_numpy(frame_state.get_state(k))

        for k, v in frame_state.stats.items():
            if k not in self.stats:
                self.stats[k] = [0. for _ in range(self.n_frames)]
            self.stats[k][i] = float(v)

    def __len__(self):
        return self.n_frames
