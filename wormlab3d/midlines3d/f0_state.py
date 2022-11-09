import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from wormlab3d import MF_DATA_PATH
from wormlab3d import logger
from wormlab3d.data.model import MFParameters, Reconstruction, Trial, Frame
from wormlab3d.data.model.mf_parameters import CURVATURE_INTEGRATION_HT, CURVATURE_INTEGRATION_RAND
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, BINARY_DATA_KEYS, \
    CURVATURE_PARAMETER_NAMES, TRANSIENTS_NAMES
from wormlab3d.midlines3d.mf_methods import make_rotation_matrix
from wormlab3d.toolkit.util import to_numpy


class F0State:
    def __init__(
            self,
            reconstruction: Reconstruction,
            frame: Frame,
            read_only: bool = True,
            load_only: bool = True,
    ):
        self.reconstruction = reconstruction
        self.frame = frame
        self.trial: Trial = reconstruction.trial
        self.parameters: MFParameters = reconstruction.mf_parameters
        self.n_steps = self.parameters.n_steps_init + 1
        self.states = {}
        self.stats = {}

        # Load the state
        loaded = self._load_state(read_only)
        if not loaded and (load_only or read_only):
            raise RuntimeError('Could not load f0 state.')
        if not loaded and not read_only:
            self._init_state()
            self.save()

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            'trial': int(self.trial.id),
            'reconstruction': str(self.reconstruction.id),
            'frame': str(self.frame.id),
            'parameters': str(self.parameters.id),
        }

    @property
    def path(self) -> Path:
        return MF_DATA_PATH / f'trial_{self.trial.id}' / str(self.reconstruction.id) / f'f0_{self.frame.frame_num:05d}'

    def _load_state(
            self,
            read_only: bool = True,
    ) -> bool:
        """
        Try to load the state.
        """

        # Check for metadata first
        path_meta = self.path / 'metadata.json'
        if not path_meta.exists():
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
                if np.isnan(state).any():
                    if read_only:
                        state = state.copy()
                    state[np.isnan(state)] = 0  # Don't know why, but seeing a lot of nans in the filters...
                states[k] = state
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

        logger.info(f'Loaded data from {self.path}.')

        return True

    def _init_state(self):
        """
        Initialise empty state.
        """
        logger.info(f'Initialising state in {self.path}.')
        os.makedirs(self.path, exist_ok=True)
        states = {}
        shapes = {}
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            states[k], shapes[k] = self._init_state_component(k)
        self.states = states
        self.shapes = shapes

    def _init_state_component(self, k: str):
        """
        Initialise an empty state component.
        """
        mp = self.parameters
        T = self.n_steps
        N = mp.n_points_total
        D = mp.depth - mp.depth_min
        path_state = self.path / f'{k}.npz'

        if k == 'cam_intrinsics':
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
        elif k in ['X0', 'T0', 'M10']:
            shape = (T, D, 3)
        elif k in ['X0ht', 'T0ht', 'M10ht']:
            shape = (T, D, 2, 3)
        elif k == 'curvatures':
            shape = (T, N, 2)
        elif k == 'points':
            shape = (T, N, 3)
        elif k == 'points_2d':
            shape = (T, N, 3, 2)
        elif k in ['length', 'sigmas', 'exponents', 'intensities']:
            shape = (T, D)
        elif k == 'scores':
            shape = (T, N)
        elif k in ['camera_sigmas', 'camera_exponents', 'camera_intensities']:
            shape = (T, 3)
        elif k == 'filters':
            ks = mp.filter_size if mp.filter_size is not None else 1
            shape = (T, 3, ks, ks)
        else:
            raise RuntimeError(f'Unknown shape for buffer/parameter key: {k}')

        shape = np.array(shape)
        shape = shape.clip(min=1)
        shape = tuple(int(s) for s in shape)
        if any(s == 0 for s in shape):
            raise RuntimeError(f'Empty shape for {k}.')
        dtype = np.float32 if k not in BINARY_DATA_KEYS else bool
        state = np.memmap(path_state, dtype=dtype, mode='w+', shape=shape)

        return state, shape

    def save(self):
        """
        Save the states to the hard drive
        """
        logger.debug(f'Saving f0 state to {self.path}.')
        for n in BUFFER_NAMES + PARAMETER_NAMES:
            self.states[n].flush()

        # Save the meta data
        meta = {**self.meta, 'shapes': self.shapes}
        with open(self.path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, separators=(',', ': '))

        # Save the stats
        with open(self.path / 'stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2, separators=(',', ': '))

    def update_step_state(self, step: int, frame_state: FrameState):
        """
        Update the state of a single step.
        """
        assert 0 <= step < self.n_steps, 'Requested step is not available.'
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            # Collate outputs generated at each depth
            if k in ['points', 'points_2d', 'sigmas', 'exponents', 'intensities', 'scores', ] \
                    + CURVATURE_PARAMETER_NAMES:
                p_ms = frame_state.get_state(k)

                # Stack parameters which do not change across depths
                if k in ['X0', 'T0', 'M10', 'X0ht', 'T0ht', 'M10ht', 'length', 'sigmas', 'exponents', 'intensities']:
                    p = np.array([to_numpy(p) for p in p_ms])

                # Vectorise everything else
                else:
                    p = np.concatenate([to_numpy(p) for p in p_ms], axis=0)

                self.states[k][step] = p
            else:
                self.states[k][step] = to_numpy(frame_state.get_state(k))

        # Add the stats into the json dictionary
        for k, v in frame_state.stats.items():
            if k not in self.stats:
                self.stats[k] = [0. for _ in range(self.trial.n_frames_min)]
            self.stats[k][step] = float(v)

    def init_frame_state(
            self,
            step: int,
            device: torch.device = None
    ) -> FrameState:
        """
        Initialise a frame state at the given step.
        """
        assert 0 <= step < self.n_steps, 'Requested step is not available.'
        frame = self.trial.get_frame(step)
        logger.info(f'Initialising frame state for frame #{frame.frame_num} (id={frame.id}) at step {step}.')

        # Initialise an empty FrameState for this frame
        frame_state = FrameState(
            frame=frame,
            parameters=self.parameters,
        )
        frame_state.freeze()

        # Load buffer and parameter values from saved step
        D = self.parameters.depth
        D_min = self.parameters.depth_min
        idx_offset = 2**D_min - 1
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            v = torch.from_numpy(self.get(k)[step])

            # Expand collapsed
            if k in ['curvatures', 'points', 'points_2d', 'scores']:
                v = [v[2**d - idx_offset - 1:2**(d + 1) - idx_offset - 1] for d in range(D_min, D)]
            elif k in ['X0', 'T0', 'M10', 'X0ht', 'T0ht', 'M10ht', 'length', 'sigmas', 'exponents', 'intensities']:
                v = [v[i] for i in range(D - D_min)]

            frame_state.set_state(k, v)

        # Check HT data is valid if required
        update_from_mp = False
        if self.parameters.curvature_integration == CURVATURE_INTEGRATION_HT:
            for k in ['X0ht', 'T0ht', 'M10ht']:
                if self.get(k)[step].sum() == 0:
                    update_from_mp = True
                    break
        elif self.parameters.curvature_integration == CURVATURE_INTEGRATION_RAND:
            update_from_mp = True
        if update_from_mp:
            frame_state.update_data_from_mp()

        # Move onto target device
        if device is not None:
            frame_state.to(device)

        return frame_state

    def get(self, k: str, start_step: int = None, end_step: int = None) -> np.ndarray:
        """
        Return a slice of data for a given buffer/parameter key.
        """
        assert k in BUFFER_NAMES + PARAMETER_NAMES + TRANSIENTS_NAMES, f'Unrecognised key: {k}.'

        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = self.n_steps
        assert 0 <= start_step < end_step <= self.n_steps

        if k in TRANSIENTS_NAMES:
            raise RuntimeError(f'Transient key = {k} not yet supported!')

        if k == 'cam_rotations':
            # Build camera rotation matrics
            Rs = []
            rotation_preangles = self.get('cam_rotation_preangles', start_step, end_step)
            for i in range(3):
                pre = rotation_preangles[:, i]
                cos_phi, sin_phi = pre[:, 0, 0], pre[:, 0, 1]
                cos_theta, sin_theta = pre[:, 1, 0], pre[:, 1, 1]
                cos_psi, sin_psi = pre[:, 2, 0], pre[:, 2, 1]
                Ri = make_rotation_matrix(
                    torch.from_numpy(cos_phi),
                    torch.from_numpy(sin_phi),
                    torch.from_numpy(cos_theta),
                    torch.from_numpy(sin_theta),
                    torch.from_numpy(cos_psi),
                    torch.from_numpy(sin_psi),
                )
                Rs.append(Ri.numpy().transpose(2, 0, 1).reshape(len(rotation_preangles), 9))
            Rs = np.stack(Rs, axis=1)
            return Rs

        state = self.states[k]

        return state[start_step:end_step]

    def __len__(self):
        return self.n_steps
