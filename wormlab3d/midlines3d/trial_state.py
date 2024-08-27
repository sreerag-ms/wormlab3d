import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from wormlab3d import MF_DATA_PATH
from wormlab3d import logger
from wormlab3d.data.model import MFParameters, MFCheckpoint, Reconstruction, Trial
from wormlab3d.data.model.mf_parameters import CURVATURE_INTEGRATION_HT, CURVATURE_INTEGRATION_RAND
from wormlab3d.midlines3d.frame_state import FrameState, BUFFER_NAMES, PARAMETER_NAMES, BINARY_DATA_KEYS, \
    CURVATURE_PARAMETER_NAMES, TRANSIENTS_NAMES
from wormlab3d.midlines3d.mf_methods import make_rotation_matrix
from wormlab3d.toolkit.util import to_numpy


class TrialState:
    def __init__(
            self,
            reconstruction: Reconstruction,
            start_frame: int = None,
            end_frame: int = None,
            read_only: bool = True,
            load_only: bool = True,
            partial_load_ok: bool = False,
            copy_state: 'TrialState' = None,
    ):
        self.reconstruction = reconstruction
        self.trial: Trial = reconstruction.trial
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

        # Copy state if required
        if copy_state is not None:
            assert not read_only and not load_only
            self._copy_state(copy_state)

        # Load the state
        loaded = self._load_state(read_only, partial_load_ok)
        if not loaded and (load_only or read_only):
            raise RuntimeError('Could not load trial state.')
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
        return MF_DATA_PATH / f'trial_{self.trial.id}' / str(self.reconstruction.id)

    def _load_state(
            self,
            read_only: bool = True,
            partial_load_ok: bool = False
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
                if k == 'filters':
                    # If filters don't exist already then just create them
                    if k not in meta['shapes'] or not path_state.exists():
                        _, shape = self._init_state_component('filters')
                        meta['shapes'][k] = shape
                        with open(self.path / 'metadata.json', 'w') as f:
                            json.dump(meta, f, indent=2, separators=(',', ': '))

                    # Check if the filter shapes have changed
                    T = meta['shapes'][k][0]
                    ks_old = meta['shapes'][k][2]
                    ks_new = self.parameters.filter_size if self.parameters.filter_size is not None else 1
                    new_filter_shape = (T, 3, ks_new, ks_new)
                    if ks_old != ks_new:
                        if mode == 'r+':
                            # Make a backup of the old filters
                            backup_path = path_state.with_suffix(f'.{meta["checkpoint"]}.npz.bkup')
                            shutil.copy(path_state, backup_path)
                            logger.warning(f'Filter shape has changed! Old filters backed up to: {backup_path}.')

                            # Initialise the new filters
                            self._init_state_component('filters')

                            # Update the meta data on disk
                            meta['shapes'][k] = new_filter_shape
                            with open(self.path / 'metadata.json', 'w') as f:
                                json.dump(meta, f, indent=2, separators=(',', ': '))
                        else:
                            raise RuntimeError('Filter shape invalid for loading!')

                # Head/Tail integration parameters
                if k in ['X0ht', 'T0ht', 'M10ht'] and (k not in meta['shapes'] or not path_state.exists()):
                    _, shape = self._init_state_component(k)
                    meta['shapes'][k] = shape
                    with open(self.path / 'metadata.json', 'w') as f:
                        json.dump(meta, f, indent=2, separators=(',', ': '))

                state = np.memmap(path_state, dtype=np.float32, mode=mode, shape=tuple(meta['shapes'][k]))
                if np.isnan(state).any():
                    if read_only:
                        state = state.copy()
                    state[np.isnan(state)] = 0  # Don't know why, but seeing a lot of nans in the filters...
                states[k] = state
            except Exception as e:
                logger.warning(f'Could not load from {path_state}. {e}')
                if not partial_load_ok:
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

    def _copy_state(self, from_state: 'TrialState' = None):
        """
        Copy state data across from another trial state.
        """
        logger.info(f'Copying state data across from {from_state.path} to {self.path}.')
        os.makedirs(self.path, exist_ok=True)

        # Copy metadata
        assert not (self.path / 'metadata.json').exists(), 'Cannot copy state into an existing state.'
        shutil.copy(from_state.path / 'metadata.json', self.path / 'metadata.json')

        # Copy state files
        for k in BUFFER_NAMES + PARAMETER_NAMES:
            shutil.copy(from_state.path / f'{k}.npz', self.path / f'{k}.npz')

        # Copy statistics
        shutil.copy(from_state.path / 'stats.json', self.path / 'stats.json')

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
        T = self.trial.n_frames_min
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
        dtype = np.float32 if k not in BINARY_DATA_KEYS else np.bool
        state = np.memmap(path_state, dtype=dtype, mode='w+', shape=shape)

        return state, shape

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
            meta['checkpoint'] = str(self.checkpoint.id)
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
            if k in ['points', 'points_2d', 'sigmas', 'exponents', 'intensities', 'scores', ] \
                    + CURVATURE_PARAMETER_NAMES:
                p_ms = frame_state.get_state(k)

                # Stack parameters which do not change across depths
                if k in ['X0', 'T0', 'M10', 'X0ht', 'T0ht', 'M10ht', 'length', 'sigmas', 'exponents', 'intensities']:
                    p = np.array([to_numpy(p) for p in p_ms])

                # Vectorise everything else
                else:
                    p = np.concatenate([to_numpy(p) for p in p_ms], axis=0)

                self.states[k][frame_num] = p
            else:
                self.states[k][frame_num] = to_numpy(frame_state.get_state(k))

        # Add the stats into the json dictionary
        for k, v in frame_state.stats.items():
            if k not in self.stats:
                self.stats[k] = [0. for _ in range(self.trial.n_frames_min)]
            self.stats[k][frame_num] = float(v)

    def init_frame_state(
            self,
            frame_num: int,
            trainable: bool = False,
            load: bool = True,
            prev_frame_state: FrameState = None,
            master_frame_state: FrameState = None,
            use_master_points: bool = True,
            device: torch.device = None
    ) -> FrameState:
        """
        Initialise a frame state
        """
        assert self.start_frame <= frame_num <= self.end_frame, 'Requested frame is not available.'
        frame = self.trial.get_frame(frame_num)
        logger.info(f'Initialising frame state for frame #{frame_num} (id={frame.id}).')

        frame_state = FrameState(
            frame=frame,
            parameters=self.parameters,
            prev_frame_state=prev_frame_state,
            master_frame_state=master_frame_state,
            use_master_points=use_master_points
        )

        if not trainable:
            frame_state.freeze()

        # Restore buffer and parameter values from saved
        if load:
            D = self.parameters.depth
            D_min = self.parameters.depth_min
            idx_offset = 2**D_min - 1
            for k in BUFFER_NAMES + PARAMETER_NAMES:
                v = torch.from_numpy(self.get(k)[frame_num - self.start_frame])

                # Expand collapsed
                if k in ['curvatures', 'points', 'points_2d', 'scores']:
                    v = [v[2**d - idx_offset - 1:2**(d + 1) - idx_offset - 1] for d in range(D_min, D)]
                elif k in ['X0', 'T0', 'M10', 'X0ht', 'T0ht', 'M10ht', 'length', 'sigmas', 'exponents', 'intensities']:
                    v = [v[i] for i in range(D - D_min)]

                # Check filters
                if k == 'filters' and v.sum() == 0:
                    ks = v.shape[1]
                    v[:, int(ks / 2), int(ks / 2)] = 1.
                    v += torch.randn_like(v) * 1e-4
                    v /= v.norm(dim=(1, 2), keepdim=True)

                frame_state.set_state(k, v)

            # Check HT data is valid if required
            update_from_mp = False
            if self.parameters.curvature_integration == CURVATURE_INTEGRATION_HT:
                for k in ['X0ht', 'T0ht', 'M10ht']:
                    if self.get(k)[frame_num].sum() == 0:
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

    def get(self, k: str, start_frame: int = None, end_frame: int = None) -> np.ndarray:
        """
        Return a slice of data for a given buffer/parameter key.
        """
        assert k in BUFFER_NAMES + PARAMETER_NAMES + TRANSIENTS_NAMES, f'Unrecognised key: {k}.'

        if start_frame is None:
            start_frame = self.start_frame
        else:
            assert 0 <= start_frame <= self.trial.n_frames_min
        if end_frame is None:
            end_frame = self.end_frame + 1
        else:
            assert start_frame <= end_frame <= self.trial.n_frames_min + 1
        to_end = end_frame == self.end_frame + 1

        if k in TRANSIENTS_NAMES:
            if k == 'points_3d_base':
                centres_3d, _ = self.trial.get_tracking_data(
                    fixed=True,
                    start_frame=start_frame,
                    end_frame=None if to_end else end_frame
                )
                return centres_3d

            elif k == 'points_2d_base':
                centres_2d, _ = self.trial.get_tracking_data(
                    fixed=True,
                    start_frame=start_frame,
                    end_frame=None if to_end else end_frame,
                    return_2d_points=True
                )
                return centres_2d

            else:
                raise RuntimeError(f'Transient key = {k} not yet supported!')

        if k == 'cam_rotations':
            # Build camera rotation matrics
            Rs = []
            rotation_preangles = self.get('cam_rotation_preangles', start_frame, None if to_end else end_frame)
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

        return state[start_frame:end_frame]

    def __len__(self):
        return self.n_frames
