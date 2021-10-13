import datetime
from typing import List, Union

import numpy as np
from mongoengine import *

from simple_worm.controls import CONTROL_KEYS
from simple_worm.material_parameters import MaterialParameters
from simple_worm.material_parameters_torch import MaterialParametersTorch
from wormlab3d import logger
from wormlab3d.data.model.frame_sequence import FrameSequence
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.data.triplet_field import TripletField


class SwMaterialParameters(EmbeddedDocument):
    K = FloatField(required=True)
    K_rot = FloatField(required=True)
    A = FloatField(required=True)
    B = FloatField(required=True)
    C = FloatField(required=True)
    D = FloatField(required=True)

    def get_material_parameters(self) -> MaterialParameters:
        return MaterialParametersTorch(
            K=self.K,
            K_rot=self.K_rot,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D
        )


class SwFrameSequence(EmbeddedDocument):
    x = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    psi = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)


class SwControlSequence(EmbeddedDocument):
    alpha = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    beta = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    gamma = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)

    def to_dict(self):
        return {abg: getattr(self, abg) for abg in CONTROL_KEYS}


class SwRun(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    sim_params = ReferenceField('SwSimulationParameters', required=True)
    frame_sequence = ReferenceField('FrameSequence')
    sim_run_target = ReferenceField('SwRun')
    checkpoint = ReferenceField('SwCheckpoint')
    loss = FloatField()
    loss_data = FloatField()
    loss_reg = FloatField()
    reg_losses = DictField()

    # Material parameters
    MP = EmbeddedDocumentField(SwMaterialParameters, required=True)

    # Initial midline position and orientation
    F0 = EmbeddedDocumentField(SwFrameSequence, required=True)

    # Inputs - sequence of controls
    CS = EmbeddedDocumentField(SwControlSequence, required=True)

    # Outputs - midline positions and orientations
    FS = EmbeddedDocumentField(SwFrameSequence, required=True)

    # Projections - 2d camera projections using cameras from the associated input midlines
    X_projections = ListField(TripletField(NumpyField(dtype=np.float32, compression=COMPRESS_BLOSC_PACK)))

    meta = {
        'indexes': ['sim_params', 'frame_sequence', 'created'],
        'ordering': ['-created'],
    }

    def set_target(self, target: Union[FrameSequence, 'SwRun']):
        """
        Target can either be a FrameSequence or a SwRun instance.
        """
        if isinstance(target, SwRun):
            self.sim_run_target = target
        elif isinstance(target, FrameSequence):
            self.frame_sequence = target
        else:
            raise RuntimeError(f'Unrecognised target type: {type(target).__name__}.')

    def clean(self):
        """
        Convert tensors to floats.
        """
        if self.loss is not None:
            self.loss = float(self.loss)
        if self.loss_data is not None:
            self.loss_data = float(self.loss_data)
        if self.loss_reg is not None:
            self.loss_reg = float(self.loss_reg)
        for k1, d in self.reg_losses.items():
            for k2, l in d.items():
                self.reg_losses[k1][k2] = float(l)

    def get_prepared_2d_coordinates(self, regenerate: bool = False) -> List[List[np.ndarray]]:
        """
        Project the 3D midline coordinates down and return relative to the prepared 2D images.
        Caches results into the database on request.
        """
        if self.X_projections is not None and len(self.X_projections) == len(self.FS.x) and not regenerate:
            return self.X_projections

        if self.X_projections is None:
            logger.debug(f'Projected coordinates not available for simulation run={self.id}, generating now.')
        else:
            logger.debug(f'Generating projected coordinates for simulation run={self.id}.')

        prepared_coords = []
        midlines = self.frame_sequence.midlines
        n_midlines = len(midlines)
        n_timesteps = self.FS.x.shape[0]

        # Use the nearest midline to do the projections
        if n_midlines != n_timesteps:
            midline_idxs = np.arange(0, n_midlines, n_midlines / n_timesteps).astype(np.int32)
        else:
            midline_idxs = np.arange(n_midlines)
        for i, midline_idx in enumerate(midline_idxs):
            X = self.FS.x[i].T + self.frame_sequence.centre
            prepared_coords.append(midlines[midline_idx].prepare_2d_coordinates(X=X))

        self.X_projections = prepared_coords
        self.save()

        return self.X_projections
