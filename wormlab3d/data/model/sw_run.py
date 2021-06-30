import datetime
from typing import List

import numpy as np
from mongoengine import *

from simple_worm.controls import CONTROL_KEYS
from wormlab3d import logger
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.data.triplet_field import TripletField


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
    frame_sequence = ReferenceField('FrameSequence', required=True)
    checkpoint = ReferenceField('SwCheckpoint')
    loss_data = FloatField()

    # Initial midline position and orientation
    F0 = EmbeddedDocumentField(SwFrameSequence, required=True)

    # Inputs - sequence of controls
    CS = EmbeddedDocumentField(SwControlSequence, required=True)

    # Outputs - midline positions and orientations
    FS = EmbeddedDocumentField(SwFrameSequence, required=True)

    # Projections - 2d camera projections using cameras from the associated input midlines
    X_projections = ListField(TripletField(NumpyField(dtype=np.float32, compression=COMPRESS_BLOSC_PACK)))

    meta = {
        'indexes': ['sim_params', 'frame_sequence'],
        'ordering': ['-created'],
    }

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
        for i, midline in enumerate(self.frame_sequence.midlines):
            X = self.FS.x[i].T + self.frame_sequence.centre
            prepared_coords.append(midline.prepare_2d_coordinates(X=X))

        self.X_projections = prepared_coords
        self.save()

        return self.X_projections
