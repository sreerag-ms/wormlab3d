import datetime

import numpy as np
from mongoengine import *

from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK


class SwFrameSequence(EmbeddedDocument):
    x = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    psi = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)


class SwControlSequence(EmbeddedDocument):
    alpha = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    beta = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    gamma = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)


class SwRun(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    sim_params = ReferenceField('SwSimulationParameters', required=True)
    frame_sequence = ReferenceField('FrameSequence', required=True)
    checkpoint = ReferenceField('SwCheckpoint')

    # Initial midline position and orientation
    F0 = EmbeddedDocumentField(SwFrameSequence, required=True)

    # Inputs - sequence of controls
    CS = EmbeddedDocumentField(SwControlSequence, required=True)

    # Outputs - midline positions and orientations
    FS = EmbeddedDocumentField(SwFrameSequence, required=True)

    meta = {
        'indexes': ['sim_params', 'frame_sequence'],
        'ordering': ['-created'],
    }
