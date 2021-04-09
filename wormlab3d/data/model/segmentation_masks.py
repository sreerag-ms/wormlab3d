import datetime

import numpy as np
from mongoengine import *

from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_POINTER


class SegmentationMasks(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    trial = ReferenceField('Trial', required=True)
    frame = ReferenceField('Frame', required=True)
    checkpoint = ReferenceField('Checkpoint', required=True)

    # Segmentation masks - one for each view, stored as a single array of shape (3, 200, 200)
    X = NumpyField(
        required=True,
        dtype=np.float32,
        shape=(3,) + PREPARED_IMAGE_SIZE,
        compression=COMPRESS_BLOSC_POINTER
    )

    meta = {
        'indexes': [
            {
                'fields': ['trial', 'frame', 'checkpoint'],
                'unique': True
            }
        ],
    }
