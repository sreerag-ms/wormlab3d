import datetime
from typing import List

import numpy as np
from mongoengine import *

from wormlab3d import PREPARED_IMAGE_SIZE_DEFAULT
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
        shape=(3, PREPARED_IMAGE_SIZE_DEFAULT, PREPARED_IMAGE_SIZE_DEFAULT),
        compression=COMPRESS_BLOSC_POINTER
    )

    # Indexes
    meta = {
        'indexes': [
            'trial',
            'frame',
            'checkpoint',
            {
                'fields': ['trial', 'frame', 'checkpoint'],
                'unique': True
            }
        ],
        'ordering': ['-created']
    }

    def get_images(self) -> List[np.ndarray]:
        return self.frame.images  # todo: resize to fit
