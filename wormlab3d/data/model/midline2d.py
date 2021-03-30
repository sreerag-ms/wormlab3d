import numpy as np
from mongoengine import *

from wormlab3d.data.model.cameras import CAMERA_IDXS
from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK


class Midline2D(Document):
    frame = ReferenceField('Frame', required=True)
    camera = IntField(choices=CAMERA_IDXS, required=True)

    # Midline coordinates
    X = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)

    # Hand-annotated
    user = StringField()

    # Model generated
    model = ReferenceField(Model)

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline2d',
        'indexes': ['frame']
    }

    def get_image(self) -> np.ndarray:
        """
        Get the image associated with this midline annotation from the video.
        midline -> frame -> trial -> video[c] -> frame[f]
        """
        trial = self.frame.trial
        video = trial.get_video_reader(camera_idx=self.camera)
        image = video[self.frame.frame_num]
        return image

    def get_prepared_image(self) -> np.ndarray:
        """
        Fetch the pre-prepared image if available from the frame.
        """
        if len(self.frame.images) == 3:
            return self.frame.images[self.camera]
        return None
