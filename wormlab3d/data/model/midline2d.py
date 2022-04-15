import numpy as np
from mongoengine import *

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask


class Midline2D(Document):
    frame = ReferenceField('Frame', required=True)
    camera = IntField(choices=CAMERA_IDXS, required=True)

    # Midline coordinates
    X = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)

    # Hand-annotated
    user = StringField()

    # Model generated
    model = ReferenceField('Model')

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline2d',
        'indexes': ['frame', 'user']
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

    def get_prepared_coordinates(self) -> np.ndarray:
        """
        Get the midline coordinates relative to the cropped image.
        """
        centre_2d = self.frame.centre_3d.reprojected_points_2d[self.camera]
        X = self.X.copy()
        X[:, 0] = X[:, 0] - centre_2d[0] + self.frame.trial.crop_size / 2
        X[:, 1] = X[:, 1] - centre_2d[1] + self.frame.trial.crop_size / 2
        X = X[(X[:, 0] >= 0) & (X[:, 1] >= 0)
              & (X[:, 0] < self.frame.trial.crop_size - 0.5)
              & (X[:, 1] < self.frame.trial.crop_size - 0.5)]
        return X

    def get_segmentation_mask(self, blur_sigma: float = None, draw_mode: str = 'line_aa') -> np.ndarray:
        """
        Turn the midline coordinates into a segmentation mask by drawing the coordinates onto the mask
        either using (anti-aliased or not) straight-line interpolations or just the individual pixels.
        Optionally apply a gaussian blur to the mask and then renormalise -- this has the effect of making the midline thicker.
        """
        mask = make_segmentation_mask(
            X=self.get_prepared_coordinates(),
            blur_sigma=blur_sigma,
            draw_mode=draw_mode,
            image_size=(self.frame.trial.crop_size, self.frame.trial.crop_size)
        )
        return mask
