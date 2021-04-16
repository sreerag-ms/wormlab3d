import numpy as np
from mongoengine import *
from skimage.draw import line, line_aa
from skimage.filters import gaussian

from wormlab3d import PREPARED_IMAGE_SIZE, CAMERA_IDXS, logger
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK


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
        X[:, 0] = X[:, 0] - centre_2d[0] + PREPARED_IMAGE_SIZE[0] / 2
        X[:, 1] = X[:, 1] - centre_2d[1] + PREPARED_IMAGE_SIZE[1] / 2
        X = X[(X[:, 0] >= 0) & (X[:, 1] >= 0)
              & (X[:, 0] < PREPARED_IMAGE_SIZE[0] - 0.5)
              & (X[:, 1] < PREPARED_IMAGE_SIZE[1] - 0.5)]
        return X

    def get_segmentation_mask(self, blur_sigma: float = None, draw_mode: str = 'line_aa') -> np.ndarray:
        """
        Turn the midline coordinates into a segmentation mask by drawing the coordinates onto the mask
        either using (anti-aliased or not) straight-line interpolations or just the individual pixels.
        Optionally apply a gaussian blur to the mask and then renormalise -- this has the effect of making the midline thicker.
        """
        X = self.get_prepared_coordinates()
        X = X.round().astype(np.uint8)
        mask = np.zeros(PREPARED_IMAGE_SIZE, dtype=np.float32)

        # Anti-aliased lines between coordinates
        if draw_mode == 'line_aa':
            for i in range(len(X) - 1):
                rr, cc, val = line_aa(X[i, 1], X[i, 0], X[i + 1, 1], X[i + 1, 0])
                mask[rr, cc] = val

        # Simpler single-pixel lines between coordinates
        elif draw_mode == 'line':
            for i in range(len(X) - 1):
                rr, cc = line(X[i, 1], X[i, 0], X[i + 1, 1], X[i + 1, 0])
                mask[rr, cc] = 1

        # Draw the coordinate pixels only
        elif draw_mode == 'pixels':
            mask[X[:, 1], X[:, 0]] = 1

        else:
            raise RuntimeError(f'Unrecognised draw_mode: {draw_mode}')

        # Apply a gaussian blur and then re-normalise to "fatten" the midline
        if blur_sigma is not None:
            mask = gaussian(mask, sigma=blur_sigma)

            # Normalise to [0-1] with float32 dtype
            mask_range = mask.max() - mask.min()
            if mask_range > 0:
                mask = (mask - mask.min()) / mask_range
            else:
                logger.warning(f'Mask range zero! (midline id={self.id})')

        return mask
