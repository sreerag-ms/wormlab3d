import numpy as np
from mongoengine import *

from wormlab3d import CAMERA_IDXS, PREPARED_IMAGE_SIZE
from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask

M3D_SOURCE_WT3D = 'WT3D'
M3D_SOURCE_RECONST = 'reconst'
M3D_SOURCE_MODEL = 'model'

M3D_SOURCES = [
    M3D_SOURCE_WT3D,
    M3D_SOURCE_RECONST,
    M3D_SOURCE_MODEL
]


class Midline3D(Document):
    frame = ReferenceField('Frame', required=True)

    # Midline coordinates
    X = NumpyField(required=True, dtype=np.float32, compression=COMPRESS_BLOSC_PACK)
    base_3d = NumpyField(shape=(3,), dtype=np.float32)
    error = FloatField()

    # Model/source used to generate this midline
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    model = ReferenceField(Model)

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline3d',
        'indexes': ['frame', 'source']
    }

    def get_segmentation_masks(self, blur_sigma: float = None, draw_mode: str = 'line_aa') -> np.ndarray:
        """
        Turn the midline coordinates into a set of segmentation masks by projecting the 3D object points
        down to the 3*2D camera views using the camera model and then drawing the coordinates onto the mask
        either using (anti-aliased or not) straight-line interpolations or just the individual pixels.
        Optionally apply a gaussian blur to the mask and then renormalise -- this has the effect of making the midline thicker.
        """
        from wormlab3d.data.model import Frame
        self.frame: Frame
        cams = self.frame.get_cameras()
        cameras = cams.get_camera_model_triplet()

        image_points = cameras.project_to_2d(object_points=self.X)
        image_points = np.array(image_points).transpose(1, 0, 2)

        masks = []
        for c in CAMERA_IDXS:
            mask = make_segmentation_mask(
                X=self._get_prepared_2d_coordinates(camera_idx=c, image_points=image_points[c]),
                blur_sigma=blur_sigma,
                draw_mode=draw_mode,
                image_size=PREPARED_IMAGE_SIZE
            )
            masks.append(mask)
        return masks

    def _get_prepared_2d_coordinates(self, camera_idx: int, image_points: np.ndarray) -> np.ndarray:
        """
        Get the midline coordinates relative to the cropped image.
        """
        centre_2d = self.frame.centre_3d.reprojected_points_2d[camera_idx]
        X = image_points.copy()
        X[:, 0] = X[:, 0] - centre_2d[0] + PREPARED_IMAGE_SIZE[0] / 2
        X[:, 1] = X[:, 1] - centre_2d[1] + PREPARED_IMAGE_SIZE[1] / 2
        X = X[(X[:, 0] >= 0) & (X[:, 1] >= 0)
              & (X[:, 0] < PREPARED_IMAGE_SIZE[0] - 0.5)
              & (X[:, 1] < PREPARED_IMAGE_SIZE[1] - 0.5)]
        return X
