from typing import List

import numpy as np
from mongoengine import *

from wormlab3d import CAMERA_IDXS, PREPARED_IMAGE_SIZE, logger
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.cameras import CAM_SOURCE_WT3D, CAM_SOURCE_ANNEX
from wormlab3d.data.model.model import Model
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.midlines2d.masks_from_coordinates import make_segmentation_mask
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.toolkit.camera_model_triplet import CameraModelTriplet

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

    # Reprojected 2d coordinates
    X_projections = TripletField(NumpyField(dtype=np.float32, compression=COMPRESS_BLOSC_PACK))

    # Model/source used to generate this midline
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    model = ReferenceField(Model)

    # Natural frame representation
    natural_frame = NumpyField(dtype=np.complex64, compression=COMPRESS_BLOSC_PACK)

    # Specify collection name otherwise it puts an underscore in it
    meta = {
        'collection': 'midline3d',
        'indexes': ['frame', 'source', 'source_file']
    }

    def get_segmentation_masks(self, blur_sigma: float = None, draw_mode: str = 'line_aa') -> np.ndarray:
        """
        Turn the midline coordinates into a set of segmentation masks by projecting the 3D object points
        down to the 3*2D camera views using the camera model and then drawing the coordinates onto the mask
        either using (anti-aliased or not) straight-line interpolations or just the individual pixels.
        Optionally apply a gaussian blur to the mask and then renormalise -- this has the effect of making the midline thicker.
        """
        prepared_coords = self.get_prepared_2d_coordinates()
        masks = []
        for c in CAMERA_IDXS:
            mask = make_segmentation_mask(
                X=prepared_coords[c],
                blur_sigma=blur_sigma,
                draw_mode=draw_mode,
                image_size=PREPARED_IMAGE_SIZE
            )
            masks.append(mask)
        return masks

    def get_prepared_2d_coordinates(self, regenerate: bool = False, cameras: CameraModelTriplet = None) -> np.ndarray:
        """
        Project the 3D midline coordinates down and return relative to the prepared 2D images.
        Caches results into the database on request.
        """
        if self.X_projections is not None and len(self.X_projections) == 3 and not regenerate:
            return self.X_projections

        if self.X_projections is None:
            logger.debug(f'Projected coordinates not available for midline={self.id}, generating now.')
        else:
            logger.debug(f'Generating projected coordinates for midline={self.id}.')

        self.X_projections = self.prepare_2d_coordinates(cameras=cameras)
        self.save()

        return self.X_projections

    def prepare_2d_coordinates(self, X: np.ndarray = None, cameras: CameraModelTriplet = None) -> np.ndarray:
        """
        Project the 3D midline coordinates down and return relative to the prepared 2D images.
        Allow different 3D coordinates to be passed in, otherwise use the object's property.
        """
        if cameras is None:
            cams = self.get_cameras()
            cameras = cams.get_camera_model_triplet()

        # Use the camera models to project the object points to the image points.
        X = self.X if X is None else X
        image_points = cameras.project_to_2d(object_points=X)
        image_points = np.array(image_points).transpose(1, 0, 2)

        prepared_coords = []
        for c in CAMERA_IDXS:
            prepared_coords.append(
                self._project_coordinates(camera_idx=c, image_points=image_points[c])
            )

        return prepared_coords

    def get_cameras(self) -> Cameras:
        """
        Fetch the correct camera model for this midline.
        """
        cams: List[Cameras] = Cameras.objects(
            source=CAM_SOURCE_WT3D if self.source == M3D_SOURCE_WT3D else CAM_SOURCE_ANNEX,
            trial=self.frame.trial
        )

        # If no cams found for the same midline source and trial, fetch the best for the frame.
        if cams.count() == 0:
            return self.frame.get_cameras()

        # If just one model was found return this.
        if cams.count() == 1:
            return cams[0]

        # Otherwise we have multiple cameras for this trial and source, pick the one closest to the frame
        frame_num = self.frame.frame_num
        offsets = np.zeros(cams.count())
        for i, cam in enumerate(cams):
            if cam.frame is not None and cam.frame.frame_num is not None:
                offsets[i] = abs(cam.frame.frame_num - frame_num)
            else:
                offsets[i] = np.inf
        best_cams = cams[int(np.argmin(offsets))]

        return best_cams

    def _project_coordinates(self, camera_idx: int, image_points: np.ndarray) -> np.ndarray:
        """
        Get the midline coordinates relative to the cropped image for the given camera.
        """
        centre_2d = self.frame.centre_3d.reprojected_points_2d[camera_idx]
        X = image_points.copy()
        X[:, 0] = X[:, 0] - centre_2d[0] + PREPARED_IMAGE_SIZE[0] / 2
        X[:, 1] = X[:, 1] - centre_2d[1] + PREPARED_IMAGE_SIZE[1] / 2
        X = X[(X[:, 0] >= 0) & (X[:, 1] >= 0)
              & (X[:, 0] < PREPARED_IMAGE_SIZE[0] - 0.5)
              & (X[:, 1] < PREPARED_IMAGE_SIZE[1] - 0.5)]
        return X

    def get_natural_frame(self, regenerate: bool=False) -> np.ndarray:
        """
        Get the natural frame (m1+i.m2) representation.
        """
        if self.natural_frame is not None and not regenerate:
            return self.natural_frame

        if self.natural_frame is None:
            logger.debug(f'Natural frame not available for midline={self.id}, generating now.')
        else:
            logger.debug(f'Generating natural frame representation for midline={self.id}.')

        nf = NaturalFrame(self.X)
        self.natural_frame = nf.mc
        self.save()

        return self.natural_frame
