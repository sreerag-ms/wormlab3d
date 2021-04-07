from typing import List

import numpy as np
from mongoengine import *

from wormlab3d import logger, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.object_point import ObjectPoint
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_POINTER
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.preprocessing.contour import CONT_THRESH_DEFAULT
from wormlab3d.preprocessing.cropper import crop_image
from wormlab3d.toolkit.triangulate import triangulate


class Frame(Document):
    experiment = ReferenceField(Experiment, required=True)
    trial = ReferenceField('Trial', required=True)
    frame_num = IntField(required=True)

    # Triangulations
    centres_2d = TripletField(ListField(ListField()))
    centre_3d = EmbeddedDocumentField(ObjectPoint)

    # Prepared images (we don't store high-resolution images)
    images = TripletField(
        NumpyField(
            shape=PREPARED_IMAGE_SIZE,
            dtype=np.float32,
            compression=COMPRESS_BLOSC_POINTER
        )
    )

    # Tags
    tags = ListField(ReferenceField(Tag))

    # Indexes
    meta = {
        'indexes': [
            'trial',
            {
                'fields': ['trial', 'frame_num'],
                'unique': True
            }
        ],
        'ordering': ['+trial', '+frame_num']
    }

    def get_midlines2d(
            self,
            manual_only: bool = False,
            generated_only: bool = False,
            filters: dict = None
    ) -> List['Midline2D']:
        """
        Fetch all the 2D midlines associated with this frame.
        """
        assert not (manual_only and generated_only)
        if filters is None:
            filters = {}
        filters = {'frame': self, **filters}
        if manual_only:
            filters['user__exists'] = True
            filters['model__exists'] = False
        if generated_only:
            filters['user__exists'] = False
            filters['model__exists'] = True

        return Midline2D.objects(**filters)

    def centres_2d_available(self) -> bool:
        """
        Check that we have 2d centre points available in each camera view
        """
        image_points_valid = True
        if len(self.centres_2d) != 3:
            image_points_valid = False
        else:
            for centres_2d_cam in self.centres_2d:
                if len(centres_2d_cam) == 0:
                    image_points_valid = False
                    break
        return image_points_valid

    def generate_centres_2d(self, cont_threshold: float = CONT_THRESH_DEFAULT):
        """
        Find centre points of any objects in each of the views.
        """
        reader = self.trial.get_video_triplet_reader()
        reader.set_frame_num(self.frame_num)
        self.centres_2d = reader.find_objects(cont_threshold=cont_threshold)
        reader.close()

    def generate_centre_3d(self, x0=None, error_threshold: float = 50):
        """
        Find the triangulated 3d object centre point.
        """
        if not self.centres_2d_available():
            logger.warning('Frame does not have 2d centre points available for all views, generating now.')
            self.generate_centres_2d()

        best_err = 1000

        # Try own camera model for the benchmark
        trial_cameras = self.trial.get_cameras(best=True, fallback_to_experiment=False)
        trial_best = None
        if trial_cameras is not None:
            try:
                logger.debug(f'Trying trial cameras, id={trial_cameras.id}. error={trial_cameras.reprojection_error}.')
                res_3d = triangulate(
                    image_points=self.centres_2d,
                    cameras=trial_cameras,
                    x0=x0,
                    matching_threshold=best_err
                )
                trial_best = res_3d[0]
                best_err = trial_best.error
                if best_err < error_threshold:
                    logger.debug(f'Error ({best_err:.2f}) < {error_threshold:.1f}, happy days.')
                    self.centre_3d = trial_best
                    return
            except ValueError:
                logger.warning('Triangulation failed using trial cameras.')
        else:
            logger.warning('Trial cameras not found in database.')

        # Check to see if another camera model from the same experiment can do a better job
        logger.debug(f'Error ({best_err:.1f}) > Threshold ({error_threshold:.1f}). '
                     f'Trying other camera models from same experiment.')
        if trial_best is not None:
            x0 = trial_best.point_3d
            logger.debug(f'Using x0={x0}.')

        # Try also lowering the contouring threshold to find more 2d points
        # if we didn't find anything better than twice the threshold
        if best_err > error_threshold * 2:
            cont_threshold = CONT_THRESH_DEFAULT - 0.1
            logger.debug(f'Regenerating 2d centres with contour threshold = {cont_threshold:.2f}')
            self.generate_centres_2d(cont_threshold=cont_threshold)

        # This will also include the trial cameras, but from a better starting point
        exp_cameras = self.experiment.get_cameras(best=False)
        exps_best = None if trial_best is None else trial_best
        for cameras in exp_cameras:
            # If the camera reprojection error is greater than the best error we currently have, skip it
            if cameras.reprojection_error > best_err:
                continue
            try:
                res_3d = triangulate(
                    image_points=self.centres_2d,
                    cameras=cameras,
                    x0=x0,
                    matching_threshold=best_err
                )
                exp_best = res_3d[0]
                if exps_best is None or exp_best.error < best_err:
                    exps_best = exp_best
                    best_err = exp_best.error
                    logger.debug(f'New best error: {best_err:.2f}')
            except ValueError:
                pass

        # Check if one of the experiment cameras gave a good enough result
        if exps_best is not None:
            logger.debug(f'Best error with any experiment cameras = {best_err:.2f}.')
            if best_err < error_threshold * 2:
                if best_err < error_threshold:
                    logger.debug(f'Error ({best_err:.1f}) < Threshold ({error_threshold:.1f}), happy days.')
                else:
                    logger.debug(f'Error ({best_err:.1f}) < 2*Threshold ({2 * error_threshold:.1f}), stopping here.')
                self.centre_3d = exps_best
                return

        # Check to see if another camera model from any experiment can do a better job
        logger.debug(f'Error ({best_err:.1f}) > 2*Threshold ({2 * error_threshold:.1f}), '
                     f'Trying other camera models present in any experiments.')

        # Lower the contouring threshold again to find even more 2d points
        # if we still didn't find anything better than twice the threshold
        if best_err > error_threshold * 2:
            cont_threshold = CONT_THRESH_DEFAULT - 0.15
            logger.debug(f'Regenerating 2d centres with contour threshold = {cont_threshold:.2f}')
            self.generate_centres_2d(cont_threshold=cont_threshold)

        # Try across all camera models
        all_cameras = Cameras.objects
        all_best = None
        for cameras in all_cameras:
            # If the camera reprojection error is greater than the best error we currently have, skip it
            if cameras.reprojection_error > best_err:
                continue
            try:
                res_3d = triangulate(
                    image_points=self.centres_2d,
                    cameras=cameras,
                    x0=x0,
                    matching_threshold=best_err
                )
                cam_best = res_3d[0]
                if all_best is None or cam_best.error < all_best.error:
                    all_best = cam_best
                    best_err = all_best.error
                    logger.debug(f'New best error: {best_err:.2f}')
            except ValueError:
                pass

        if all_best is None:
            raise RuntimeError('Triangulation failed, completely.')

        logger.debug(f'Best error with any cameras = {best_err:.2f}.')
        if all_best.error > error_threshold:
            logger.warning(f'Error > {error_threshold:.1f}, storing anyway.')
        self.centre_3d = all_best

    def generate_prepared_images(self):
        """
        Generate prepared image crops.
        """

        # Check the centre point exists and if not, create it
        if self.centre_3d is None:
            logger.warning('Frame does not have a 3d centre point available, generating now.')
            self.generate_centre_3d()
            assert self.centre_3d is not None

        # Set the frame number, fetch the images from each video and generate the crops
        reader = self.trial.get_video_triplet_reader()
        reader.set_frame_num(self.frame_num)
        images = reader.get_images(invert=True, subtract_background=True)
        reader.close()
        crops = []
        for c, image in enumerate(images):
            crop = crop_image(
                image=image,
                centre_2d=self.centre_3d.reprojected_points_2d[c],
                size=PREPARED_IMAGE_SIZE,
                fix_overlaps=True
            )

            # Normalise to [0-1] with float32 dtype
            crop = crop.astype(np.float32) / 255.
            crop = (crop - crop.min()) / (crop.max() - crop.min())
            crops.append(crop)

        self.images = crops
