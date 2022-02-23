from typing import List, Union

import numpy as np
from mongoengine import *

from wormlab3d import logger, PREPARED_IMAGE_SIZE, CAMERA_IDXS, PREPARED_IMAGES_PATH
from wormlab3d.data.model import Cameras, CameraShifts
from wormlab3d.data.model.cameras import CAM_SOURCE_ANNEX
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.midline3d import Midline3D
from wormlab3d.data.model.object_point import ObjectPoint
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.preprocessing.contour import CONT_THRESH_RATIO_DEFAULT, MIN_REQ_THRESHOLD
from wormlab3d.preprocessing.cropper import crop_image
from wormlab3d.toolkit.triangulate import triangulate


class Frame(Document):
    experiment = ReferenceField(Experiment, required=True)
    trial = ReferenceField('Trial', required=True)
    frame_num = IntField(required=True)
    max_brightnesses = TripletField(IntField())
    locked = BooleanField(default=False)

    # Triangulations
    centres_2d = TripletField(ListField(ListField()))
    centres_2d_thresholds = TripletField(IntField())
    centre_3d = EmbeddedDocumentField(ObjectPoint)
    centre_3d_fixed = EmbeddedDocumentField(ObjectPoint)

    # Tags
    tags = ListField(ReferenceField(Tag))

    # Indexes
    meta = {
        'indexes': [
            'trial',
            'frame_num',
            {
                'fields': ['trial', 'frame_num'],
                'unique': True
            },
            'centre_3d.error'
        ],
        'ordering': ['+trial', '+frame_num']
    }

    def __getattribute__(self, k):
        # Load images from the file system
        if k != 'images':
            return super().__getattribute__(k)
        path = PREPARED_IMAGES_PATH / f'{self.trial.id:03d}' / f'{self.frame_num:06d}.npz'
        try:
            return np.load(path)['images']
        except Exception:
            return None

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

    def get_midlines3d(
            self,
            filters: dict = None
    ) -> List['Midline3D']:
        """
        Fetch all the 3D midlines associated with this frame.
        """
        if filters is None:
            filters = {}
        filters = {'frame': self, **filters}
        return Midline3D.objects(**filters).order_by('+error')

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

    def generate_centres_2d(self, cont_threshold_ratios: List[float] = None):
        """
        Find centre points of any objects in each of the views.
        """
        if cont_threshold_ratios is None:
            cont_threshold_ratios = [CONT_THRESH_RATIO_DEFAULT] * 3
        assert len(
            cont_threshold_ratios) == 3, f'Received {len(cont_threshold_ratios)} contour threshold ratios. Needed 3.'
        assert len(
            self.max_brightnesses) == 3, f'Max brightnesses not set for all views in frame: {self.max_brightnesses}.'

        # Check if we've already got results with this threshold for any of the cameras
        if self.centres_2d_available() and len(self.centres_2d_thresholds) == 3 and len(self.max_brightnesses) == 3:
            cam_idxs = []
            for c in CAMERA_IDXS:
                threshold = max(MIN_REQ_THRESHOLD, int(self.max_brightnesses[c] * cont_threshold_ratios[c]))
                if self.centres_2d_thresholds[c] != threshold:
                    cam_idxs.append(c)
        else:
            cam_idxs = CAMERA_IDXS

        if len(cam_idxs) == 0:
            logger.debug('2D centres already recorded at these thresholds for all cameras.')
            return

        if len(cam_idxs) != 3:
            logger.debug(f'2D centres not already recorded at this threshold for cameras: {cam_idxs}.')

        reader = self.trial.get_video_triplet_reader()
        reader.set_frame_num(self.frame_num)
        centres_2d, centres_2d_thresholds = reader.find_objects(
            cont_threshold_ratios=cont_threshold_ratios,
            cam_idxs=cam_idxs
        )
        assert len(centres_2d) == len(centres_2d_thresholds) == len(cam_idxs)

        return centres_2d, centres_2d_thresholds

    def update_centres_2d(self, centres_2d: Union[list, dict], centres_2d_thresholds):
        """
        Update the 2D centre points and the thresholds they were found at.
        """
        if type(centres_2d) == list:
            assert len(centres_2d) == 3
            cam_idxs = CAMERA_IDXS
        else:
            cam_idxs = centres_2d.keys()
        prev_centres_2d = self.centres_2d.copy()

        if len(self.centres_2d) == 3:
            for c in cam_idxs:
                self.centres_2d[c] = centres_2d[c]
        else:
            self.centres_2d = list(centres_2d.values())

        if len(self.centres_2d_thresholds) == 3:
            for c in cam_idxs:
                self.centres_2d_thresholds[c] = centres_2d_thresholds[c]
        else:
            self.centres_2d_thresholds = list(centres_2d_thresholds.values())

        # If the centre points have changed then this invalidated the 3D centre point and any prepared images also
        if prev_centres_2d != self.centres_2d:
            self.centre_3d = None
            self.images = None

    def generate_centre_3d(
            self,
            x0=None,
            error_threshold: float = 50,
            try_experiment_cams: bool = True,
            try_all_cams: bool = False,
            only_replace_if_better: bool = True,
            store_bad_result: bool = True,
            ratio_adj_orig: float = 0,
            ratio_adj_exp: float = 0,
            ratio_adj_all: float = 0,
    ) -> bool:
        """
        Find the triangulated 3d object centre point.
        """
        if self.centres_2d_available() and len(self.centres_2d_thresholds) == 3:
            centres_2d = self.centres_2d.copy()
            centres_2d_thresholds = self.centres_2d_thresholds.copy()
        else:
            logger.warning('Frame does not have 2d centre points available for all views, generating now.')
            centres_2d, centres_2d_thresholds = self.generate_centres_2d()

        # Try own camera model for the benchmark
        trial_cameras = self.trial.get_cameras(best=False, fallback_to_experiment=False)
        best_err = 1000
        best = None
        best_centres_2d = None
        best_centres_2d_thresholds = None
        all_results = {}

        def _update_centre_3d(ignore_threshold: bool = False) -> bool:
            # Error exceeds threshold
            if not ignore_threshold and best_err > error_threshold:
                return False

            # Error exceeds previous result
            if only_replace_if_better and self.centre_3d is not None and best_err > self.centre_3d.error:
                # logger.debug(f'Error ({best_err:.2f}) > previous best ({self.centre_3d.error:.1f}).')
                return False

            # Error less than threshold and probably previous error also
            if ignore_threshold and best_err > error_threshold:
                logger.debug(f'Error ({best_err:.2f}) > Threshold ({error_threshold:.1f}), storing anyway.')
            else:
                logger.debug(f'Error ({best_err:.2f}) < Threshold ({error_threshold:.1f}), happy days.')

            # # If the 3d point has changed then we need to discard any associated images as these will need recreating
            # prev_3d = (0, 0, 0) if self.centre_3d is None else list(self.centre_3d.point_3d)
            # self.update_centres_2d(best_centres_2d, best_centres_2d_thresholds)
            # self.centre_3d = best
            # if prev_3d != list(self.centre_3d.point_3d) and self.images:
            #     self.images = None
            return True

        def _triangulate(cams: Cameras) -> bool:
            nonlocal best, best_err, x0, all_results, best_centres_2d, best_centres_2d_thresholds
            if cams.id in all_results:
                return False
            try:
                # Apply shifts
                if cams.source == CAM_SOURCE_ANNEX:
                    shifts = self.get_shifts()
                    if shifts is not None:
                        cams.set_shifts(shifts)

                res_3d = triangulate(
                    image_points=centres_2d,
                    cameras=cams,
                    x0=x0,
                    matching_threshold=best_err
                )

                # Only consider the first result with the lowest error
                tmp_best = res_3d[0]
                if best is None or tmp_best.error < best_err:
                    best = tmp_best
                    best_err = tmp_best.error
                    best_centres_2d = centres_2d.copy()
                    best_centres_2d_thresholds = centres_2d_thresholds.copy()
                    logger.debug(f'New best error: {best_err:.2f}')

                # Result counts as a success if the error is below threshold and better than existing
                if best_err < error_threshold and (self.centre_3d is None or best_err < self.centre_3d.error):
                    all_results[cams.id] = True
                    return True

            except ValueError:
                pass

            all_results[cams.id] = False
            return False

        def _triangulate_batch(cameras_list: List[Cameras]) -> bool:
            nonlocal best, best_err, x0
            if best is not None:
                x0 = best.point_3d
                logger.debug(f'Using x0={x0}.')
            results = []
            for cams in cameras_list:
                # If the camera reprojection error is greater than the best error we currently have, skip it
                if cams.reprojection_error > best_err:
                    continue
                results.append(_triangulate(cams))
            return any(results)

        def _find_more_centres(ratio_adj: float = 0) -> bool:
            nonlocal all_results, centres_2d, centres_2d_thresholds

            # Try lowering the contouring threshold to find more 2d points.
            if ratio_adj > 0:
                cont_ratios = [
                    centres_2d_thresholds[c] / self.max_brightnesses[c] - ratio_adj
                    for c in CAMERA_IDXS
                ]
                logger.debug(f'Regenerating 2d centres with contour thresholds = {[f"{c:.2f}" for c in cont_ratios]}. '
                             f'ratio_adj = {ratio_adj}')
                centres_2d, centres_2d_thresholds = self.generate_centres_2d(cont_threshold_ratios=cont_ratios)
                all_results = {}
                return True
            return False

        def _check_trial_and_frame_cams() -> bool:
            # Trial cameras
            if trial_cameras is not None:
                logger.debug('Trying trial cameras.')
                if _triangulate_batch(trial_cameras):
                    return True

            # # Cameras from nearby frames
            # btfs = Frame.objects(  # "best triangulated frames"
            #     trial=self.trial,
            #     frame_num__gte=self.frame_num - 15,
            #     frame_num__lte=self.frame_num + 15,
            #     centre_3d__error__exists=True,
            #     centre_3d__error__lte=min(best_err, error_threshold * 3),
            # ).order_by('+centre_3d__error')[:10]
            # btfs_cameras = [f.centre_3d.cameras for f in btfs]
            # if len(btfs_cameras) > 0:
            #     logger.debug('Trying cameras used in nearby frames.')
            #     if _triangulate_batch(btfs_cameras):
            #         return True

            return False

        ratio_adj_steps = {
            1: ratio_adj_orig,
            2: 0 if not try_experiment_cams else ratio_adj_exp,
            3: 0 if not try_all_cams else ratio_adj_all,
        }
        for step in range(4):
            logger.debug(f'--------------- Step = {step}')

            # Find more centres
            if step > 0:
                changed = _find_more_centres(ratio_adj_steps[step])
                if not changed:  # If the centres haven't changed then skip to the next step
                    continue

            # Check trial and frame cameras
            if step >= 0:
                res = _check_trial_and_frame_cams()

            # Check to see if another camera model from the same experiment can do a better job
            if not res and step >= 2 and try_experiment_cams:
                logger.debug(f'Trying other camera models from same experiment.')
                exp_cameras = self.experiment.get_cameras(best=False)
                res = _triangulate_batch(exp_cameras)

            # Check to see if another camera model from any experiment can do a better job
            if not res and step >= 3 and try_all_cams:
                logger.debug(f'Trying other camera models from any experiment.')
                all_cameras = list(Cameras.objects(reprojection_error__lte=min(best_err, 10)))
                np.random.shuffle(all_cameras)
                all_cameras = all_cameras[:10]  # just take 10 at random
                res = _triangulate_batch(all_cameras)

            # Try to save result
            if res and _update_centre_3d():
                return True

        # If we've got here then we've exhausted our options without success.
        if best is None:
            logger.error('Triangulation failed, completely.')
            return False
        logger.debug(f'Best error with any cameras = {best_err:.2f}.')

        # But there may be a bad result to save
        res = False
        if store_bad_result:
            res = _update_centre_3d(ignore_threshold=True)

        return res

    def generate_prepared_images(self):
        """
        Generate prepared image crops.
        """

        # Check the centre point exists and if not, create it
        if self.centre_3d is None and self.centre_3d_fixed is None:
            logger.warning('Frame does not have a 3d centre point available, generating now.')
            res = self.generate_centre_3d()
            if not res:
                return False
            assert self.centre_3d is not None

        # Use the fixed (interpolated and smoothed) centre point if available
        if self.centre_3d_fixed is not None:
            p3d = self.centre_3d_fixed
        else:
            p3d = self.centre_3d

        # Set the frame number, fetch the images from each video and generate the crops
        reader = self.trial.get_video_triplet_reader()
        reader.set_frame_num(self.frame_num)
        images = reader.get_images(invert=True, subtract_background=True)
        crops = []
        for c, image in images.items():
            crop = crop_image(
                image=image,
                centre_2d=p3d.reprojected_points_2d[c],
                size=PREPARED_IMAGE_SIZE,
                fix_overlaps=True
            )

            # Normalise to [0-1] with float32 dtype
            crop = crop.astype(np.float32) / 255.
            crop = (crop - crop.min()) / (crop.max() - crop.min())
            crops.append(crop)

        self.images = crops

    def get_cameras(self, use_shifts: bool = True) -> Cameras:
        if self.centre_3d is not None:
            cams = self.centre_3d.cameras
        else:
            cams = self.trial.get_cameras()
        if use_shifts and cams.source == CAM_SOURCE_ANNEX:
            shifts = self.get_shifts()
            if shifts is not None:
                cams.set_shifts(shifts)
        return cams

    def get_shifts(self) -> CameraShifts:
        try:
            shifts = CameraShifts.objects.get(frame=self)
        except DoesNotExist:
            shifts = None
        return shifts

    def is_ready(self) -> bool:
        """
        Checks to see if the frame has 2D centres for all views, a 3D centre point and 3 prepared images.
        """
        return self.centres_2d_available() and self.centre_3d is not None and len(self.images) == 3

    def get_lock(self) -> bool:
        """
        Marks the frame as locked in the database.
        Not atomic, so might hit race conditions!
        """
        self.reload('locked')
        if self.locked:
            return False
        self.locked = True
        self.save()
        return True

    def release_lock_and_save(self) -> bool:
        """
        Marks the frame as not-locked in the database.
        Not atomic, and since the get_lock also isn't, might cause problems!
        """
        self.locked = False
        self.save()

    @staticmethod
    def unlock_all():
        """Helper method to clear the locked-state for all frames."""
        Frame.objects.update(locked=False)

    @staticmethod
    def reset_centres():
        """Helper method to remove ALL centres_2d and centre_3d points from all frames."""
        Frame.objects.update(
            locked=False,
            centres_2d=[[], [], []],
            centre_3d=None,
        )
