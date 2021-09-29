import numpy as np

from wormlab3d import logger
from wormlab3d.data.model import Trial
from wormlab3d.data.model.frame import PREPARED_IMAGE_SIZE, Frame
from wormlab3d.preprocessing.cropper import crop_image
from wormlab3d.toolkit.util import resolve_targets


def generate_prepared_images(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None,
        missing_only: bool = True,
        fix_missing_centres: bool = False,
        max_3d_error: float = 0,
):
    """
    Using the centre_3d point for a frame and its corresponding 2d reprojection points,
    generate a prepared image by cropping around this point, inverting, normalising and saving to database.
    """
    if max_3d_error > 0:
        assert fix_missing_centres is False, 'Can\'t fix missing centres and filter by max 3d error!'

    trials, _ = resolve_targets(experiment_id, trial_id, frame_num=frame_num)
    trial_ids = [t.id for t in trials]
    if missing_only:
        logger.info(f'Generating any missing prepared images for {len(trial_ids)} trials.')
    else:
        logger.info(f'(Re)generating ALL prepared images for {len(trial_ids)} trials.')

    # Iterate over matching trials
    for trial_id in trial_ids:
        logger.info(f'Processing trial id={trial_id}')
        trial = Trial.objects.get(id=trial_id)

        # Filter frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            filters = {}
            if missing_only:
                filters['__raw__'] = {
                    '$or': [
                        {'images': None},
                        {'images': {'$size': 0}},
                    ]
                }
            if max_3d_error > 0:
                filters['centre_3d__error__lte'] = max_3d_error
            elif not fix_missing_centres:
                filters['centre_3d__error__exists'] = True
            frames = trial.get_frames(filters).only('id').no_dereference()
        frame_ids = [f.id for f in frames]

        if len(frame_ids) == 0:
            logger.debug('No frames missing images!')
            continue

        reader = trial.get_video_triplet_reader()

        # Iterate over the frames
        for frame_id in frame_ids:
            frame = Frame.objects.get(id=frame_id)
            log_prefix = f'Frame #{frame.frame_num}/{trial.n_frames_min} (id={frame.id}). '

            # Due to the large number of frames in each trial, we need to reload from the
            # database to catch changes since we fetched the results.
            if missing_only and len(frame.images) == 3:
                logger.info(log_prefix + 'Has images, skipping.')
                continue

            # Assign the readers from the trial as reloading from the database will break this reference
            frame.trial = trial

            # Check the centre point exists and if not, create it
            if frame.centre_3d is None:
                if fix_missing_centres:
                    logger.warning(log_prefix + '3D centre point unavailable, generating now.')
                    res = frame.generate_centre_3d(
                        error_threshold=50,
                        try_experiment_cams=True,
                        try_all_cams=False,
                        only_replace_if_better=True,
                        store_bad_result=True,
                        ratio_adj_orig=0.1,
                        ratio_adj_exp=0.1,
                    )
                    if not res:
                        logger.warning(log_prefix + '3D centre point could not be generated, skipping.')
                        continue
                    frame.save()
                else:
                    logger.warning(log_prefix + '3D centre point unavailable, skipping.')
                    continue
            assert frame.centre_3d is not None

            if 0 < max_3d_error < frame.centre_3d.error:
                logger.warning(log_prefix + f'3D centre point error ({frame.centre_3d.error:.2f}) '
                                            f'> max_3d_error ({max_3d_error:.2f}), skipping.')
                continue

            # Set the frame number, fetch the images from each video and generate the crops
            logger.info(log_prefix + 'Generating crops.')
            reader.set_frame_num(frame.frame_num)
            images = reader.get_images(invert=True, subtract_background=True)
            crops = []
            for c, image in images.items():
                crop = crop_image(
                    image=image,
                    centre_2d=frame.centre_3d.reprojected_points_2d[c],
                    size=PREPARED_IMAGE_SIZE,
                    fix_overlaps=True
                )

                # Normalise to [0-1] with float32 dtype
                crop = crop.astype(np.float32) / 255.
                crop = (crop - crop.min()) / (crop.max() - crop.min())
                crops.append(crop)
            frame.images = crops

            # Unlock the frame when finished
            frame.release_lock_and_save()


if __name__ == '__main__':
    generate_prepared_images(
        # # Poor error, spot obscured in one view
        # trial_id=186,
        # frame_num=823

        # Lots of 2d points
        # trial_id=301,
        # frame_num=79

        missing_only=True,
        fix_missing_centres=False,
        max_3d_error=100,
    )
