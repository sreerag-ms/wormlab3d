import numpy as np

from scripts.preprocessing.generate_centres import generate_centres_3d
from wormlab3d import logger
from wormlab3d.data.model.frame import PREPARED_IMAGE_SIZE
from wormlab3d.preprocessing.cropper import crop_image
from wormlab3d.toolkit.util import resolve_targets


def generate_prepared_images(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None
):
    """
    Using the centre_3d point for a frame and its corresponding 2d reprojection points,
    generate a prepared image by cropping around this point, inverting, normalising and saving to database.
    """
    trials, _ = resolve_targets(experiment_id, trial_id, frame_num=frame_num)
    logger.info(f'Generating cropped frames for {len(trials)} trials.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        reader = trial.get_video_triplet_reader()

        # Iterate over the frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            frames = trial.get_frames()
        for frame in frames:
            logger.info(f'Frame #{frame.frame_num}/{trial.num_frames} (id={frame.id}).')

            # Check the centre point exists and if not, create it
            if frame.centre_3d is None:
                logger.warning('Frame does not have a 3d centre point available, generating now.')
                generate_centres_3d(
                    trial_id=trial.id,
                    frame_num=frame.frame_num
                )
                frame = frame.reload()
                assert frame.centre_3d is not None

            # Set the frame number, fetch the images from each video and generate the crops
            reader.set_frame_num(frame.frame_num)
            images = reader.get_images(invert=True, subtract_background=True)
            crops = []
            for c, image in enumerate(images):
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
            frame.save()


if __name__ == '__main__':
    generate_prepared_images(
        # # Poor error, spot obscured in one view
        # trial_id=186,
        # frame_num=823

        # Lots of 2d points
        trial_id=301,
        frame_num=79
    )
