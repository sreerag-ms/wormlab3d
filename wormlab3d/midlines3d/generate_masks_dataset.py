from bson import ObjectId

from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import DatasetMidline2D, Dataset
from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.nn.generate_dataset import build_pipeline, build_dataset

from wormlab3d import logger
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.data.model.dataset import DatasetSegmentationMasks
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.nn.data_loader import DatasetLoader, make_data_loader

def generate_masks_dataset(args: DatasetSegmentationMasksArgs, fix_frames: bool = False) -> DatasetSegmentationMasks:
    """
    Generate a segmentation masks dataset.
    """
    logger.info(
        f'Generating dataset: ---- \n' +
        '\n'.join(args.get_info()) +
        '\n-----------------------------------\n'
    )
    failed_midline_ids = []

    def validate_midline(midline_id: ObjectId) -> bool:
        midline = Midline2D.objects.no_cache().get(id=midline_id)
        frame = midline.frame
        logger.debug(f'Checking frame id={frame.id}')
        try:
            if fix_frames:
                if not frame.centres_2d_available():
                    logger.warning('Frame does not have 2d centre points available for all views, generating now.')
                    frame.generate_centres_2d()
                    frame.save()
                if frame.centre_3d is None:
                    logger.warning('Frame does not have 3d centre point available, generating now.')
                    frame.generate_centre_3d()
                    frame.save()
                if len(frame.images) != 3:
                    logger.warning(f'Frame does not have prepared images, generating now.')
                    frame.generate_prepared_images()
                    frame.save()

            assert len(frame.images) == 3, 'Frame does not have prepared images.'
            assert midline.frame.centre_3d is not None, 'Frame does not have 3d centre point available.'
            assert len(midline.get_prepared_coordinates()) > 1, 'Midline coordinates empty after crop.'
            return True
        except AssertionError as e:
            failed_midline_ids.append(midline_id)
            if fix_frames:
                logger.error(f'Failed to prepare frame: {e}')
            else:
                logger.error(f'Frame is not ready: {e}')
        return False

    # Fetch manually-annotated midlines, grouped by tags (each might appear for multiple tags)
    # The query starts matching on the midline2d collection.
    logger.info('Querying database.')
    pipeline = [
        {'$match': {'user': {'$exists': True}}},
        *build_pipeline(args)
    ]
    cursor = Midline2D.objects().aggregate(pipeline)

    # Build the dataset
    DS = build_dataset(cursor, args, validate_midline)

    n_failed = len(failed_midline_ids)
    if n_failed > 0:
        logger.error(f'Failed to include {n_failed}/{DS.size_all + n_failed} matching midlines: {failed_midline_ids}.')

    # Save dataset
    logger.debug('Saving dataset.')
    DS.save()

    return DS
