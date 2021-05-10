from wormlab3d import logger
from wormlab3d.data.model import Checkpoint
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.data.model.dataset import DatasetMidline2D
from wormlab3d.data.model.dataset import DatasetSegmentationMasks
from wormlab3d.midlines3d.args import DatasetSegmentationMasksArgs
from wormlab3d.nn.generate_dataset import build_pipeline, build_dataset


def generate_masks_dataset(args: DatasetSegmentationMasksArgs) -> DatasetSegmentationMasks:
    """
    Generate a segmentation masks dataset.
    """
    logger.info(
        f'Generating dataset: ---- \n' +
        '\n'.join(args.get_info()) +
        '\n-----------------------------------\n'
    )
    failed_masks_ids = []

    # Fetch the checkpoint to use, if one wasn't specified
    if args.masks_model_checkpoint_id is None:
        dataset_ids = DatasetMidline2D.objects.scalar('id')
        checkpoint = Checkpoint.objects(dataset__in=dataset_ids).order_by('+loss_test').first()
        if checkpoint is None:
            raise RuntimeError('No checkpoints found for models trained on Midline2D annotations.')
    else:
        checkpoint = Checkpoint.objects.get(id=args.masks_model_checkpoint_id)
    logger.info(f'Using checkpoint id="{checkpoint.id}". '
                f'Test loss = {checkpoint.loss_test:.5f}. '
                f'Created = {checkpoint.created:%Y-%m-%d %H:%M}.')

    # Fetch masks, grouped by tags (each might appear for multiple tags)
    # The query starts matching on the segmentation_masks collection.
    logger.info('Querying database.')
    pipeline = [
        {'$match': {'checkpoint': checkpoint.id}},
        *build_pipeline(args)
    ]
    cursor = SegmentationMasks.objects().timeout(False).aggregate(pipeline)

    # Build the dataset
    DS = build_dataset(cursor, args)
    if DS.size_train == 0 and args.train_test_split > 0 or DS.size_test == 0 and args.train_test_split < 1:
        raise RuntimeError('No results returned from database!')

    n_failed = len(failed_masks_ids)
    if n_failed > 0:
        logger.error(f'Failed to include {n_failed}/{DS.size_all + n_failed} matching masks: {failed_masks_ids}.')

    # Save dataset
    DS.save()
    logger.debug(f'Saved dataset, id={DS.id}')

    return DS
