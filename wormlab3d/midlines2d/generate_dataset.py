import random
from collections import OrderedDict

from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import TagInfo, DatasetMidline2D
from wormlab3d.midlines2d.args import DatasetArgs


def generate_dataset(args: DatasetArgs, fix_frames: bool = False) -> DatasetMidline2D:
    """
    Generate a 2D midline dataset.
    """
    logger.info(
        f'Generating dataset. '
        f'Train/test split={args.train_test_split}. '
        f'Restrict tags={args.restrict_tags}. '
        f'Restrict concs={args.restrict_concs}.'
    )
    DS = DatasetMidline2D(
        train_test_split_target=args.train_test_split,
        restrict_tags=args.restrict_tags,
        restrict_concs=args.restrict_concs,
        centre_3d_max_error=args.centre_3d_max_error,
        exclude_experiments=args.exclude_experiments,
        include_experiments=args.include_experiments,
        exclude_trials=args.exclude_trials,
        include_trials=args.include_trials,
    )

    # Fetch manually-annotated midlines, grouped by tags (each might appear for multiple tags)
    # The query starts matching on the midline2d collection.
    logger.info('Querying database')
    pipeline = [
        {'$match': {'user': {'$exists': True}}},
        {'$lookup': {'from': 'frame', 'localField': 'frame', 'foreignField': '_id', 'as': 'frame'}},
        {'$unwind': {'path': '$frame'}},
        {'$project': {
            '_id': 1,
            'experiment_id': '$frame.experiment',
            'trial_id': '$frame.trial',
            'tags': '$frame.tags',
            'reprojection_error': '$frame.centre_3d.error'
        }},
    ]

    # Filter by centre-point reprojection error
    matches = []
    if args.centre_3d_max_error > 0:
        matches.append({'reprojection_error': {'$lte': args.centre_3d_max_error}})

    # Restrict to frames which have been tagged as ANY of the given tags
    if len(args.restrict_tags) > 0:
        matches.append({'tags': {'$elemMatch': {'$in': args.restrict_tags}}})

    # Include/exclude experiments/trials
    if len(args.include_experiments) > 0:
        matches.append({'experiment_id': {'$in': args.include_experiments}})
    if len(args.exclude_experiments) > 0:
        matches.append({'experiment_id': {'$nin': args.exclude_experiments}})
    if len(args.include_trials) > 0:
        matches.append({'trial_id': {'$in': args.include_trials}})
    if len(args.exclude_trials) > 0:
        matches.append({'trial_id': {'$nin': args.exclude_trials}})

    # Add matches to pipeline
    if len(matches):
        pipeline.append({'$match': {'$and': matches}})

    # Restrict to concentrations (lookup against experiment)
    if len(args.restrict_concs) > 0:
        pipeline.extend([
            {'$lookup': {'from': 'experiment', 'localField': 'experiment_id', 'foreignField': '_id',
                         'as': 'experiment'}},
            {'$unwind': {'path': '$experiment'}},
            {'$match': {'experiment.concentration': {'$in': args.restrict_concs}}},
            {'$project': {'_id': 1, 'tags': 1}},
        ])

    # Unwind and then group by tag_id, with each tag collecting all associated midlines
    pipeline.extend([
        {'$unwind': {'path': '$tags', 'preserveNullAndEmptyArrays': True}},
        {'$project': {'_id': 1, 'tag_id': '$tags'}},
        {'$group': {'_id': '$tag_id', 'n': {'$sum': 1}, 'ids': {'$addToSet': '$_id'}}},
        {'$lookup': {'from': 'tag', 'localField': '_id', 'foreignField': '_id', 'as': 'tag'}},
        {'$unwind': {'path': '$tag', 'preserveNullAndEmptyArrays': True}},
        {'$project': {'_id': 0, 'tag_id': '$_id', 'name': '$tag.name', 'n': 1, 'ids': 1}},
        {'$sort': {'n': 1}},
    ])
    cursor = Midline2D.objects().aggregate(pipeline)

    # Process results
    logger.info('Processing results')
    tags_info = OrderedDict()
    failed_midline_ids = []
    for tag in cursor:
        # Check the frame has the necessary details, try to generate if not
        midline_ids = []
        for midline_id in tag['ids']:
            midline = Midline2D.objects.no_cache().get(id=midline_id)
            frame = midline.frame
            logger.debug(f'--------------- Checking frame id={frame.id}')
            try:
                if fix_frames:
                    if not frame.centres_2d_available():
                        logger.warn('Frame does not have 2d centre points available for all views, generating now.')
                        frame.generate_centres_2d()
                        frame.save()
                    if frame.centre_3d is None:
                        logger.warn('Frame does not have 3d centre point available, generating now.')
                        frame.generate_centre_3d()
                        frame.save()
                    if len(frame.images) != 3:
                        logger.warn(f'Frame (id={frame.id}) does not have prepared images, generating now.')
                        frame.generate_prepared_images()
                        frame.save()

                assert len(frame.images) == 3, 'Frame does not have prepared images.'
                assert midline.frame.centre_3d is not None, 'Frame does not have 3d centre point available.'
                assert len(midline.get_prepared_coordinates()) > 1, 'Midline coordinates empty after crop.'

                midline_ids.append(midline_id)
            except AssertionError as e:
                failed_midline_ids.append(midline_id)
                if fix_frames:
                    logger.error(f'Failed to prepare frame: {e}')
                else:
                    logger.error(f'Frame is not ready: {e}')

        n = len(midline_ids)
        if n > 0:
            # Shuffle the midline ids
            random.shuffle(midline_ids)

            # Set target train/test splits
            target_train = int(n * args.train_test_split)
            target_test = n - target_train

            # Set name for untagged midlines
            if 'name' not in tag:
                tag['name'] = 'UNTAGGED'

            tags_info[tag['tag_id']] = {
                'name': tag['name'],
                'n': n,
                'ids_to_assign': midline_ids,
                'ids_train': [],
                'ids_test': [],
                'n_target_train': target_train,
                'n_target_test': target_test,
            }

    # Train/test split
    for i, (tag_id, tag_info) in enumerate(tags_info.items()):
        ids_to_assign = tag_info['ids_to_assign'].copy()
        # Loop over the midlines, and assign them to train or test proportionally
        for midline_id in ids_to_assign:
            # This tag has fewer test midlines than we want, so add to test
            if len(tag_info['ids_test']) < tag_info['n_target_test']:
                tt = 'test'
            else:
                tt = 'train'

            # Move the midline id to the relevant list
            tag_info[f'ids_{tt}'].append(midline_id)
            tag_info['ids_to_assign'].remove(midline_id)

            # Now loop over all the other tags to assign this frame (if present) in the same way
            for j, (tag_id2, tag_info2) in enumerate(tags_info.items()):
                if tag_id == tag_id2:
                    continue

                # This midline should not be present for any previous or subsequent tags
                # as if it had been then it should have already been assigned in this tag.
                assert midline_id not in tag_info2['ids_train']
                assert midline_id not in tag_info2['ids_test']

                # Frame is to be assigned in a subsequent tag, so assign in the same way
                if j > i and midline_id in tag_info2['ids_to_assign']:
                    tag_info2[f'ids_{tt}'].append(midline_id)
                    tag_info2['ids_to_assign'].remove(midline_id)

    # Group midlines into train/test sets and collate tag info
    ids_train = []
    ids_test = []
    infos = []
    for tag_id, tag_info in tags_info.items():
        # Check no midlines are left to assign and all the numbers add up
        assert len(tag_info['ids_to_assign']) == 0
        assert len(tag_info['ids_train']) + len(tag_info['ids_test']) == tag_info['n']
        ids_train.extend(tag_info['ids_train'])
        ids_test.extend(tag_info['ids_test'])

        # Tag info to save to database
        info = TagInfo(
            tag=tag_id,
            name=tag_info['name'],
            n=tag_info['n'],
            n_train=len(tag_info['ids_train']),
            n_test=len(tag_info['ids_test']),
            n_target_train=tag_info['n_target_train'],
            n_target_test=tag_info['n_target_test'],
        )
        n = info.n_train + info.n_test
        if n > 0:
            info.split = info.n_train / n
        infos.append(info)
    DS.tag_info = infos

    # De-duplicate
    ids_train = list(set(ids_train))
    ids_test = list(set(ids_test))
    DS.set_data(train=ids_train, test=ids_test)

    n_failed = len(failed_midline_ids)
    if n_failed > 0:
        logger.error(f'Failed to include {n_failed}/{DS.size_all + n_failed} matching midlines: {failed_midline_ids}')

    # Save dataset
    logger.debug('Saving dataset')
    DS.save()

    return DS
