import random
from collections import OrderedDict

from pymongo.command_cursor import CommandCursor

from wormlab3d import logger
from wormlab3d.data.model.dataset import TagInfo, Dataset
from wormlab3d.nn.args import DatasetArgs


def build_pipeline(args: DatasetArgs) -> list:
    """
    Build a pipeline to fetch items, filtered as needed and grouped by tags (each might appear for multiple tags).
    Assumes that the pipeline will be executed against a collection which has a "frame" reference, or at least
    that field must be available at the point this pipeline is inserted.
    """
    pipeline = [
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

    # Unwind and then group by tag_id, with each tag collecting all associated
    pipeline.extend([
        {'$unwind': {'path': '$tags', 'preserveNullAndEmptyArrays': True}},
        {'$project': {'_id': 1, 'tag_id': '$tags'}},
        {'$group': {'_id': '$tag_id', 'n': {'$sum': 1}, 'ids': {'$addToSet': '$_id'}}},
        {'$lookup': {'from': 'tag', 'localField': '_id', 'foreignField': '_id', 'as': 'tag'}},
        {'$unwind': {'path': '$tag', 'preserveNullAndEmptyArrays': True}},
        {'$project': {'_id': 0, 'tag_id': '$_id', 'name': '$tag.name', 'n': 1, 'ids': 1}},
        {'$sort': {'n': 1}},
    ])

    return pipeline


def build_dataset(
        cursor: CommandCursor,
        args: DatasetArgs,
        validation_callback: callable = None
) -> Dataset:
    """
    Fetch the results from the database, validate, and split the dataset into test and train subsets.
    """
    logger.info('Processing results.')
    tags_info = OrderedDict()
    for tag in cursor:
        # Check the frame has the necessary details, try to generate if not
        document_ids = []
        for doc_id in tag['ids']:
            if validation_callback is None or validation_callback(doc_id) is True:
                document_ids.append(doc_id)

        n = len(document_ids)
        if n > 0:
            # Shuffle the ids
            random.shuffle(document_ids)

            # Set target train/test splits
            target_train = int(n * args.train_test_split)
            target_test = n - target_train

            # Set name for untagged documents
            if 'name' not in tag:
                tag['name'] = 'UNTAGGED'

            tags_info[tag['tag_id']] = {
                'name': tag['name'],
                'n': n,
                'ids_to_assign': document_ids,
                'ids_train': [],
                'ids_test': [],
                'n_target_train': target_train,
                'n_target_test': target_test,
            }

    logger.info('Splitting valid results into train and test sets.')
    for i, (tag_id, tag_info) in enumerate(tags_info.items()):
        logger.debug(f'Splitting tag id = {tag_id}')
        ids_to_assign = tag_info['ids_to_assign'].copy()
        # Loop over the ids, and assign them to train or test proportionally
        for doc_id in ids_to_assign:
            # This tag has fewer test ids than we want, so add to test
            if len(tag_info['ids_test']) < tag_info['n_target_test']:
                tt = 'test'
            else:
                tt = 'train'

            # Move the document id to the relevant list
            tag_info[f'ids_{tt}'].append(doc_id)
            tag_info['ids_to_assign'].remove(doc_id)

            # Now loop over all the other tags to assign this frame (if present) in the same way
            for j, (tag_id2, tag_info2) in enumerate(tags_info.items()):
                if tag_id == tag_id2:
                    continue

                # This document should not be present for any previous or subsequent tags
                # as if it had been then it should have already been assigned in this tag.
                assert doc_id not in tag_info2['ids_train']
                assert doc_id not in tag_info2['ids_test']

                # Document is waiting to be assigned in a subsequent tag, so assign now in the same way
                if j > i and doc_id in tag_info2['ids_to_assign']:
                    tag_info2[f'ids_{tt}'].append(doc_id)
                    tag_info2['ids_to_assign'].remove(doc_id)

    # Group documents into train/test sets and collate tag info
    logger.info('Final validation pass.')
    ids_train = []
    ids_test = []
    infos = []
    for tag_id, tag_info in tags_info.items():
        logger.debug(f'Validating tag id = {tag_id}')
        # Check no documents are left to assign and all the numbers add up
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

    # Build dataset
    logger.debug('Building dataset.')
    DS = Dataset.from_args(args)
    DS.tag_info = infos

    # De-duplicate
    logger.debug('Removing any duplicates.')
    ids_train = list(set(ids_train))
    ids_test = list(set(ids_test))
    logger.debug('Setting data')
    DS.set_data(train=ids_train, test=ids_test)

    return DS
