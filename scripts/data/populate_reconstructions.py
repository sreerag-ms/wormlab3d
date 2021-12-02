from typing import List, Tuple

import numpy as np

from wormlab3d import logger
from wormlab3d.data.model import Trial, Frame
from wormlab3d.data.model.reconstruction import Reconstruction


def fetch_reconstructed_frames(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Load trial midlines from the database
    """

    # Filter the midlines by appropriate source
    midline_matches = {'midline.source': midline_source}
    if midline_source_file is not None:
        midline_matches['midline.source_file'] = midline_source_file

    pipeline = [
        {'$match': {'trial': trial_id}},
        {'$lookup': {'from': 'midline3d', 'localField': '_id', 'foreignField': 'frame', 'as': 'midline'}},
        {'$unwind': {'path': '$midline'}},
        {'$match': midline_matches},
        {'$project': {
            '_id': 0,
            'frame_id': '$_id',
            'frame_num': '$frame_num',
            'midline_id': '$midline._id',
            'midline_error': '$midline.error',
        }},
        {'$sort': {'frame_num': 1, 'midline_error': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

    # Fetch results
    frame_ids = []
    frame_nums = []
    midline_ids = []
    for res in cursor:
        # Check frame id is not the same
        frame_id = res['frame_id']
        if len(frame_ids) and frame_id == frame_ids[-1]:
            logger.warning(f'Multiple results returned for frame = {frame_id}. Skipping.')
            continue
        frame_ids.append(frame_id)

        # Check that the frame number is increasing
        frame_num = res['frame_num']
        if len(frame_nums) and frame_num != frame_nums[-1] + 1:
            logger.warning(f'Breakage in reconstruction sequence, stopping here.')
            break
        frame_nums.append(frame_num)
        midline_ids.append(res['midline_id'])

    n_results = len(midline_ids)
    if n_results > 0:
        logger.info(f'Found reconstruction for frames {frame_nums[0]}-{frame_nums[-1]} ({n_results}).')
    else:
        logger.warning('No reconstructed frames found!')
    assert len(frame_nums) == len(midline_ids)

    return frame_nums, midline_ids


def populate_reconstructions():
    trial_ids = []
    for trial in Trial.objects:
        trial_ids.append(trial.id)

    for trial_id in trial_ids:
        logger.info(f'Checking reconstructions for trial={trial_id}.')

        # Find what types of reconstructions have been attempted
        pipeline = [
            {'$match': {'trial': trial_id}},
            {'$lookup': {'from': 'midline3d', 'localField': '_id', 'foreignField': 'frame', 'as': 'midline'}},
            {'$unwind': {'path': '$midline'}},
            {'$group': {
                '_id': {'source': '$midline.source', 'source_file': '$midline.source_file', 'model': '$midline.model'},
                'count': {'$sum': 1},
            }},
        ]
        cursor = Frame.objects().aggregate(pipeline)
        attempt_types = list(cursor)

        if len(attempt_types) == 0:
            logger.info('No reconstructions found.')
            continue

        for res in attempt_types:
            attempt_keys = {**{'source_file': None, 'model': None}, **res['_id']}
            logger.info(f'Checking for reconstructed frames for {attempt_keys}.')

            frame_nums, midline_ids = fetch_reconstructed_frames(
                trial_id=trial_id,
                midline_source=attempt_keys['source'],
                midline_source_file=attempt_keys['source_file'],
            )

            if len(frame_nums) == 0:
                continue

            start_frame = frame_nums[0]
            end_frame = frame_nums[-1]

            # Check for existing reconstruction
            existing = Reconstruction.objects(
                trial=trial_id,
                source=attempt_keys['source'],
                source_file=attempt_keys['source_file'],
                model=attempt_keys['model']
            )

            if existing.count() > 0:
                assert existing.count() == 1
                existing = existing[0]
                if start_frame != existing.start_frame or end_frame != existing.end_frame:
                    logger.info('Existing record found with different start/end frames - updating.')
                    existing.start_frame = start_frame
                    existing.end_frame = end_frame
                    existing.save()
                else:
                    logger.info('Existing record found.')
                    continue

            reconst = Reconstruction(
                trial=trial_id,
                start_frame=start_frame,
                end_frame=end_frame,
                midlines=midline_ids,
                source=attempt_keys['source'],
                source_file=attempt_keys['source_file'],
                model=attempt_keys['model'],
            )
            reconst.save()
            logger.info(f'Added new reconstruction record id={reconst.id}.')


if __name__ == '__main__':
    populate_reconstructions()
