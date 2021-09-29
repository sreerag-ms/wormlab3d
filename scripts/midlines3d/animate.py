import math

from wormlab3d import logger
from wormlab3d.data.model import Trial, FrameSequence
from wormlab3d.simple_worm.args import FrameSequenceArgs
from wormlab3d.toolkit.plot_utils import generate_interactive_3d_clip_with_projections
from wormlab3d.toolkit.util import parse_target_arguments


def get_frame_sequence() -> FrameSequence:
    """
    Find a frame-sequence for a given trial and frame num, or by id.
    If one can't be found then it will be created and saved to the database.
    """
    args = parse_target_arguments()
    if args.frame_sequence:
        return FrameSequence.objects.get(id=args.frame_sequence)

    if args.trial is None or args.frame_num is None or args.duration is None:
        raise RuntimeError('Either all of [trial, frame_num and duration] or a FrameSequence id must be specified.')
    trial = Trial.objects.get(id=args.trial)
    n_frames = math.ceil(args.duration * trial.fps)

    # Look for existing FS
    fsa = FrameSequenceArgs(
        trial=args.trial,
        start_frame=args.frame_num,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
    )
    FS_db = FrameSequence.find_from_args(fsa, n_frames)
    if FS_db.count() > 0:
        logger.info(f'Found {len(FS_db)} matching frame sequences in database, using most recent.')
        FS_db = FS_db[0]
        return FS_db

    # Create a new one
    logger.info(f'Found no matching frame sequences in database, creating new.')
    f0 = args.frame_num
    fn = f0 + n_frames
    frame_nums = range(f0, fn)
    seq = []
    for frame_num in frame_nums:
        frame = trial.get_frame(frame_num)
        logger.debug(f'Loading 3D midline for frame #{frame_num} (id={frame.id}).')
        filters = {'source': args.midline3d_source}
        if args.midline3d_source_file is not None:
            filters['source_file'] = args.midline3d_source_file
        midlines = frame.get_midlines3d(filters)
        if len(midlines) > 1:
            logger.info(
                f'Found {len(midlines)} 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}. Picking with lowest error..')
            midline = midlines[0]
        elif len(midlines) == 1:
            midline = midlines[0]
        else:
            raise RuntimeError(f'Could not find any 3D midlines for trial_id = {trial.id}, frame_num = {frame_num}.')
        seq.append(midline)

    FS_db = FrameSequence()
    FS_db.set_from_sequence(seq)
    FS_db.save()
    logger.info('Saved FS to database')

    return FS_db


def generate_interactive_scatter_clip():
    """
    Generate an interactive video clip from a FrameSequence.
    """
    FS_db = get_frame_sequence()
    generate_interactive_3d_clip_with_projections(FS_db)


if __name__ == '__main__':
    generate_interactive_scatter_clip()
