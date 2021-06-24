import math

from wormlab3d import logger
from wormlab3d.data.model import Trial, FrameSequence, SwRun, SwCheckpoint
from wormlab3d.simple_worm.args import FrameSequenceArgs
from wormlab3d.toolkit.plot_utils import generate_interactive_3d_clip_with_projections
from wormlab3d.toolkit.util import parse_target_arguments


def get_simulation_run() -> SwRun:
    """
    Find a simulation run for a given trial and frame num, or by id.
    """
    args = parse_target_arguments()
    if args.sw_run:
        return SwRun.objects.get(id=args.sw_run)

    if args.trial is None or args.frame_num is None or args.duration is None:
        raise RuntimeError('Either all of [trial, frame_num and duration] or a FrameSequence id must be specified.')
    trial = Trial.objects.get(id=args.trial)
    n_frames = math.ceil(args.duration * trial.fps)

    # Look for existing FS
    fsa = FrameSequenceArgs(
        trial=args.trial,
        start_frame=args.frame_num,
        midline_source=args.midline3d_source
    )
    FS_db = FrameSequence.find_from_args(fsa, n_frames)
    if FS_db.count() > 0:
        logger.info(f'Found {len(FS_db)} matching frame sequences in database, using most recent.')
        FS_db = FS_db[0]
    else:
        raise RuntimeError('Frame sequence matching arguments could not be found.')

    # Look for checkpoints
    checkpoints = SwCheckpoint.objects(frame_sequence=FS_db).order_by('+loss')
    if checkpoints.count() > 0:
        logger.info(f'Found {len(checkpoints)} matching checkpoints in database, using lowest loss.')
        checkpoint = checkpoints[0]
    else:
        raise RuntimeError('No checkpoints found for frame sequence.')

    # Look for runs
    runs = SwRun.objects(checkpoint=checkpoint)
    if runs.count() > 0:
        logger.info(f'Found {len(runs)} matching runs in database, using most recent.')
        run = runs[0]
    else:
        raise RuntimeError('No runs found for frame sequence and checkpoint.')

    return run


def generate_interactive_scatter_clip():
    """
    Generate an interactive video clip from a simulation run.
    """
    run = get_simulation_run()

    generate_interactive_3d_clip_with_projections(
        sim_run=run
    )


if __name__ == '__main__':
    generate_interactive_scatter_clip()
