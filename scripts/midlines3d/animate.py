import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import GridSpec

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, cla
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Trial, FrameSequence
from wormlab3d.simple_worm.args import FrameSequenceArgs
from wormlab3d.toolkit.plot_utils import CameraImageArtist, interactive_plots
from wormlab3d.toolkit.plot_utils import generate_interactive_3d_clip_with_projections
from wormlab3d.toolkit.util import parse_target_arguments

save_video = True
show_video = True


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


def generate_3d_clip_with_projections(
        FS_db: FrameSequence,
        fps: int = 25,
):
    if show_video:
        interactive_plots()

    # Load from the database
    x = FS_db.X.transpose(0, 2, 1)
    FS = FrameSequenceNumpy(x=x)

    # Create the midline projection sequence
    logger.debug('Fetching 2d coordinate projections.')
    MS = []
    for i, m in enumerate(FS_db.midlines):
        m.x = FS[i].x.T + FS_db.centre
        MS.append(m.get_prepared_2d_coordinates())

    # Load the trial
    trial = FS_db.trial

    # Pick frames closest to the timesteps used in the simulation
    frames = FS_db.frames
    n_frames = len(frames)
    n_timesteps = FS.x.shape[0]
    if n_frames != n_timesteps:
        frame_idxs = np.arange(0, n_frames, n_frames / n_timesteps).astype(np.int32)
    else:
        frame_idxs = np.arange(n_frames)

    # Load the camera image sequences
    IS = np.zeros((FS.n_frames, 3, trial.crop_size, trial.crop_size))
    for i, frame_idx in enumerate(frame_idxs):
        frame = frames[frame_idx]
        if not frame.is_ready():
            logger.warning(f'Frame #{frame.frame_num} is not ready! Preparing now...')
            frame.trial = trial  # Use the same trial so it doesn't keep reloading the reader each frame
            frame.generate_prepared_images()
            frame.save()
        IS[i] = frame.images

    # Set up figure and axes
    fig = plt.figure(facecolor='white', figsize=(12, 12))
    gs = GridSpec(
        3, 3,
        top=0.95,
        bottom=0.01,
        left=0.02,
        right=0.98
    )

    ax3d = fig.add_subplot(gs[:, :2], projection='3d')
    axc1 = fig.add_subplot(gs[0, 2])
    axc2 = fig.add_subplot(gs[1, 2])
    axc3 = fig.add_subplot(gs[2, 2])

    axc1.set_title('Camera 0')
    axc2.set_title('Camera 1')
    axc3.set_title('Camera 2')

    # 3D midline axes
    mins, maxs = FS.get_bounding_box(zoom=1)
    cla(ax3d)
    fa = FrameArtist(FS[0], n_arrows=0)
    fa.add_midline(ax3d)
    ax3d.set_xlim(mins[0], maxs[0])
    ax3d.set_ylim(mins[1], maxs[1])
    ax3d.set_zlim(mins[2], maxs[2])

    # Aspects
    n_rotations = 1
    azims = np.linspace(start=0, stop=360 * n_rotations, num=len(FS_db.frames))
    ax3d.view_init(azim=azims[0])  # elev

    # Camera views
    ca = CameraImageArtist(IS[0], MS[0])
    ca.add_images(axc1, axc2, axc3)
    ca.add_midline_projections(axc1, axc2, axc3)

    def update(frame_num: int):
        """Update the midline and camera views."""
        fa.update(FS[frame_num])
        ca.update(IS[frame_num], MS[frame_num])
        ax3d.view_init(azim=azims[frame_num])
        return ()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=FS.n_frames,
        blit=True,
        interval=1 / fps
    )

    if save_video:
        os.makedirs(LOGS_PATH, exist_ok=True)
        fn = f'trial={trial.id}' \
             f'_frames={FS_db.frames[0].frame_num}-{FS_db.frames[-1].frame_num}' \
             f'_{FS_db.source}' \
             f'_fps={fps}'
        metadata = dict(
            title=fn,
            artist='WormLab Leeds'
        )
        save_path = LOGS_PATH / (START_TIMESTAMP + '_' + fn + '.mp4')
        logger.info(f'Saving animation to {save_path}.')
        ani.save(save_path, writer='ffmpeg', fps=fps, metadata=metadata)

    if show_video:
        plt.show()


def generate_video():
    """
    Generate a video with rotating 3D view, camera images with overlaid projections and alpha/beta.
    """
    FS_db = get_frame_sequence()
    generate_3d_clip_with_projections(FS_db, fps=15)


def generate_interactive_scatter_clip():
    """
    Generate an interactive video clip from a FrameSequence.
    """
    FS_db = get_frame_sequence()
    generate_interactive_3d_clip_with_projections(FS_db)


if __name__ == '__main__':
    # generate_interactive_scatter_clip()
    generate_video()
