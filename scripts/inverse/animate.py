import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import GridSpec

from simple_worm.controls import ControlSequenceNumpy
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, cla, MidpointNormalize, interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Trial, FrameSequence, SwRun, SwCheckpoint
from wormlab3d.simple_worm.args import FrameSequenceArgs
from wormlab3d.toolkit.plot_utils import generate_interactive_3d_clip_with_projections, CameraImageArtist
from wormlab3d.toolkit.util import build_target_arguments_parser

save_video = False
show_video = True


def get_simulation_run() -> SwRun:
    """
    Find a simulation run for a given trial and frame num, or by id.
    """
    parser = build_target_arguments_parser()
    parser.add_argument('--sim-params', type=str, help='Simulation parameters id.')
    parser.add_argument('--reg-params', type=str, help='Regularisation parameters id.')
    args = parser.parse_args()
    if args.sw_run:
        return SwRun.objects.get(id=args.sw_run)

    if args.sw_checkpoint:
        checkpoint = SwCheckpoint.objects.get(id=args.sw_checkpoint)

    else:
        if args.frame_sequence is None and (args.trial is None or args.frame_num is None or args.duration is None):
            raise RuntimeError('Either all of [trial, frame_num and duration] or a FrameSequence id must be specified.')

        # Look for existing FS
        if args.frame_sequence is not None:
            FS_db = FrameSequence.objects.get(id=args.frame_sequence)
        else:
            trial = Trial.objects.get(id=args.trial)
            n_frames = math.ceil(args.duration * trial.fps)
            fsa = FrameSequenceArgs(
                trial=args.trial,
                start_frame=args.frame_num,
                midline_source=args.midline3d_source
            )
            FS_db = FrameSequence.find_from_args(fsa, n_frames)
            n_results = FS_db.count()
            if n_results > 0:
                FS_db = FS_db[0]
                logger.info(
                    f'Found {n_results} matching frame sequences in database, using most recent. Id={FS_db.id}.')
            else:
                raise RuntimeError('Frame sequence matching arguments could not be found.')

        # Look for checkpoints
        filters = {'frame_sequence': FS_db}
        if args.sim_params is not None:
            filters['sim_params'] = args.sim_params
        if args.reg_params is not None:
            filters['reg_params'] = args.reg_params
        checkpoints = SwCheckpoint.objects(**filters).order_by('+loss')
        n_results = checkpoints.count()
        if n_results > 0:
            checkpoint = checkpoints[0]
            logger.info(f'Found {n_results} matching checkpoints in database, using lowest loss. Id={checkpoint.id}.')
        else:
            raise RuntimeError('No checkpoints found for frame sequence.')

    # Look for runs
    runs = SwRun.objects(checkpoint=checkpoint)
    n_results = runs.count()
    if n_results > 0:
        run = runs[0]
        logger.info(f'Found {n_results} matching runs in database, using most recent. Id={run.id}.')
    else:
        raise RuntimeError('No runs found for frame sequence and checkpoint.')

    return run


def generate_3d_clip_with_projections(
        sim_run: SwRun = None,
        fps: int = 25,
):
    # Load from the database
    FS_db = sim_run.frame_sequence
    FS = FrameSequenceNumpy(x=sim_run.FS.x, psi=sim_run.FS.psi, calculate_components=True)
    CS = ControlSequenceNumpy(**sim_run.CS.to_dict())

    # Create the midline projection sequence
    logger.debug('Fetching 2d coordinate projections.')
    MS = sim_run.get_prepared_2d_coordinates()

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
        6, 6,
        top=0.95,
        bottom=0.01,
        left=0.02,
        right=0.98
    )

    ax3d = fig.add_subplot(gs[:4, :4], projection='3d')
    axc1 = fig.add_subplot(gs[0:2, 4:6])
    axc2 = fig.add_subplot(gs[2:4, 4:6])
    axc3 = fig.add_subplot(gs[4:6, 4:6])
    ax_alpha = fig.add_subplot(gs[4, :4])
    ax_beta = fig.add_subplot(gs[5, :4])

    axc1.set_title('Camera 0')
    axc2.set_title('Camera 1')
    axc3.set_title('Camera 2')

    # 3D midline axes
    mins, maxs = FS.get_bounding_box(zoom=1)
    cla(ax3d)
    if 1 or CS is None:
        fa = FrameArtist(FS[0], n_arrows=0)
    else:
        max_ab = max(np.abs(CS.alpha).max(), np.abs(CS.beta).max())
        fa = FrameArtist(FS[0], n_arrows=0, alpha_max=max_ab, beta_max=max_ab)
    fa.add_component_vectors(ax3d, draw_e0=False, C=CS[0] if CS is not None else None)
    fa.add_midline(ax3d)
    ax3d.set_xlim(mins[0], maxs[0])
    ax3d.set_ylim(mins[1], maxs[1])
    ax3d.set_zlim(mins[2], maxs[2])

    # Aspects
    azims = np.linspace(start=0, stop=360, num=len(FS_db.frames))
    ax3d.view_init(azim=azims[0])  # elev

    # Camera views
    ca = CameraImageArtist(IS[0], MS[0])
    ca.add_images(axc1, axc2, axc3)
    ca.add_midline_projections(axc1, axc2, axc3)

    # Controls
    Ms = [CS.controls['alpha'], CS.controls['beta']]
    cmap = plt.cm.PRGn
    vmin = min(CS.alpha.min(), CS.beta.min())
    vmax = max(CS.alpha.max(), CS.beta.max())
    time_indicators = []
    for i, M in enumerate(Ms):
        ax = [ax_alpha, ax_beta][i]
        ax.matshow(
            M.T,
            cmap=cmap,
            clim=(vmin, vmax),
            norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
            aspect='auto'
        )
        ax.set_title(['$\\alpha$', '$\\beta$'][i])
        ax.text(-0.01, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                fontweight='bold')
        ax.text(-0.01, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                fontweight='bold')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Add time indicator
        t = ax.axvline(x=0, color='red', linewidth=2)
        time_indicators.append(t)

    def update(frame_num: int):
        """Update the midline and camera views."""
        fa.update(FS[frame_num], C=CS[frame_num] if CS is not None else None)
        ca.update(IS[frame_num], MS[frame_num])
        ax3d.view_init(azim=azims[frame_num])
        for t in time_indicators:
            t.set_xdata(frame_num)
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
             f'_run={sim_run.id}' \
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
    interactive()
    generate_3d_clip_with_projections(sim_run=get_simulation_run(), fps=15)


def generate_interactive_scatter_clip():
    """
    Generate an interactive video clip from a simulation run.
    """
    generate_interactive_3d_clip_with_projections(
        sim_run=get_simulation_run()
    )


if __name__ == '__main__':
    # generate_interactive_scatter_clip()
    generate_video()
