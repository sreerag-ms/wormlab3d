from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.widgets import Slider

from simple_worm.controls import ControlSequenceNumpy
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, cla, MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import FrameSequence, SwRun


def interactive_plots():
    """Puts matplotlib into interactive mode by switching to the Qt5 backend."""
    import matplotlib
    gui_backend = 'Qt5Agg'
    matplotlib.use(gui_backend, force=True)


def clear_axes(ax):
    """Removes the ticks from a 3D plot"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])


def tex_mode():
    """Use latex font rendering."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica']})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


class CameraImageArtist:
    """
    Draw camera images and midline coordinates.
    """

    midline_opt_defaults = {
        's': 5,
        'alpha': 0.9
    }

    def __init__(
            self,
            images: np.ndarray,
            coords: List[np.ndarray],
            midline_opts: dict = None,
            cmap_name: str = MIDLINE_CMAP_DEFAULT
    ):
        self.images = images
        self.coords = coords
        if midline_opts is None:
            midline_opts = {}
        self.midline_opts = {**CameraImageArtist.midline_opt_defaults, **midline_opts}

        # Camera images
        self.c0: AxesImage = None
        self.c1: AxesImage = None
        self.c2: AxesImage = None

        # Midline scatter plots
        self.m0: PathCollection = None
        self.m1: PathCollection = None
        self.m2: PathCollection = None
        self.cmap = cm.get_cmap(cmap_name)

    def add_images(self, axc0: Axes, axc1: Axes, axc2: Axes):
        """
        Add the camera images to the given axes.
        """
        self.c0 = axc0.imshow(self.images[0], cmap='gray', vmin=0, vmax=1)
        self.c1 = axc1.imshow(self.images[1], cmap='gray', vmin=0, vmax=1)
        self.c2 = axc2.imshow(self.images[2], cmap='gray', vmin=0, vmax=1)
        axc0.axis('off')
        axc1.axis('off')
        axc2.axis('off')

    def add_midline_projections(self, axc0: Axes, axc1: Axes, axc2: Axes):
        """
        Add midline projections to the given axes.
        """
        c0, c1, c2 = self.coords[0], self.coords[1], self.coords[2]
        self.m0 = axc0.scatter(c0[:, 0], c0[:, 1], c=self._get_scatter_colors(c0), **self.midline_opts)
        self.m1 = axc1.scatter(c1[:, 0], c1[:, 1], c=self._get_scatter_colors(c1), **self.midline_opts)
        self.m2 = axc2.scatter(c2[:, 0], c2[:, 1], c=self._get_scatter_colors(c2), **self.midline_opts)

    def _get_scatter_colors(self, coords: np.ndarray):
        N = len(coords)
        fc = self.cmap((np.arange(N) + 0.5) / N)
        return fc

    def update(self, images: np.ndarray, coords: List[np.ndarray] = None):
        """
        Update the axes with the new images and midlines.
        """
        self.images = images
        self.c0.set_array(images[0])
        self.c1.set_array(images[1])
        self.c2.set_array(images[2])

        if coords is not None:
            # Need to update the colours as well as the offsets since the number of points may change.
            self.coords = coords
            c0, c1, c2 = self.coords[0], self.coords[1], self.coords[2]
            self.m0.set_offsets(c0)
            self.m0.set_facecolor(self._get_scatter_colors(c0))
            self.m1.set_offsets(c1)
            self.m1.set_facecolor(self._get_scatter_colors(c1))
            self.m2.set_offsets(c2)
            self.m2.set_facecolor(self._get_scatter_colors(c2))


def generate_interactive_3d_clip_with_projections(
        FS_db: FrameSequence = None,
        sim_run: SwRun = None
):
    """
    Generate an interactive video clip from a FrameSequence
    """
    assert FS_db is not None or sim_run is not None, \
        'Either a frame sequence or a simulation run must be passed.'

    interactive_plots()

    # Load from the database
    if sim_run is not None:
        FS_db = sim_run.frame_sequence
        FS = FrameSequenceNumpy(x=sim_run.FS.x, psi=sim_run.FS.psi, calculate_components=True)
        CS = ControlSequenceNumpy(**sim_run.CS.to_dict())

        # Create the midline projection sequence
        logger.debug('Fetching 2d coordinate projections.')
        MS = sim_run.get_prepared_2d_coordinates()
    else:
        x = FS_db.X.transpose(0, 2, 1)
        FS = FrameSequenceNumpy(x=x)
        CS = None

        # Create the midline projection sequence
        logger.debug('Fetching 2d coordinate projections.')
        MS = []
        for i, m in enumerate(FS_db.midlines):
            m.x = FS[i].x.T + FS_db.centre
            MS.append(m.get_prepared_2d_coordinates())

    # Load the trial
    trial = FS_db.trial
    fps = trial.fps

    # Load the camera image sequences
    IS = np.zeros((FS.n_frames, 3, *PREPARED_IMAGE_SIZE))
    for i, frame in enumerate(FS_db.frames):
        if not frame.is_ready():
            logger.warning(f'Frame #{frame.frame_num} is not ready! Preparing now...')
            frame.trial = trial  # Use the same trial so it doesn't keep reloading the reader each frame
            frame.generate_prepared_images()
            frame.save()
        IS[i] = frame.images

    # Set up figure and axes
    fig = plt.figure(facecolor='white', figsize=(12, 12))

    # 4-tuple of floats *rect* = ``[left, bottom, width, height]``.
    ax3d = plt.axes([0.05, 0.4, 0.9, 0.6], projection='3d')
    axc1 = plt.axes([0.05, 0.1, 0.25, 0.25])
    axc2 = plt.axes([0.38, 0.1, 0.25, 0.25])
    axc3 = plt.axes([0.71, 0.1, 0.25, 0.25])
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])

    # Infos
    ax3d.text2D(
        -0.3, 0.8,
        f'Experiment: {trial.experiment.id}\n'
        f'Strain: {trial.experiment.strain}\n'
        f'Sex: {trial.experiment.sex}\n'
        f'Age: {trial.experiment.age}\n'
        f'Concentration: {trial.experiment.concentration}\n'
        f'\n'
        f'Trial: {trial.id}\n'
        f'Legacy id: {trial.legacy_id}\n'
        f'Date: {trial.date:%Y-%m-%d}\n'
        f'Trial #{trial.trial_num}\n'
        f'Frames: {FS_db.frames[0].frame_num}-{FS_db.frames[-1].frame_num}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax3d.transAxes
    )
    if sim_run is not None:
        ax3d.text2D(
            1.1, 0.8,
            f'Run: {sim_run.id}\n'
            f'Date: {sim_run.created:%Y-%m-%d %H:%M}\n'
            f'Checkpoint: {sim_run.checkpoint.id}\n'
            f'Step: {sim_run.checkpoint.step}\n'
            f'Loss: {sim_run.checkpoint.loss:.5f}\n'
            f'dt: {sim_run.sim_params.dt:.3f}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax3d.transAxes
        )
    axc1.set_title(trial.videos[0])
    axc2.set_title(trial.videos[1])
    axc3.set_title(trial.videos[2])

    # 3D midline axes
    mins, maxs = FS.get_bounding_box(zoom=1)
    cla(ax3d)
    if CS is None:
        fa = FrameArtist(FS[0], n_arrows=12)
    else:
        max_ab = max(np.abs(CS.alpha).max(), np.abs(CS.beta).max())
        fa = FrameArtist(FS[0], n_arrows=12, alpha_max=max_ab, beta_max=max_ab)
    fa.add_component_vectors(ax3d, draw_e0=False, C=CS[0] if CS is not None else None)
    fa.add_midline(ax3d)
    ax3d.set_xlim(mins[0], maxs[0])
    ax3d.set_ylim(mins[1], maxs[1])
    ax3d.set_zlim(mins[2], maxs[2])

    # Camera views
    ca = CameraImageArtist(IS[0], MS[0])
    ca.add_images(axc1, axc2, axc3)
    ca.add_midline_projections(axc1, axc2, axc3)

    # Animation controls
    time_slider = Slider(ax_slider, 'Frame', 0, FS.n_frames - 1, valinit=0, valstep=1)
    is_manual = False  # True if user has taken control of the animation

    def update_frame(frame_num: int):
        """Update the midline and camera views."""
        fa.update(FS[frame_num], C=CS[frame_num] if CS is not None else None)
        ca.update(IS[frame_num], MS[frame_num])
        return ()

    def update_slider(frame_num: int):
        """Handle clicks in the slider to manually change the frame."""
        nonlocal is_manual
        is_manual = True
        update_frame(frame_num)

    def update_plot(frame_num: int):
        """Callback to update the plot."""
        nonlocal is_manual
        if is_manual:
            return ()
        val = frame_num % FS.n_frames
        time_slider.set_val(val)
        is_manual = False
        return ()

    def on_click(event: Event):
        """Handle clicks on the canvas to toggle pause."""
        # Check where the click happened
        (xm, ym), (xM, yM) = time_slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider.
            return
        else:
            # User clicked somewhere else on canvas = toggle pause
            nonlocal is_manual
            if is_manual:
                is_manual = False
                ani.event_source.start()
            else:
                is_manual = True
                ani.event_source.stop()

    # Call update function on slider value change
    time_slider.on_changed(update_slider)
    fig.canvas.mpl_connect('button_press_event', on_click)
    ani = animation.FuncAnimation(fig, update_plot, interval=1 / fps)
    plt.show()
