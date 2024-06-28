import itertools
from typing import Callable, List, Optional, Tuple, Union

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from mayavi import mlab
from mayavi.core.scene import Scene
from mayavi.modules.surface import Surface
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

from simple_worm.controls import ControlSequenceNumpy
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, MIDLINE_CMAP_DEFAULT, cla
from wormlab3d import logger
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
        'font.sans-serif': ['Arial']})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def equal_aspect_ratio(ax: Axes):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))


def to_rgb(c: Union[str, np.ndarray]):
    if type(c) == str:
        return mcolors.to_rgb(c)
    return c


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
    IS = np.zeros((FS.n_frames, 3, trial.crop_size, trial.crop_size))
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


def fancy_dendrogram(ax: Axes, *args, **kwargs) -> dict:
    """
    Plot a fancy dendogram.
    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        ax.set_title('Hierarchical Clustering Dendrogram (truncated)')
        ax.set_xlabel('Sample index or (cluster size)')
        ax.set_ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                            textcoords='offset points',
                            va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k', linestyle='--')

    return ddata


def reorder_distance_correlations(
        distances: np.ndarray,
        clusters: np.ndarray,
        order_by_size: bool = False
) -> Tuple[np.ndarray, List[Rectangle], np.ndarray]:
    """
    Reorder the distance correlations.
    """
    cluster_nums, counts = np.unique(clusters, return_counts=True)
    sorted_idxs = np.argsort(counts)[::-1]
    sorted_cluster_nums = cluster_nums[sorted_idxs]
    reordered_distances = np.zeros_like(distances)
    all_idxs = []
    squares = []
    block_idx = 0
    for i in range(len(cluster_nums)):
        if order_by_size:
            cluster_num = sorted_cluster_nums[i]
        else:
            cluster_num = cluster_nums[i]
        idxs = (clusters == cluster_num).nonzero()[0]
        cluster_size = len(idxs)
        reordered_distances[block_idx:(block_idx + cluster_size)] = distances[idxs].copy()
        all_idxs.append(idxs)
        sqr = Rectangle((block_idx - 0.5, block_idx - 0.5), cluster_size, cluster_size,
                        linewidth=0.5, edgecolor='r', linestyle='--', facecolor='none')
        squares.append(sqr)
        block_idx += cluster_size

    # Reorder columns
    all_idxs = np.concatenate(all_idxs)
    reordered_distances[:, :] = reordered_distances[:, all_idxs]

    return reordered_distances, squares, all_idxs


def plot_reordered_distances(ax: Axes, distances: np.ndarray, clusters: np.ndarray):
    """
    Reorder and plot a distances matrix.
    """
    cluster_nums = np.unique(clusters)
    reordered_distances, squares, all_idxs = reorder_distance_correlations(distances, clusters)

    # Plot reordered distances
    ax.matshow(reordered_distances, cmap=plt.cm.Blues)
    ax.set_title(f'Number of clusters = {len(cluster_nums)}.')

    # Show clusters on plot
    for sqr in squares:
        ax.add_patch(sqr)


def make_3d_posture_plot_for_animation(
        X_full: np.ndarray,
        width: int,
        height: int,
        show_axis: bool = True,
        show_ticks: bool = True,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D posture plot.
    Returns an update function to call which updates the worm.
    """

    # Get axis size limits and trajectory points
    X_centred = X_full - X_full.mean(axis=1, keepdims=True)
    size_limit = max(X_centred.max(axis=(0, 1)) - X_centred.min(axis=(0, 1))) * 1.
    traj_points = X_full.mean(axis=1)

    # Create figure
    fig = plt.figure(figsize=(int(width / 2 / 100), int(height * 2 / 3 / 100)))
    ax = fig.add_subplot(projection='3d')

    # Add worm
    FS = FrameSequenceNumpy(x=X_full.transpose(0, 2, 1))
    fa = FrameArtist(F=FS[0])
    fa.add_midline(ax)

    # Setup axis
    if not show_axis:
        ax.axis('off')
    if not show_ticks:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def centre_axis(frame_idx: int):
        cp = traj_points[frame_idx]
        for i, axis in enumerate('xyz'):
            getattr(ax, f'set_{axis}lim')([cp[i] - size_limit / 2, cp[i] + size_limit / 2])

    def update(frame_idx: int):
        # Update the worm
        fa.update(FS[frame_idx])

        # Update the axis limits
        centre_axis(frame_idx)

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def make_box_faces_from_pca(
        X: np.ndarray,
        pca: PCA,
        scale: np.ndarray = np.array([1., 1., 1.]),
        use_extents: bool = False
) -> List[np.ndarray]:
    """
    Make a list of cuboid faces as 3D polygons centred at the centre of X
    with shape and orientation taken from the PCA components.
    """
    R = np.stack(pca.components_, axis=1)
    Xt = np.einsum('ij,bj->bi', R.T, X)
    dims = np.ptp(Xt, axis=0)
    centre = Xt.min(axis=0) + dims / 2

    if use_extents:
        v0, v1, v2 = np.eye(3) * scale * dims / 2
    else:
        v0, v1, v2 = np.eye(3) * scale * pca.explained_variance_ratio_

    polygons = []
    for i in range(3):
        va = [v0, v1, v2][i]
        vb = [v1, v2, v0][i]
        vc = [v2, v0, v1][i]

        for j in range(2):
            if j == 1:
                vc *= -1
            verts = np.zeros((4, 3))
            for k in range(3):
                verts[:, k] = [
                    centre[k] - va[k] - vb[k] - vc[k],
                    centre[k] + va[k] - vb[k] - vc[k],
                    centre[k] + va[k] + vb[k] - vc[k],
                    centre[k] - va[k] + vb[k] - vc[k]
                ]
            polygons.append(verts @ R.T)

    return polygons


def make_box_faces_from_dims(
        dims: np.ndarray,
        scale: np.ndarray = np.array([1., 1., 1.]),
) -> List[np.ndarray]:
    """
    Make a list of cuboid faces as 3D polygons from given dims, centred at the origin.
    """
    centre = np.zeros(3)
    v0, v1, v2 = np.eye(3) * scale * dims / 2
    polygons = []
    for i in range(3):
        va = [v0, v1, v2][i]
        vb = [v1, v2, v0][i]
        vc = [v2, v0, v1][i]

        for j in range(2):
            if j == 1:
                vc *= -1
            verts = np.zeros((4, 3))
            for k in range(3):
                verts[:, k] = [
                    centre[k] - va[k] - vb[k] - vc[k],
                    centre[k] + va[k] - vb[k] - vc[k],
                    centre[k] + va[k] + vb[k] - vc[k],
                    centre[k] - va[k] + vb[k] - vc[k]
                ]
            polygons.append(verts)

    return polygons


def make_box_from_pca(X: np.ndarray, pca: PCA, colour: str, scale: Tuple[float] = (1., 1., 1.)) -> Poly3DCollection:
    """
    Make a 3D polygon centred at the centre of X with shape and orientation taken from the PCA components.
    """
    polygons = make_box_faces_from_pca(X, pca, scale)
    plane = Poly3DCollection(polygons, alpha=0.2, facecolors=colour, edgecolors='dark' + colour)
    return plane


def make_box_from_pca_mlab(
        X: np.ndarray,
        pca: PCA = None,
        colour: str = None,
        opacity: float = 1.,
        dimensions: str = 'pca',
        scale: Union[int, float, Tuple[float]] = 1,
        draw_outline: bool = False,
        outline_colour: str = None,
        outline_opacity: float = 1.,
        outline_tube_radius: float = 0.01,
        fig: Scene = None,
) -> List[Surface]:
    """
    Make a cuboid using mayavi centred at the centre of X with orientation taken from the PCA components
    and dimensions either from the PCA singular values or the extents of the data X.
    """
    if pca is None:
        pca = PCA()
        pca.fit(X)
    if type(scale) == int or type(scale) == float:
        scale = np.array([1., 1., 1.])
    faces = make_box_faces_from_pca(X=X, pca=pca, scale=scale, use_extents=dimensions == 'extents')
    meshes = []
    for face in faces:
        x, y, z = face.T
        mesh = mlab.mesh([
            [x[0], x[3]],
            [x[1], x[2]],
        ], [
            [y[0], y[3]],
            [y[1], y[2]],
        ], [
            [z[0], z[3]],
            [z[1], z[2]],
        ],
            opacity=opacity,
            color=to_rgb(colour),
            figure=fig,
        )
        meshes.append(mesh)

    outline = []
    if draw_outline:
        lines = make_box_outline(X=X, pca=pca, scale=scale, use_extents=dimensions == 'extents')
        for l in lines:
            line_obj = mlab.plot3d(
                *l.T,
                figure=fig,
                opacity=outline_opacity,
                color=to_rgb(outline_colour),
                tube_radius=outline_tube_radius
            )
            outline.append(line_obj)

    return meshes, outline


def make_cuboid(
        dims: np.ndarray,
        colour: str = None,
        opacity: float = 1.,
        scale: Union[int, float, Tuple[float]] = 1,
        draw_outline: bool = False,
        outline_colour: str = None,
        outline_opacity: float = 1.,
        outline_tube_radius: float = 0.01,
        fig: Scene = None,
) -> List[Surface]:
    """
    Make a cuboid of given dimensions centred at the origin.
    """
    if isinstance(scale, int) or isinstance(scale, float):
        scale = np.array([1., 1., 1.])
    faces = make_box_faces_from_dims(dims=dims, scale=scale)
    meshes = []
    for face in faces:
        x, y, z = face.T
        mesh = mlab.mesh([
            [x[0], x[3]],
            [x[1], x[2]],
        ], [
            [y[0], y[3]],
            [y[1], y[2]],
        ], [
            [z[0], z[3]],
            [z[1], z[2]],
        ],
            opacity=opacity,
            color=to_rgb(colour),
            figure=fig,
        )
        meshes.append(mesh)

    outline = []
    if draw_outline:
        lines = make_box_outline(dims=dims, scale=scale)
        for l in lines:
            line_obj = mlab.plot3d(
                *l.T,
                figure=fig,
                opacity=outline_opacity,
                color=to_rgb(outline_colour),
                tube_radius=outline_tube_radius
            )
            outline.append(line_obj)

    return meshes, outline


def make_box_outline(
        X: Optional[np.ndarray] = None,
        dims: Optional[np.ndarray] = None,
        pca: PCA = None,
        scale: np.ndarray = np.array([1., 1., 1.]),
        use_extents: bool = False,
) -> List[np.ndarray]:
    """
    Make a cuboid outline using mayavi centred at the centre of X with orientation taken from the PCA components
    and dimensions either from the PCA singular values or the extents of the data X.
    """
    if dims is None:
        if pca is None:
            pca = PCA()
            pca.fit(X)

        # Get the bounds relative to the PCA components
        R = np.stack(pca.components_, axis=1)
        Xt = np.einsum('ij,bj->bi', R.T, X)
        dims = np.ptp(Xt, axis=0)
        centre = Xt.min(axis=0) + dims / 2

        if not use_extents:
            dims = pca.explained_variance_ratio_ * 2

    else:
        assert X is None, 'Either X or dims should be passed, not both.'
        centre = np.zeros(3)
        R = np.eye(3)

    dims *= scale

    # Calculate box vertices
    M = np.array(list(itertools.product(*[[-1, 1]] * 3)))
    v = (centre + M * dims[None, :] / 2) @ R.T

    # Bottom face outline
    l1 = np.stack([v[0], v[1]])
    l2 = np.stack([v[1], v[3]])
    l3 = np.stack([v[3], v[2]])
    l4 = np.stack([v[2], v[0]])

    # Top face outline
    l5 = np.stack([v[4], v[5]])
    l6 = np.stack([v[5], v[7]])
    l7 = np.stack([v[7], v[6]])
    l8 = np.stack([v[6], v[4]])

    # Connecting vertical lines
    l9 = np.stack([v[0], v[4]])
    l10 = np.stack([v[1], v[5]])
    l11 = np.stack([v[2], v[6]])
    l12 = np.stack([v[3], v[7]])

    # Stack the lines together
    lines = np.array([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])

    return lines


def overlay_image(
        background: np.ndarray,
        overlay: np.ndarray,
        x_offset: Optional[int] = None,
        y_offset: Optional[int] = None
) -> np.ndarray:
    """
    Add an overlay to an image.
    Adapted from https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
    """
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = overlay.shape

    assert bg_channels == 3, f'Background image should have exactly 3 channels (RGB). found:{bg_channels}'

    # Convert white to transparent if no alpha channel defined
    if fg_channels == 3:
        white = np.all(overlay == [255, 255, 255], axis=-1)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        overlay[white, -1] = 0

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    overlay = overlay[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = overlay[:, :, :3]
    alpha_channel = overlay[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    combined = background.copy()
    combined[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return combined
