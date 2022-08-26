import itertools
from typing import Union, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from mayavi.core.scene import Scene
from tvtk.tools import visual
from tvtk.tools.visual import Arrow

from simple_worm.controls import ControlsNumpy
from simple_worm.frame import FrameNumpy, FRAME_COMPONENT_KEYS
from simple_worm.plot3d import FrameArtist, Arrow3D, MIDLINE_CMAP_DEFAULT
from wormlab3d.postures.natural_frame import NaturalFrame, normalise
from wormlab3d.toolkit.plot_utils import to_rgb

SURFACE_CMAP_DEFAULT = 'coolwarm'


def _plot_arrow(
        origin: np.ndarray,
        vec: np.ndarray,
        fig: Scene = None,
        **arrow_args
) -> Arrow:
    """
    Render an arrow from the given origin pointing along vec.
    Possible arrow_args: radius_shaft, radius_cone, length_cone, opacity.
    """
    x1, y1, z1 = origin
    x2, y2, z2 = vec
    arrow_length = np.linalg.norm(vec)
    arrow = visual.arrow(x=x1, y=y1, z=z1, viewer=fig, **arrow_args)
    arrow.actor.scale = [arrow_length, arrow_length, arrow_length]
    if 'opacity' in arrow_args:
        arrow.actor.property.opacity = arrow_args['opacity']
    arrow.pos = arrow.pos / arrow_length
    arrow.axis = [x2, y2, z2]
    return arrow


class FrameArtistMLab:
    """
    Draw midlines and frame component vectors using mayavi.
    """

    midline_opt_defaults = {
        'opacity': 0.9,
        'tube_radius': None,
        'line_width': 4,
    }
    surface_opt_defaults = {
        'N_theta': 32,
        'radius': 0.03,
        'taper': 0.2,
        'shape_k1': 1.5,
        'shape_k2': 1
    }
    mesh_opt_defaults = {
        'opacity': 0.8,
    }
    outline_opt_defaults = {
        'color': to_rgb('darkgrey'),
        'tube_radius': 0.001
    }
    arrow_opt_defaults = {
        'opacity': 0.9,
        'radius_shaft': 0.01,
        'radius_cone': 0.05,
        'length_cone': 0.1
    }
    arrow_colour_defaults = {
        'e0': 'red',
        'e1': 'blue',
        'e2': 'green',
    }

    def __init__(
            self,
            NF: NaturalFrame,
            midline_opts: Dict = None,
            surface_opts: Dict = None,
            mesh_opts: Dict = None,
            outline_opts: Dict = None,
            arrow_opts: Dict = None,
            arrow_colours: Dict = None,
            arrow_scale: float = 0.1,
            n_arrows: int = 0,
            alpha_max: float = None,
            beta_max: float = None,
            use_centred_midline: bool = True
    ):
        self.NF = NF
        self.use_centred_midline = use_centred_midline or np.iscomplexobj(NF.X)
        if self.use_centred_midline:
            self.X = NF.X_pos
        else:
            self.X = NF.X
        self.N = self.NF.N
        self.arrows = {}
        self.path = None
        self.surface = None
        self.outline = None
        if midline_opts is None:
            midline_opts = {}
        self.midline_opts = {**FrameArtistMLab.midline_opt_defaults, **midline_opts}
        if surface_opts is None:
            surface_opts = {}
        if 'use_centred' not in surface_opts:
            surface_opts['use_centred'] = self.use_centred_midline
        self.surface_opts = {**FrameArtistMLab.surface_opt_defaults, **surface_opts}
        if mesh_opts is None:
            mesh_opts = {}
        self.mesh_opts = {**FrameArtistMLab.mesh_opt_defaults, **mesh_opts}
        if outline_opts is None:
            outline_opts = {}
        self.outline_opts = {**FrameArtistMLab.outline_opt_defaults, **outline_opts}
        if arrow_opts is None:
            arrow_opts = {}
        self.arrow_opts = {**FrameArtistMLab.arrow_opt_defaults, **arrow_opts}
        if arrow_colours is None:
            arrow_colours = {}
        self.arrow_colours = {**FrameArtistMLab.arrow_colour_defaults, **arrow_colours}
        self.arrow_scale = arrow_scale

        # Show a subset of the arrows
        if 0 < n_arrows < 1:
            idxs = np.round(np.linspace(0, self.N - 1, np.round(n_arrows * self.N).astype(int))).astype(int)
        elif 0 < n_arrows < self.N:
            idxs = np.round(np.linspace(0, self.N - 1, n_arrows)).astype(int)
        else:
            idxs = range(self.N)
        self.arrow_idxs = idxs

        self.worm_length = self.NF.length

        # Controls colour-maps
        self.cmaps = {
            'alpha': cm.get_cmap('OrRd'),
            'beta': cm.get_cmap('BuPu'),
        }

        # Controls fixed maximums
        self.max_vals = {
            'alpha': alpha_max,
            'beta': beta_max
        }

    def add_midline(self, fig: Scene, cmap_name: str = MIDLINE_CMAP_DEFAULT):
        """
        Add the midline.
        """
        if cmap_name is None:
            cmap_name = MIDLINE_CMAP_DEFAULT
        cmap = cm.get_cmap(cmap_name)
        cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
        x, y, z = self.X.T
        t = np.linspace(0, 1, self.N)
        self.path = mlab.plot3d(x, y, z, t, figure=fig, **self.midline_opts)
        self.path.module_manager.scalar_lut_manager.lut.table = cmaplist

    def add_surface(self, fig: Scene, cmap_name: str = SURFACE_CMAP_DEFAULT, v_min: float = None, v_max: float = None):
        """
        Add the initial midline.
        """
        if cmap_name is None:
            cmap_name = SURFACE_CMAP_DEFAULT
        cmap = cm.get_cmap(cmap_name)
        cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
        surface, K_surf = self.NF.surface(**self.surface_opts)
        x, y, z = surface[..., 0], surface[..., 1], surface[..., 2]
        self.surface = mlab.mesh(x, y, z, scalars=K_surf, figure=fig, vmin=v_min, vmax=v_max, **self.mesh_opts)
        self.surface.scene.renderer.use_depth_peeling = True
        self.surface.scene.renderer.maximum_number_of_peels = 16
        self.surface.module_manager.scalar_lut_manager.lut.table = cmaplist

    def add_outline(
            self,
            fig: Scene = None,
    ):
        """
        Add outline box containing the surface.
        """
        lines = self._get_outline_points()
        outline = []
        for l in lines:
            line_obj = mlab.plot3d(
                *l.T,
                figure=fig,
                **self.outline_opts
            )
            outline.append(line_obj)
        self.outline = outline

    def _get_outline_points(self):
        """
        Calculate the outline points.
        """

        # Get the bounds for the mesh points relative to the PCA components
        mesh_pts = np.stack([
            self.surface.mlab_source.x.flatten(),
            self.surface.mlab_source.y.flatten(),
            self.surface.mlab_source.z.flatten()
        ], axis=-1)
        R = np.stack(self.NF.pca.components_, axis=1)
        Xt = np.einsum('ij,bj->bi', R.T, mesh_pts)

        # Calculate box vertices
        height, width, depth = np.ptp(Xt, axis=0)
        M = np.array(list(itertools.product(*[[-1, 1]] * 3)))
        dims = np.array([[height, width, depth]])
        v = self.X.mean(axis=0) + (M * dims / 2) @ R.T

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

    def add_component_vectors(
            self,
            fig: Scene = None,
            draw_e0: bool = True,
            draw_e1: bool = True,
            draw_e2: bool = True,
            C: ControlsNumpy = None,
    ):
        """
        Add the initial component/force vectors.
        """
        arrows = {}
        keys = FRAME_COMPONENT_KEYS.copy()
        if not draw_e0:
            keys.remove('e0')
        if not draw_e1:
            keys.remove('e1')
        if not draw_e2:
            keys.remove('e2')
        for k in keys:
            arrows[k] = []
            vec, colours = self._get_vectors_and_colours(k, C)
            for i in self.arrow_idxs:
                arrow = _plot_arrow(
                    fig=fig,
                    origin=self.X[i],
                    vec=vec[i],
                    color=colours[i],
                    **self.arrow_opts
                )
                arrows[k].append(arrow)
        self.arrows = arrows

    def update(self, NF: NaturalFrame, C: ControlsNumpy = None):
        """
        Update the midline and the component vectors.
        """
        self.NF = NF
        if self.use_centred_midline:
            self.X = NF.X_pos
        else:
            self.X = NF.X

        # Update midline
        if self.path is not None:
            x, y, z = self.X.T
            self.path.mlab_source.reset(x=x, y=y, z=z)

        # Update surface
        if self.surface is not None:
            surface, K_surf = self.NF.surface(**self.surface_opts)
            x, y, z = surface[..., 0], surface[..., 1], surface[..., 2]
            self.surface.mlab_source.reset(x=x, y=y, z=z, scalars=K_surf)

        # Update outline
        if self.outline is not None:
            new_lines = self._get_outline_points()
            for i, l in enumerate(self.outline):
                l.mlab_source.points = new_lines[i]

        # todo: Update component vectors
        if len(self.arrows) is not None:
            for k in self.arrows:
                vec, colours = self._get_vectors_and_colours(k, C)
                for i, j in enumerate(self.arrow_idxs):
                    self.arrows[k][i].actor.trait_set(scale=[max_z, max_z, max_z])
                    self.arrows[k][i].actor.trait_set(color=colours[j])
                    # self.arrows[k][i].set_verts(origin=X[:, j], vec=vec[:, j])
                    # self.arrows[k][i].set_color(colours[j])

    def _get_vectors_and_colours(self, k: str, C: ControlsNumpy = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the component vectors for the frame.
        These are scaled and coloured according to the (gated) controls if passed.
        """
        vectors = getattr(self.NF, k) * self.worm_length * self.arrow_scale
        if C is not None and k != 'e0':
            fk = 'alpha' if k == 'e1' else 'beta'
            force = getattr(C, 'get_' + fk)()
            max_val = self.max_vals[fk]
            if max_val is None:
                max_val = np.abs(force).max()
            vectors *= force
            colours = self.cmaps[fk](np.abs(force) / max_val)
        else:
            colours = [to_rgb(self.arrow_colours[k]) for _ in range(self.N)]

        return vectors, colours


def plot_natural_frame_3d(
        NF: NaturalFrame,
        azim: float = -60.,
        elev: float = 30.,
        midline_opts: dict = None,
        show_frame_arrows: bool = True,
        n_frame_arrows: int = 30,
        arrow_opts: dict = None,
        arrow_scale: float = 0.1,
        show_pca_arrows: bool = True,
        show_pca_arrow_labels: bool = True,
        midline_cmap: str = None,
        ax: Axes = None,
        zoom: float = 1.,
        use_centred_midline: bool = True
) -> Union[Figure, Axes]:
    """
    Make a 3D plot of a midline with optional frame vectors and pca arrows.
    Uses matplotlib.
    """
    # Set up frame
    if use_centred_midline or np.iscomplexobj(NF.X):
        F = FrameNumpy(x=NF.X_pos.T)
    else:
        F = FrameNumpy(x=NF.X.T)
    F.e1 = NF.M1.T
    F.e2 = NF.M2.T

    # Create figure if required
    return_ax = False
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', azim=azim, elev=elev)
    else:
        fig = ax.get_figure()
        return_ax = True
    # cla(ax)

    # Add frame arrows and midline
    if arrow_opts is None:
        arrow_opts = {}
    arrow_opts = {**{'alpha': 0.3}, **arrow_opts}
    fa = FrameArtist(
        F,
        midline_opts=midline_opts,
        n_arrows=n_frame_arrows,
        arrow_opts=arrow_opts,
        arrow_scale=arrow_scale
    )
    if show_frame_arrows:
        fa.add_component_vectors(ax)
    fa.add_midline(ax, cmap_name=midline_cmap)

    # Add PCA component vectors
    if show_pca_arrows:
        centre = NF.X_pos.mean(axis=0)
        for i in range(2, -1, -1):
            vec = NF.pca.components_[i] * NF.pca.explained_variance_ratio_[i] * NF.length / 3
            if vec.sum() == 0:
                continue
            origin = centre - vec / 2
            arrow = Arrow3D(
                origin=origin,
                vec=vec,
                color=fa.arrow_colours[f'e{i}'],
                mutation_scale=20,
                arrowstyle='->',
                linewidth=3,
                alpha=0.9
            )
            ax.add_artist(arrow)

            if show_pca_arrow_labels:
                origin = origin - normalise(vec) * NF.length * 0.15
                ax.text(
                    origin[0],
                    origin[1],
                    origin[2],
                    f'$v_{i}$',
                    color=fa.arrow_colours[f'e{i}'],
                    fontsize=40,
                    zorder=10,
                    horizontalalignment='center',
                    verticalalignment='center',
                )

    # Fix axes range
    mins, maxs = F.get_bounding_box(zoom=zoom)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # ax.axis('off')

    if return_ax:
        return ax

    fig.tight_layout()

    return fig


def plot_natural_frame_3d_mlab(
        NF: NaturalFrame,
        azimuth: float = -60.,
        elevation: float = 30.,
        roll: float = 0.,
        distance: float = 1.,
        midline_opts: dict = None,
        surface_opts: dict = None,
        mesh_opts: dict = None,
        show_frame_arrows: bool = True,
        show_outline: bool = True,
        show_axis: bool = True,
        n_frame_arrows: int = 30,
        arrow_opts: dict = None,
        arrow_scale: float = 0.1,
        show_pca_arrows: bool = True,
        show_pca_arrow_labels: bool = True,
        midline_cmap: str = None,
        surface_cmap: str = None,
        use_centred_midline: bool = True,
        offscreen: bool = True
) -> Scene:
    """
    Make a 3D plot of a midline with optional frame vectors and pca arrows.
    Uses mayavi.
    """

    # Set up mlab figure
    mlab.options.offscreen = offscreen
    fig = mlab.figure(size=(2000, 2000), bgcolor=(1, 1, 1))
    if 1:
        # Doesn't really seem to make any difference
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Set up the artist and add the pieces
    fa = FrameArtistMLab(
        NF,
        midline_opts=midline_opts,
        surface_opts=surface_opts,
        mesh_opts=mesh_opts,
        n_arrows=n_frame_arrows,
        arrow_opts=arrow_opts,
        arrow_scale=arrow_scale,
        use_centred_midline=use_centred_midline
    )
    centre = fa.X.mean(axis=0)
    if show_frame_arrows:
        fa.add_component_vectors(fig, draw_e0=False)
    fa.add_midline(fig, cmap_name=midline_cmap)
    fa.add_surface(fig, cmap_name=surface_cmap)
    if show_outline:
        fa.add_outline(fig)

    if show_axis:
        axes = mlab.axes(color=(0, 0, 0), nb_labels=5, xlabel='', ylabel='', zlabel='')
        axes.axes.label_format = ''

    # Add PCA component vectors
    if show_pca_arrows:
        for i in range(2, -1, -1):
            vec = NF.pca.components_[i] * NF.pca.singular_values_[i] / 3
            if vec.sum() == 0:
                continue
            origin = centre - vec / 2
            scale = np.linalg.norm(vec)
            _plot_arrow(
                origin=origin,
                vec=vec,
                fig=fig,
                color=to_rgb(fa.arrow_colours[f'e{i}']),
                opacity=0.8,
                radius_shaft=0.01 / scale,
                radius_cone=0.02 / scale,
                # length_cone=0.02 / scale
            )
            if show_pca_arrow_labels:
                origin = origin - vec / np.linalg.norm(vec) * 0.1
                mlab.text3d(origin[0], origin[1], origin[2], f'v{i}',
                            color=to_rgb(fa.arrow_colours[f'e{i}']), scale=0.05)

    mlab.view(
        azimuth=azimuth,
        elevation=elevation,
        roll=roll,
        distance=distance,
        focalpoint=centre
    )

    # # Useful for getting the view parameters when recording from the gui:
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [-0.22882408164660478, 0.3015073944478576, -1.0862516659511565]
    # scene.scene.camera.focal_point = [0.10072373567797888, 0.15345832376500054, 0.3325747923468798]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.4541512399773104, -0.8908364976462779, 0.01252939297748773]
    # scene.scene.camera.clipping_range = [0.7203740696263817, 2.3976973567212854]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    return fig


def plot_natural_frame_components(NF: NaturalFrame) -> Figure:
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2)
    gs2 = GridSpec(2, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    N = len(NF.m1)
    cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('$m_1$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.m1[i:i + 2], c=fc[i])

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.set_title('$m_2$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.m2[i:i + 2], c=fc[i])

    ax = fig.add_subplot(gs[0, 1], sharex=ax)
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.kappa[i:i + 2], c=fc[i])
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])

    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    for i in range(N - 1):
        ax.plot(NF.psi[i:i + 2], ind[i:i + 2], c=fc[i])
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    fig.tight_layout()

    return fig


def plot_component_comparison(NF1: NaturalFrame, NF2: NaturalFrame) -> Figure:
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2)
    gs2 = GridSpec(2, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    NF1_args = {'c': 'red', 'linestyle': '--', 'alpha': 0.8}
    NF2_args = {'c': 'blue', 'linestyle': ':', 'alpha': 0.8}

    N = len(NF1.m1)
    ind = np.arange(N)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('$m_1$')
    ax.plot(ind, NF1.m1, **NF1_args)
    ax.plot(ind, NF2.m1, **NF2_args)

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.set_title('$m_2$')
    ax.plot(ind, NF1.m2, **NF1_args)
    ax.plot(ind, NF2.m2, **NF2_args)

    ax = fig.add_subplot(gs[0, 1], sharex=ax)
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    ax.plot(ind, NF1.kappa, **NF1_args)
    ax.plot(ind, NF2.kappa, **NF2_args)
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])

    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    ax.plot(NF1.psi, ind, **NF1_args)
    ax.plot(NF2.psi, ind, **NF2_args)
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    fig.tight_layout()

    return fig
