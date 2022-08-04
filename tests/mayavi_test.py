import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mayavi import mlab
from tvtk.tools import visual

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d.postures.natural_frame import NaturalFrame

n_points = 100


def example():
    dphi, dtheta = np.pi / 250.0, np.pi / 250.0
    [phi, theta] = np.mgrid[0:np.pi + dphi * 1.5:dphi, 0:2 * np.pi + dtheta * 1.5:dtheta]
    m0 = 4
    m1 = 3
    m2 = 2
    m3 = 3
    m4 = 6
    m5 = 2
    m6 = 6
    m7 = 4
    r = np.sin(m0 * phi)**m1 + np.cos(m2 * phi)**m3 + np.sin(m4 * theta)**m5 + np.cos(m6 * theta)**m7
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.cos(phi)
    z = r * np.sin(phi) * np.sin(theta)
    mlab.mesh(x, y, z)
    mlab.show()


def plot_sphere_slice_border(r: float, h: float, c: str, alpha: float = 1.):
    angle = np.arccos(h / r)
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(angle, np.pi - angle, n_points)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(n_points), np.cos(v))
    mlab.mesh(x, y, z, color=to_rgb(c), opacity=alpha)


def plot_sphere_slice_caps(r: float, h: float, c: str, alpha: float = 1.):
    angle = np.arccos(h / r)
    radius = r * np.sin(angle)
    r2 = np.linspace(0, radius, n_points)
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.outer(r2, np.sin(theta))
    y = np.outer(r2, np.cos(theta))
    z = h * np.ones((n_points, n_points))
    mlab.mesh(x, y, z, color=to_rgb(c), opacity=alpha)
    mlab.mesh(x, y, -z, color=to_rgb(c), opacity=alpha)


def plot_sphere_slice_capped(r: float, h: float, c: str, alpha: float = 1.):
    plot_sphere_slice_border(r, h, c, alpha)
    plot_sphere_slice_caps(r, h, c, alpha)


def sphere_slice_border():
    plot_sphere_slice_border(1, 0.3, 'blue')
    mlab.show()


def sphere_slice_capped():
    r = 1
    h = 0.2
    plot_sphere_slice_capped(r, h, 'red', 1)
    mlab.show()


def multiple_sphere_slices():
    alpha = 0.9
    plot_sphere_slice_capped(1, 0.7, 'red', alpha)
    plot_sphere_slice_capped(1.2, 0.4, 'blue', alpha)
    plot_sphere_slice_capped(1.8, 0.2, 'green', alpha)
    mlab.show()


def arrow_from_a_to_b(x1, y1, z1, x2, y2, z2):
    ar1 = visual.arrow(x=x1, y=y1, z=z1)
    ar1.length_cone = 0.4

    arrow_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos / arrow_length
    ar1.axis = [x2 - x1, y2 - y1, z2 - z1]
    return ar1


def sphere_slice_with_axes():
    r = 1
    h = 0.2
    plot_sphere_slice_capped(r, h, 'red', 0.5)
    visual.set_viewer(mlab.gcf())
    arrow_from_a_to_b(0, 0, 0, 1, 0, 0)
    arrow_from_a_to_b(0, 0, 0, 0, 1, 0)
    arrow_from_a_to_b(0, 0, 0, 0, 0, 1)

    mlab.show()


def tapered_cylinder():
    """
    Tapered cylinder surface tests.
    """
    length = 1
    N = 128
    u = np.linspace(0, length, N)

    # Midline as straight line
    X = np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=-1)

    # Midline as a planar sine-wave
    X = 5 * np.sin(np.linspace(0, 2 * np.pi, N)) + 0.j

    # Midline as a spiral
    X = np.zeros((N, 3))
    X[:, 0] = np.sin(2 * np.pi * u) / 10
    X[:, 1] = np.cos(2 * np.pi * u) / 10
    X[:, 2] = np.linspace(1 / np.sqrt(3), 0, N)

    NF = NaturalFrame(X)

    surface, K_surf = NF.surface(
        N_theta=32,
        radius=0.03,
        taper=0.2,
        shape_k1=1.5,
        shape_k2=1
    )

    # Add midline
    x, y, z = X.T
    t = np.linspace(0, 1, N)
    pts = mlab.plot3d(x, y, z, t, opacity=1, tube_radius=None, line_width=5)
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
    pts.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Plot surface
    x, y, z = surface[..., 0], surface[..., 1], surface[..., 2]
    msh = mlab.mesh(x, y, z, scalars=K_surf, opacity=0.8, colormap='coolwarm')
    msh.scene.renderer.use_depth_peeling = True  # This is needed for opacity to work properly
    msh.scene.renderer.maximum_number_of_peels = 8
    mlab.show()


def grid():
    """
    Axis properties.
    """
    x, y = np.mgrid[-3:3:50j, -3:3:50j]
    z = 3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2) \
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
        - 1. / 3 * np.exp(-(x + 1)**2 - y**2)

    mlab.figure(bgcolor=(1, 1, 1))  # Make background white.
    surf = mlab.surf(x, y, z, colormap='RdYlBu', warp_scale=0.3, representation='wireframe', line_width=0.5)
    # mlab.pipeline.user_defined(surf, filter=tvtk.CubeAxesActor())
    mlab.outline(color=(0, 0, 0))
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
    axes.title_text_property.color = (0.0, 0.0, 0.0)
    axes.title_text_property.font_family = 'times'
    axes.label_text_property.color = (0.0, 0.0, 0.0)
    axes.label_text_property.font_family = 'times'
    # mlab.savefig("vector_plot_in_3d.pdf")
    mlab.gcf().scene.parallel_projection = True  # Source: <<https://stackoverflow.com/a/32531283/2729627>>.
    mlab.orientation_axes()  # Source: <<https://stackoverflow.com/a/26036154/2729627>>.
    mlab.show()


def antialiasing(aa: bool = True):
    """
    Tests to see if the various anti-aliasing options found online do anything.
    Doesn't look like they do much if anything.
    """
    mlab.options.offscreen = True
    fig = mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(1024, 768))
    fig.scene.parallel_projection = True

    if aa:
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 16
        fig.scene.anti_aliasing_frames = 16
        # fig.scene.render_window.open_gl_init()

    fig.scene.show_axes = True

    axColor = (0.45, 0.45, 0.45)
    centerAxis = mlab.points3d(0.0, 0.0, 0.0, 4, mode='axes', color=axColor,
                               line_width=1.0, scale_factor=1., opacity=1.0)
    centerAxis.actor.property.lighting = False

    axes = mlab.axes(centerAxis, color=axColor, nb_labels=9)
    axes.property.display_location = 'background'
    axes.title_text_property.opacity = 0
    axes.label_text_property.bold = 0
    axax = axes.axes
    axax.label_format = '%-#6.2g'
    axax.fly_mode = 'none'
    axax.font_factor = 1.0

    def quiv3dlabels(vectors=(0, 0, 0, 1, 1, 1), mode='arrow', scale_factor=1,
                     colormap='gist_rainbow', scale_mode='vector', resolution=12,
                     labels=None):
        x, y, z, u, v, w = vectors
        numVectors = len(x)
        quivs = mlab.quiver3d(x, y, z, u, v, w, mode=mode, scale_factor=scale_factor,
                              scale_mode=scale_mode, colormap=colormap,
                              resolution=resolution)
        quivs.glyph.color_mode = 'color_by_vector'
        quivsProperties = quivs.glyph.glyph_source.glyph_source
        quivsProperties.shaft_radius = 0.006
        quivsProperties.tip_length = 0.08
        quivsProperties.tip_radius = 0.02

        retLabels = list()
        indexes = range(numVectors)
        if labels and len(labels) == numVectors:
            pass
        else:
            labels = ['v' + str(ind) for ind in indexes]
        for ind, label in zip(indexes, labels):
            x, y, z, u, v, w = vectors[0][ind], vectors[1][ind], vectors[2][ind], \
                               vectors[3][ind], vectors[4][ind], vectors[5][ind]
            text = mlab.text(x=x + u / 2., y=y + v / 2., z=z + w / 2.,
                             text=str(label), width=0.01 * len(label))
            retLabels.append(text)
        return quivs, retLabels

    v1 = (0, 0, 0, 4, 4, 4)
    v2 = (1, 0, 1, -2, -4, 5)
    v3 = (-1, 2, -4, 5, 0, -1)
    v4 = (0, 0, 0, -3, -3, -4)
    v5 = (-3, -3, -4, 4, 5, -8)
    vectors = np.stack([v1, v2, v3, v4, v5], axis=1)

    quivs, _ = quiv3dlabels(vectors)

    fig.scene._lift()
    img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
    mlab.clf(fig)
    mlab.close()
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img)
    ax.set_title(f'aa={"ON" if aa else "OFF"} img.sum()={img.sum()}')
    ax.axis('off')
    fig.tight_layout()
    plt.show()


def animate_sphere():
    """
    Test to animate a sphere to different radii.
    Ideally the outline and axes should follow automatically, but they don't.
    """
    from mayavi import mlab

    phi, theta = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]

    def make_sphere(r):
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        return x, y, z

    # mlab.figure()
    x, y, z = make_sphere(1)
    mesh = mlab.mesh(x, y, z)
    outline = mlab.outline(mesh)
    axes = mlab.axes(color=(0, 0, 0))

    def update(t):
        x, y, z = make_sphere(np.abs(np.random.randn(1)))
        mesh.mlab_source.reset(x=x, y=y, z=z)

        mesh.update_data()
        mesh.update_pipeline()
        mesh.actor.update_data()
        mesh.actor.update_pipeline()
        mesh.mlab_source.update()

    @mlab.animate(delay=100)
    def animate():
        for t in range(1, 1000):
            update(t)
            yield

    animate()
    mlab.show()


if __name__ == '__main__':
    example()
    sphere_slice_border()
    sphere_slice_capped()
    multiple_sphere_slices()
    sphere_slice_with_axes()
    tapered_cylinder()
    grid()
    antialiasing(False)
    antialiasing(True)
    animate_sphere()
