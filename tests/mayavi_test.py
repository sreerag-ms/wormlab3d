import matplotlib.colors as mcolors
import numpy as np
from mayavi import mlab
from tvtk.tools import visual

n_points = 100


def main():
    dphi, dtheta = np.pi / 250.0, np.pi / 250.0
    [phi, theta] = np.mgrid[0:np.pi + dphi * 1.5:dphi, 0:2 * np.pi + dtheta * 1.5:dtheta]
    m0 = 4;
    m1 = 3;
    m2 = 2;
    m3 = 3;
    m4 = 6;
    m5 = 2;
    m6 = 6;
    m7 = 4;
    r = np.sin(m0 * phi)**m1 + np.cos(m2 * phi)**m3 + np.sin(m4 * theta)**m5 + np.cos(m6 * theta)**m7
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.cos(phi)
    z = r * np.sin(phi) * np.sin(theta)
    mlab.mesh(x, y, z)
    mlab.show()


def _get_rgb(c: str):
    return mcolors.to_rgb(c)


def plot_sphere_slice_border(r: float, h: float, c: str, alpha: float = 1.):
    angle = np.arccos(h / r)
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(angle, np.pi - angle, n_points)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(n_points), np.cos(v))
    mlab.mesh(x, y, z, color=_get_rgb(c), opacity=alpha)


def plot_sphere_slice_caps(r: float, h: float, c: str, alpha: float = 1.):
    angle = np.arccos(h / r)
    radius = r * np.sin(angle)
    r2 = np.linspace(0, radius, n_points)
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.outer(r2, np.sin(theta))
    y = np.outer(r2, np.cos(theta))
    z = h * np.ones((n_points, n_points))
    mlab.mesh(x, y, z, color=_get_rgb(c), opacity=alpha)
    mlab.mesh(x, y, -z, color=_get_rgb(c), opacity=alpha)


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


if __name__ == '__main__':
    main()
    sphere_slice_border()
    sphere_slice_capped()
    multiple_sphere_slices()
    sphere_slice_with_axes()
