import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.postures.natural_frame import NaturalFrame, normalise
from wormlab3d.postures.plot_utils import plot_natural_frame_3d

show_plots = True
save_plots = True
img_extension = 'png'


def test_straight_line():
    """
    Check that helicity is zero for a straight line.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, np.zeros_like(x), np.zeros_like(x)]).T

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    if show_plots:
        plot_natural_frame_3d(NF)
        plt.show()

    h = NF.helicity()

    assert h == 0


def test_planar_bends():
    """
    Check that helicity is zero for planar curves.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, 0.8 * np.sin(np.pi * x), np.zeros_like(x)]).T
    X2 = np.stack([x, np.zeros_like(x), 0.1 * np.sin(12 * x)]).T

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    NF2 = NaturalFrame(X2)
    if show_plots:
        plot_natural_frame_3d(NF)
        plot_natural_frame_3d(NF2)
        plt.show()

    h = NF.helicity()
    h2 = NF2.helicity()

    assert abs(h) < 1e-2
    assert abs(h2) < 1e-4


def test_reversed_helices():
    """
    Construct a helix and check that the helicity measure returns same when the order of points reversed.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, 0.1 * np.sin(12 * x), 0.1 * np.cos(12 * x)]).T
    X2 = X[::-1]

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    NF2 = NaturalFrame(X2)
    if show_plots:
        plot_natural_frame_3d(NF)
        plot_natural_frame_3d(NF2)
        plt.show()

    h = NF.helicity()
    h2 = NF2.helicity()

    assert np.allclose(h, h2)


def test_inverted_helices():
    """
    Construct helices and check that the helicity measure returns negative when the chirality is reversed.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, 0.1 * np.sin(12 * x), 0.1 * np.cos(12 * x)]).T
    X2 = np.stack([x, 0.1 * np.cos(12 * x), 0.1 * np.sin(12 * x)]).T

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    NF2 = NaturalFrame(X2)
    if show_plots:
        plot_natural_frame_3d(NF)
        plot_natural_frame_3d(NF2)
        plt.show()

    h = NF.helicity()
    h2 = NF2.helicity()

    assert np.allclose(h, -h2)


def test_helices_with_different_radii():
    """
    Construct helices with different radii and check that the chirality measure is smaller for the smaller radius.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, 0.1 * np.sin(12 * x), 0.1 * np.cos(12 * x)]).T
    X2 = np.stack([x, 0.2 * np.sin(12 * x), 0.2 * np.cos(12 * x)]).T

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    NF2 = NaturalFrame(X2)
    if show_plots:
        plot_natural_frame_3d(NF)
        plot_natural_frame_3d(NF2)
        plt.show()

    c = NF.helicity()
    c2 = NF2.helicity()

    assert np.sign(c) == np.sign(c2)
    assert c < c2


def illustrate_method(outlier: bool):
    """
    Demonstrate method.
    """
    if not show_plots and not save_plots:
        return
    N = 100
    s = np.linspace(0, 1, N)

    cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 6)

    x = np.linspace(0, 1, 100)
    X = np.stack([x, 0.1 * np.sin(12 * x), 0.1 * np.cos(12 * x)]).T
    if outlier:
        X[-1] = [1, 0.3, 0.3]

    # Load the helix into a natural frame
    NF = NaturalFrame(X)
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax.set_title('Original curve')
    plot_natural_frame_3d(NF, ax=ax)

    # Try to align v1 with the direction of the curve
    u = normalise(NF.X_pos[-1] - NF.X_pos[0])
    v1, v2, v3 = NF.pca.components_
    if np.dot(u, v1) < 0:
        v1 *= -1

    # Rotate points to align with the principal components.
    R = np.stack([v1, v2, v3], axis=1)
    Xt = np.einsum('ij,bj->bi', R.T, NF.X_pos)

    # Plot the projection of the points onto the 2nd two components
    ax = fig.add_subplot(gs[0:2, 2:4])
    ax.set_title('Projection onto v1/v2 plane')
    ax.scatter(x=Xt[:, 1], y=Xt[:, 2], c=fc)

    # Use the difference vectors between adjacent points and ignore first coordinate.
    diff = (Xt[1:] - Xt[:-1])[:, 1:]

    # Plot the difference vectors
    ax = fig.add_subplot(gs[0:2, 4:6])
    ax.set_title('Differences between adjacent projected points')
    ax.scatter(x=diff[:, 0], y=diff[:, 1], c=fc[:-1], alpha=0.7)
    for i, dv in enumerate(diff):
        ax.plot(np.linspace(0, dv[0], 100), np.linspace(0, dv[1], 100), color=fc[i], alpha=0.7)

    # Convert into polar coordinates
    r = np.linalg.norm(diff, axis=-1)
    theta = np.unwrap(np.arctan2(*diff.T))

    # Plot the rs and thetas
    ax = fig.add_subplot(gs[2, 0:3])
    ax.set_title('Polar coordinate: $\\theta$')
    ax.scatter(x=s[:-1], y=theta, c=fc[:-1])
    ax = fig.add_subplot(gs[2, 3:6])
    ax.set_title('Polar coordinate: r')
    ax.scatter(x=s[:-1], y=r, c=fc[:-1])

    # Weight the angular changes by the radii and sum to give helicity measure.
    r = (r[1:] + r[:-1]) / 2

    # Correct for reflections
    if np.allclose(np.linalg.det(R), -1):
        r *= -1

    dtheta = (theta[1:] - theta[:-1])
    h = np.sum(r * dtheta)

    # Plot the change in theta
    ax = fig.add_subplot(gs[3, 0:3])
    ax.set_title('Change in $\\theta$: $\delta\\theta$')
    ax.scatter(x=s[:-2], y=dtheta, c=fc[:-2])

    # Plot the cumulative sum of helicity contributions
    ax = fig.add_subplot(gs[3, 3:6])
    ax.set_title(f'Cumulative sum of $h=\sum_s r \cdot \delta\\theta={h:.2f}$')
    ax.scatter(x=s[:-2], y=np.cumsum(r * dtheta), c=fc[:-2])
    ax.scatter(x=s[-3], y=[h, ], color='red', s=150, marker='x')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_helices_method'
                            + ('_outlier' if outlier else '')
                            + f'.{img_extension}')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive, MIDLINE_CMAP_DEFAULT
    # interactive()
    test_straight_line()
    test_planar_bends()
    test_reversed_helices()
    test_inverted_helices()
    test_helices_with_different_radii()
    illustrate_method(outlier=False)
    illustrate_method(outlier=True)
