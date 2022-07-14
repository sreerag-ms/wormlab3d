import matplotlib.pyplot as plt
import numpy as np

from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d

show_plots = False


def test_straight_line():
    """
    Check that chirality is zero for a straight line.
    """
    x = np.linspace(0, 1, 100)
    X = np.stack([x, np.zeros_like(x), np.zeros_like(x)]).T

    # Load the points into a natural frame
    NF = NaturalFrame(X)
    if show_plots:
        plot_natural_frame_3d(NF)
        plt.show()

    c = NF.chirality()

    assert c == 0


def test_planar_bends():
    """
    Check that chirality is zero for planar curves.
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

    c = NF.chirality()
    c2 = NF2.chirality()

    assert abs(c) < 1e-2
    assert abs(c2) < 1e-4


def test_reversed_helices():
    """
    Construct a helix and check that the chirality measure returns same when the order of points reversed.
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

    c = NF.chirality()
    c2 = NF2.chirality()

    assert np.allclose(c, c2)


def test_inverted_helices():
    """
    Construct helices and check that the chirality measure returns negative when the chirality is reversed.
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

    c = NF.chirality()
    c2 = NF2.chirality()

    assert np.allclose(c, -c2)


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

    c = NF.chirality()
    c2 = NF2.chirality()

    assert np.sign(c) == np.sign(c2)
    assert c < c2


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    test_straight_line()
    test_planar_bends()
    test_reversed_helices()
    test_inverted_helices()
    test_helices_with_different_radii()
