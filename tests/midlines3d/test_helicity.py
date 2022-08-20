import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.postures.helicities import illustrate_helicity_method
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d

show_plots = True
save_plots = False
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

    assert abs(h) < 1e-6
    assert abs(h2) < 1e-1


def test_perfect_helix():
    """
    Construct a perfect helix: A curve with constant curvature and torsion.
    """
    N = 100
    psi = np.linspace(0, 2 * np.pi, 100)
    mc = 2 * np.pi * np.ones(N, dtype=np.complex128) * np.exp(1j * psi)
    NF = NaturalFrame(mc)
    h = NF.helicity()

    if show_plots:
        illustrate_helicity_method(NF)
        plt.show()

    assert np.allclose(h, 1, atol=1e-8)


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

    assert np.allclose(h, h2, atol=1e-1)


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
    Construct helices with different radii and check that the helicity measure is smaller for the smaller radius.
    """
    x = np.linspace(0, 1, 100)
    X1 = np.stack([x, 0.2 * np.sin(12 * x), 0.2 * np.cos(12 * x)]).T
    X2 = np.stack([x, 0.1 * np.sin(12 * x), 0.1 * np.cos(12 * x)]).T
    X3 = np.stack([x, 0.05 * np.sin(12 * x), 0.05 * np.cos(12 * x)]).T

    # Load the points into a natural frame
    NF1 = NaturalFrame(X1)
    NF2 = NaturalFrame(X2)
    NF3 = NaturalFrame(X3)
    if show_plots:
        plot_natural_frame_3d(NF1)
        plot_natural_frame_3d(NF2)
        plot_natural_frame_3d(NF3)
        plt.show()

    h1 = NF1.helicity()
    h2 = NF2.helicity()
    h3 = NF3.helicity()

    # illustrate_helicity_method(NF1)
    # illustrate_helicity_method(NF2)
    # illustrate_helicity_method(NF3)
    # plt.show()

    assert np.sign(h1) == np.sign(h2) == np.sign(h3)
    assert np.abs(h1) > np.abs(h2) > np.abs(h3)


def illustrate_method_test(outlier: bool):
    """
    Illustrate the method.
    """
    if not show_plots and not save_plots:
        return
    N = 100
    x = np.linspace(0, 1, N)
    X = np.stack([x, 1 * np.sin(2 * np.pi * x), 1 * np.cos(2 * np.pi * x)]).T
    if outlier:
        X[-1] = [1, 0.3, 0.3]

    # Load the helix into a natural frame
    NF = NaturalFrame(X)
    illustrate_helicity_method(NF)
    # illustrate_helicity_method2(NF)

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_helices_method'
                            + ('_outlier' if outlier else '')
                            + f'.{img_extension}')
        plt.savefig(path)

    if show_plots:
        plt.show()


def plot_sweep_over_radius_twists():
    """
    Vary the helix radius and and plot how the helicity changes
    """
    if not show_plots and not save_plots:
        return
    N = 100
    x = np.linspace(0, 1, N)
    n_rs = 10
    n_ks = 10
    rs = np.linspace(0, 2 * np.pi, n_rs)  # radius
    ks = np.linspace(0, 2, n_ks)  # twists

    out = np.zeros((n_rs, n_ks))

    for i, r in enumerate(rs):
        for j, k in enumerate(ks):
            X = np.stack([x, r * np.sin(2 * np.pi * k * x), r * np.cos(2 * np.pi * k * x)]).T
            nf = NaturalFrame(X)
            h = nf.helicity()
            out[i, j] = h

    # Plot results in 2D
    fig, axes = plt.subplots(2)
    cmap = cm.get_cmap('coolwarm')
    fc = cmap((np.arange(n_ks) + 0.5) / n_ks)
    ax = axes[0]
    for i in range(n_ks):
        ax.plot(out[:, i], c=fc[i], label=f'{ks[i]:.2f}')
    ax.set_xlabel('rs')
    ax = axes[1]
    for i in range(n_rs):
        ax.plot(out[i, :], c=fc[i], label=f'{rs[i]:.2f}')
    ax.set_xlabel('ks')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_sweep_res_2d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Plot 3D surface of results
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X, Y = np.meshgrid(rs, ks)
    ax.plot_surface(X, Y, out, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('k')
    ax.set_zlabel('h')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_sweep_res_3d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()


def plot_sweep_over_radius_twists2():
    """
    Vary the helix radius and and plot how the helicity changes
    """
    if not show_plots and not save_plots:
        return
    N = 100

    # x = np.linspace(0, 1, N)
    n_rs = 20
    n_ks = 20
    rs = np.linspace(0, 100 * np.pi, n_rs)  # radius
    ks = np.linspace(0, 100, n_ks)  # twists

    out = np.zeros((n_rs, n_ks))

    for i, r in enumerate(rs):
        for j, k in enumerate(ks):
            psi = np.linspace(0, k * 2 * np.pi, 100)
            mc = r * np.ones(N, dtype=np.complex128) * np.exp(1j * psi)
            nf = NaturalFrame(mc)

            # X = np.stack([x, r * np.sin(2 * np.pi * k * x), r * np.cos(2 * np.pi * k * x)]).T
            # nf = NaturalFrame(X)
            h = nf.helicity()
            out[i, j] = h

    # Plot results in 2D
    fig, axes = plt.subplots(2)
    cmap = cm.get_cmap('coolwarm')
    fc = cmap((np.arange(n_ks) + 0.5) / n_ks)
    ax = axes[0]
    for i in range(n_ks):
        ax.plot(out[:, i], c=fc[i], label=f'{ks[i]:.2f}')
    ax.set_xlabel('rs')
    ax = axes[1]
    for i in range(n_rs):
        ax.plot(out[i, :], c=fc[i], label=f'{rs[i]:.2f}')
    ax.set_xlabel('ks')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_sweep_res_2d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Plot 3D surface of results
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X, Y = np.meshgrid(rs, ks)
    ax.plot_surface(X, Y, out, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('k')
    ax.set_zlabel('h')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_sweep_res_3d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    test_straight_line()
    test_planar_bends()
    test_perfect_helix()
    test_reversed_helices()
    test_inverted_helices()
    test_helices_with_different_radii()
    illustrate_method_test(outlier=False)
    illustrate_method_test(outlier=True)
    plot_sweep_over_radius_twists()
    plot_sweep_over_radius_twists2()
