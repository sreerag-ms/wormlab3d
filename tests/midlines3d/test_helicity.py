import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.postures.helicities import illustrate_helicity_method
from wormlab3d.postures.natural_frame import NaturalFrame, EPS
from wormlab3d.postures.plot_utils import plot_natural_frame_3d
from wormlab3d.toolkit.util import normalise

show_plots = False
save_plots = False
img_extension = 'png'


def _make_helix(radius: float, pitch: float, length: float, N: int) -> np.ndarray:
    """
    Helper method to make a helix of given radius, pitch and length.
    """
    if radius == 0:
        t = np.linspace(0, length, N)
        return np.stack([np.zeros_like(t), np.zeros_like(t), t]).T
    b = pitch / (2 * np.pi)
    s = (radius**2 + b**2)**(1 / 2)
    t = np.linspace(0, length / (s + EPS), N)
    return np.stack([
        radius * np.cos(t),
        radius * np.sin(t),
        b * t
    ]).T


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

    assert np.allclose(h, 0)
    assert np.allclose(h2, 0, atol=1e-1)


def test_twisted_lines():
    """
    Test that a straight line with a full twist gives h=2pi
    -- only works with angles method
    """
    N = 100
    x = np.linspace(0, 1, 100)
    X = np.stack([x, np.zeros_like(x), np.zeros_like(x)]).T
    NF = NaturalFrame(X)
    psi = np.linspace(0, -2 * np.pi, N)
    M1 = np.stack([np.zeros_like(x), np.sin(psi), np.cos(psi)], axis=1)
    M2 = np.stack([np.zeros_like(x), np.cos(psi), np.sin(psi)], axis=1)
    NF.M1 = M1
    NF.M2 = M2
    if show_plots:
        plot_natural_frame_3d(NF)
    h = NF.helicity_angles_method()
    assert np.allclose(h, 2 * np.pi)

    # Twist the other way
    psi = np.linspace(0, 2 * np.pi, N)
    NF.M1 = np.stack([np.zeros_like(x), np.sin(psi), np.cos(psi)], axis=1)
    NF.M2 = np.stack([np.zeros_like(x), np.cos(psi), np.sin(psi)], axis=1)
    if show_plots:
        plot_natural_frame_3d(NF)
    h = NF.helicity_angles_method()
    assert np.allclose(h, -2 * np.pi)

    # Generate random rotation matrix
    A = Rotation.from_rotvec(normalise(np.random.random(3)) * np.random.rand() * np.pi).as_matrix()
    t0 = np.random.rand() * 2 * np.pi
    t1 = np.random.rand() * 2 * np.pi
    t2 = np.random.rand() * 2 * np.pi
    R = Rotation.from_rotvec(A[:, 0] * t0) * Rotation.from_rotvec(A[:, 1] * t1) * Rotation.from_rotvec(A[:, 2] * t2)
    R = R.as_matrix()

    # Rotate the line and M1/M2 and also reverse the order
    X = np.einsum('ij,bj->bi', R, X[::-1])
    NF = NaturalFrame(X)
    NF.M1 = np.einsum('ij,bj->bi', R, M1[::-1])
    NF.M2 = np.einsum('ij,bj->bi', R, M2[::-1])
    if show_plots:
        plot_natural_frame_3d(NF)
    h = NF.helicity_angles_method()
    assert np.allclose(h, 2 * np.pi)


def test_perfect_helix():
    """
    Construct a perfect helix: A curve with a single coil and single twist.
    """
    N = 100
    psi = np.linspace(0, 2 * np.pi, 100)
    mc = 2 * np.pi * np.ones(N, dtype=np.complex128) * np.exp(1j * psi)
    NF = NaturalFrame(mc)
    h = NF.helicity()

    if show_plots:
        illustrate_helicity_method(NF)
        plt.show()

    assert np.allclose(h, 2.53, rtol=0.001)


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

    assert np.allclose(h, h2, rtol=1e-2)


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

    assert np.allclose(h, -h2, atol=1e-4)


def test_helices_by_pitch_and_radius():
    """
    Tests helices of varying pitch and radius.
    """
    length = 1
    N = 100
    n_rs = 20
    n_ps = 40
    rs = np.linspace(0, length / (2 * np.pi), n_rs)
    ps = np.linspace(-length, length, n_ps)
    out = np.zeros((n_rs, n_ps))

    for i, r in enumerate(rs):
        for j, p in enumerate(ps):
            X = _make_helix(r, p, length, N)
            nf = NaturalFrame(X)
            out[i, j] = nf.helicity()

    # Plot results in 2D
    fig, axes = plt.subplots(2)
    cmap = cm.get_cmap('coolwarm')
    fc = cmap((np.arange(n_ps) + 0.5) / n_ps)
    ax = axes[0]
    for i in range(n_ps):
        ax.plot(rs, out[:, i], c=fc[i], label=f'{ps[i]:.2f}')
    ax.set_xlabel('r')
    ax.set_ylabel('h')
    ax = axes[1]
    fc = cmap((np.arange(n_rs) + 0.5) / n_rs)
    for i in range(n_rs):
        ax.plot(ps, out[i, :], c=fc[i], label=f'{rs[i]:.2f}')
    ax.set_xlabel('p')
    ax.set_ylabel('h')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_pr_sweep_res_2d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()

    # Plot 3D surface of helicity results
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X, Y = np.meshgrid(rs, ps)
    ax.plot_surface(X, Y, out.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('r')
    ax.set_ylabel('p')
    ax.set_zlabel('h')

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_pr_sweep_vols_3d.{img_extension}'
        plt.savefig(path)

    if show_plots:
        plt.show()


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
    n_rs = 20
    n_ks = 20
    rs = np.linspace(-10 * np.pi, 10 * np.pi, n_rs)  # radius
    ks = np.linspace(0, 10, n_ks)  # twists

    out = np.zeros((n_rs, n_ks))

    for i, r in enumerate(rs):
        for j, k in enumerate(ks):
            psi = np.linspace(0, k * 2 * np.pi, N)
            mc = r * np.ones(N, dtype=np.complex128) * np.exp(1j * psi)
            nf = NaturalFrame(mc)
            out[i, j] = nf.helicity()

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
    test_twisted_lines()
    test_perfect_helix()
    test_reversed_helices()
    test_inverted_helices()
    # test_helices_by_pitch_and_radius()
    # illustrate_method_test(outlier=False)
    # illustrate_method_test(outlier=True)
    # plot_sweep_over_radius_twists()
