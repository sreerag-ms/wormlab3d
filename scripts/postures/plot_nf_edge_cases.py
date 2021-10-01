import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_natural_frame_components, \
    plot_component_comparison

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})

save_plots = True
TIMESTAMP = time.strftime('%Y%m%d_%H%M')
np.random.seed(6)


def plot_donuts():
    i = np.linspace(0, 2 * np.pi, 100)
    X1 = np.array([np.sin(i), np.cos(i), np.zeros_like(i)]).T - [100, 100, 100]
    NF1 = NaturalFrame(X1)
    X2 = np.array([np.sin(i), np.cos(i), np.zeros_like(i)]).T + [100, 100, 100]
    NF2 = NaturalFrame(X2)

    plot_natural_frame_3d(NF1, azim=45, elev=15)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_donut1_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_components(NF1)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_donut1_components.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_3d(NF2, azim=45, elev=15)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_donut2_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_components(NF2)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_donut2_components.svg'
        plt.savefig(fn)
    plt.show()


def plot_helices():
    a = 0.5
    b = 0.6
    i = np.linspace(0, 2 * np.pi, 100)
    X1 = np.array([a * np.sin(1.5 * i), b * np.cos(1.5 * i) + 1, i / 2]).T
    NF1 = NaturalFrame(X1)
    X2 = np.array([b * np.sin(1.5 * i), a * np.cos(1.5 * i) + 1, i / 2]).T
    NF2 = NaturalFrame(X2)

    plot_natural_frame_3d(NF1, azim=-130, elev=-10, arrow_scale=0.05)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_helix1_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_3d(NF2, azim=-130, elev=-10, arrow_scale=0.05)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_helix2_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_component_comparison(NF1, NF2)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_helix_component_comparison.svg'
        plt.savefig(fn)
    plt.show()


def plot_balls():
    N = 200
    a = 0.5
    b = 0.6
    rng = default_rng()
    i = np.linspace(0, 2 * np.pi, N)
    j = np.concatenate([np.linspace(0, 2 * np.pi, int(N / 2)), np.linspace(2 * np.pi, 0, int(N / 2))])
    ball = np.array([j * a * np.sin(5 * i), j * b * np.cos(5 * i), i / np.sqrt(2)]).T

    X1 = ball + rng.normal(scale=0.01, size=ball.shape)
    NF1 = NaturalFrame(X1)
    X2 = ball + rng.normal(scale=0.01, size=ball.shape)
    NF2 = NaturalFrame(X2)

    plot_natural_frame_3d(NF1, azim=45, elev=-10, arrow_scale=0.)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_balls1_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_3d(NF2, azim=45, elev=-10, arrow_scale=0.)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_balls2_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_component_comparison(NF1, NF2)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_balls_component_comparison.svg'
        plt.savefig(fn)
    plt.show()


def main():
    """
    For s 0 ≈ s 1 > s 2 the shape resembles a planar circle (donut).
    For s 0 > s 1 ≈ s 2 the shape may resemble a helix.
    If s 0 ≈ s 1 ≈ s 2 the point-cloud will resemble a ball or a sphere, and the worm may be coiled up tightly.
    """
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # interactive()
    plot_donuts()
    plot_helices()
    plot_balls()
    exit()


if __name__ == '__main__':
    main()
