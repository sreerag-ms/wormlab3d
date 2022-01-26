import matplotlib.pyplot as plt
import numpy as np

from wormlab3d.data.model import Midline3D
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_component_comparison
from wormlab3d.toolkit.util import parse_target_arguments

show_plots = True


def get_midline() -> Midline3D:
    """
    Find a midline3d by id.
    """
    args = parse_target_arguments()
    if args.midline3d is None:
        raise RuntimeError('This script must be run with the --midline3d=ID argument defined.')
    return Midline3D.objects.get(id=args.midline3d)


def make_plots(
        NF1: NaturalFrame,
        NF2: NaturalFrame,
        NF3: NaturalFrame,
):
    fig1 = plot_component_comparison(NF1, NF2)
    fig1.suptitle('NF1 vs NF2')
    fig2 = plot_component_comparison(NF2, NF3)
    fig2.suptitle('NF2 vs NF3')
    fig3 = plot_component_comparison(NF1, NF3)
    fig3.suptitle('NF1 vs NF3')

    fig4 = plot_natural_frame_3d(NF1)
    fig4.suptitle(f'NF1: Length = {NF1.length:.2f}')
    fig5 = plot_natural_frame_3d(NF2)
    fig5.suptitle(f'NF2: Length = {NF2.length:.2f}')
    fig6 = plot_natural_frame_3d(NF3)
    fig6.suptitle(f'NF3: Length = {NF3.length:.2f}')

    fig7, axes = plt.subplots(3)
    for i in range(3):
        ax = axes[i]
        ax.plot(NF1.X_pos[:, i])
        ax.plot(NF2.X_pos[:, i])
    fig7.suptitle('NF1 vs NF2')

    fig8, axes = plt.subplots(3)
    for i in range(3):
        ax = axes[i]
        ax.plot(NF2.X_pos[:, i])
        ax.plot(NF3.X_pos[:, i])
    fig8.suptitle('NF2 vs NF3')

    plt.show()


def main():
    midline = get_midline()
    NF1 = NaturalFrame(midline.X)

    # Reproduce NF from complex representation
    NF2 = NaturalFrame(NF1.mc, length=NF1.length, X0=NF1.X_pos[0], T0=NF1.T[0], M0=NF1.M1[0])

    # Turn it back into the midline again
    NF3 = NaturalFrame(NF2.X_pos)

    # Plot
    if show_plots:
        make_plots(NF1, NF2, NF3)

    # Check the values are close enough
    for k in ['X_pos', 'T', 'M1', 'M2', 'mc']:
        for a, b in ([NF1, NF2], [NF1, NF3], [NF2, NF3]):
            assert np.allclose(getattr(a, k), getattr(b, k), rtol=0.1, atol=0.2)


def circle_checks():
    mc = np.ones(100, dtype=np.complex128) * (1 + 0j)
    NF1 = NaturalFrame(mc, length=0.1)
    NF2 = NaturalFrame(mc, length=1)
    NF3 = NaturalFrame(mc, length=10)
    NF1a = NaturalFrame(NF1.X_pos)
    NF2a = NaturalFrame(NF2.X_pos)
    NF3a = NaturalFrame(NF3.X_pos)

    fig1 = plot_natural_frame_3d(NF1, show_frame_arrows=False, show_pca_arrows=False)
    fig1.suptitle(f'NF1: Length = {NF1.length:.2f}')
    fig2 = plot_natural_frame_3d(NF2, show_frame_arrows=False, show_pca_arrows=False)
    fig2.suptitle(f'NF1: Length = {NF2.length:.2f}')
    fig3 = plot_natural_frame_3d(NF3, show_frame_arrows=False, show_pca_arrows=False)
    fig3.suptitle(f'NF1: Length = {NF3.length:.3f}')

    fig1 = plot_component_comparison(NF1, NF1a)
    fig1.suptitle('NF1 vs NF1a')
    fig2 = plot_component_comparison(NF2, NF2a)
    fig2.suptitle('NF2 vs NF2a')
    fig3 = plot_component_comparison(NF3, NF3a)
    fig3.suptitle('NF3 vs NF3a')
    plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    main()
    circle_checks()
