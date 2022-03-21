import matplotlib.pyplot as plt
import numpy as np

from wormlab3d.data.model import Midline3D
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_component_comparison
from wormlab3d.toolkit.util import parse_target_arguments

show_plots = True

X_orig = np.array([[0.1048975, 0., 0.],
                   [0.10545421, 0.00734377, 0.00268555],
                   [0.10601091, 0.01468778, 0.00537109],
                   [0.10656881, 0.02203238, 0.00830078],
                   [0.10712862, 0.02937829, 0.01098633],
                   [0.10769081, 0.03672659, 0.01391602],
                   [0.10825706, 0.04407859, 0.01660156],
                   [0.10882807, 0.05143631, 0.01928711],
                   [0.10940719, 0.05880165, 0.02197266],
                   [0.10999942, 0.06617618, 0.0246582],
                   [0.11060381, 0.0735631, 0.02734375],
                   [0.1112206, 0.08096755, 0.0300293],
                   [0.11185098, 0.08839428, 0.0324707],
                   [0.1124959, 0.09584737, 0.03491211],
                   [0.11315584, 0.10332823, 0.03735352],
                   [0.11383009, 0.11083567, 0.03955078],
                   [0.11451602, 0.11836672, 0.04174805],
                   [0.11520648, 0.12591732, 0.04394531],
                   [0.11588955, 0.13348353, 0.04589844],
                   [0.11654902, 0.14106262, 0.0480957],
                   [0.11716366, 0.14865267, 0.05004883],
                   [0.11771393, 0.15625083, 0.05200195],
                   [0.11818242, 0.16385365, 0.05395508],
                   [0.11855364, 0.17145705, 0.05615234],
                   [0.11881208, 0.17905581, 0.05810547],
                   [0.11894321, 0.1866442, 0.06030273],
                   [0.1189332, 0.19421577, 0.06225586],
                   [0.11876845, 0.20176315, 0.06469727],
                   [0.11843657, 0.20927715, 0.06689453],
                   [0.11792636, 0.21674657, 0.06933594],
                   [0.11722922, 0.22415924, 0.07202148],
                   [0.11633945, 0.23150074, 0.07470703],
                   [0.11524701, 0.23875487, 0.07739258],
                   [0.11394095, 0.24590266, 0.08056641],
                   [0.11241007, 0.2529223, 0.08374023],
                   [0.11064529, 0.25978887, 0.0871582],
                   [0.10864162, 0.26647675, 0.09082031],
                   [0.10640025, 0.27295947, 0.09472656],
                   [0.10392523, 0.27920842, 0.09887695],
                   [0.10122538, 0.28519785, 0.10302734],
                   [0.09831786, 0.29090607, 0.10766602],
                   [0.09522557, 0.29631722, 0.11254883],
                   [0.09197426, 0.30141914, 0.11743164],
                   [0.08858943, 0.30620289, 0.12280273],
                   [0.08509994, 0.3106643, 0.12817383],
                   [0.08153486, 0.31480432, 0.1340332],
                   [0.07792354, 0.31862915, 0.13989258],
                   [0.07429314, 0.32214987, 0.14575195],
                   [0.07067037, 0.32537985, 0.15209961],
                   [0.06708097, 0.32833552, 0.15844727],
                   [0.06354618, 0.33103895, 0.16479492],
                   [0.0600872, 0.33351469, 0.17163086],
                   [0.05672646, 0.33578777, 0.17822266],
                   [0.05348802, 0.33788574, 0.18505859],
                   [0.05039167, 0.33983648, 0.19213867],
                   [0.04745841, 0.34167218, 0.19921875],
                   [0.04470587, 0.34342706, 0.20629883],
                   [0.04214382, 0.34513366, 0.21362305],
                   [0.03977728, 0.34682155, 0.22094727],
                   [0.03760886, 0.34851408, 0.22827148],
                   [0.03563428, 0.35023463, 0.2355957],
                   [0.03384471, 0.35200536, 0.24316406],
                   [0.03222919, 0.35384643, 0.25073242],
                   [0.03076935, 0.35577869, 0.25805664],
                   [0.02944469, 0.35782111, 0.265625],
                   [0.02823162, 0.35999048, 0.27319336],
                   [0.02710891, 0.36230481, 0.28051758],
                   [0.02605939, 0.3647809, 0.2878418],
                   [0.02506471, 0.36743677, 0.29541016],
                   [0.02410603, 0.37029123, 0.30249023],
                   [0.02316546, 0.37336338, 0.30981445],
                   [0.02222991, 0.37667298, 0.31689453],
                   [0.02129126, 0.38024092, 0.32373047],
                   [0.02034974, 0.38409662, 0.33056641],
                   [0.01940632, 0.38827252, 0.3371582],
                   [0.01846409, 0.39280188, 0.34350586],
                   [0.01752996, 0.39771879, 0.34960938],
                   [0.01661181, 0.40304995, 0.35546875],
                   [0.0157218, 0.40881145, 0.3605957],
                   [0.01487541, 0.4150064, 0.36547852],
                   [0.01409483, 0.42162251, 0.36962891],
                   [0.01340747, 0.42862785, 0.37329102],
                   [0.01284504, 0.43596435, 0.37597656],
                   [0.01243496, 0.44355476, 0.37792969],
                   [0.01219797, 0.45131552, 0.37939453],
                   [0.01214457, 0.459167, 0.37988281],
                   [0.01227522, 0.46703947, 0.37988281],
                   [0.0125823, 0.47487617, 0.37915039],
                   [0.01305079, 0.48263192, 0.37792969],
                   [0.01365972, 0.49027109, 0.37597656],
                   [0.01438427, 0.49776542, 0.3737793],
                   [0.01519465, 0.50509322, 0.37109375],
                   [0.01605844, 0.51223743, 0.36791992],
                   [0.01694655, 0.51918435, 0.36425781],
                   [0.01783299, 0.5259248, 0.36035156],
                   [0.01869488, 0.53245068, 0.35595703],
                   [0.01951456, 0.53875434, 0.35131836],
                   [0.02027845, 0.54482877, 0.34643555],
                   [0.02097464, 0.55066979, 0.34106445],
                   [0.02159381, 0.55627739, 0.33569336],
                   [0.02212501, 0.56165802, 0.32983398],
                   [0.02255893, 0.56682289, 0.32397461],
                   [0.02288604, 0.57178795, 0.31787109],
                   [0.02309656, 0.57658064, 0.31152344],
                   [0.02318335, 0.58123362, 0.30517578],
                   [0.02314019, 0.58578503, 0.29882812],
                   [0.02296376, 0.59027565, 0.29223633],
                   [0.02265453, 0.59474814, 0.28588867],
                   [0.02222061, 0.59923613, 0.27954102],
                   [0.0216713, 0.60376894, 0.27294922],
                   [0.0210197, 0.60837519, 0.26660156],
                   [0.02027941, 0.61308086, 0.26049805],
                   [0.01945949, 0.61790955, 0.25415039],
                   [0.0185647, 0.62287867, 0.24829102],
                   [0.01759744, 0.62799585, 0.2421875],
                   [0.01655841, 0.63326204, 0.23657227],
                   [0.01544738, 0.63867295, 0.23095703],
                   [0.01426411, 0.64421952, 0.2253418],
                   [0.01301098, 0.64988983, 0.22021484],
                   [0.01169133, 0.65566814, 0.21484375],
                   [0.0103128, 0.66153872, 0.20996094],
                   [0.00888777, 0.66748583, 0.20483398],
                   [0.0074327, 0.67349446, 0.19995117],
                   [0.00595832, 0.67955101, 0.1953125],
                   [0.00447392, 0.68564212, 0.19042969],
                   [0.00298429, 0.69175446, 0.18579102],
                   [0.0014925, 0.697878, 0.1809082],
                   [0., 0.70400512, 0.17626953]])


def get_midline() -> np.ndarray:
    """
    Find a midline3d by id.
    """
    args = parse_target_arguments()
    if args.midline3d is not None:
        mid = Midline3D.objects.get(id=args.midline3d)
        X = mid.X
    else:
        X = X_orig
    return X


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

    # fig4 = plot_natural_frame_3d(NF1)
    # fig4.suptitle(f'NF1: Length = {NF1.length:.2f}')
    # fig5 = plot_natural_frame_3d(NF2)
    # fig5.suptitle(f'NF2: Length = {NF2.length:.2f}')
    # fig6 = plot_natural_frame_3d(NF3)
    # fig6.suptitle(f'NF3: Length = {NF3.length:.2f}')

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
    X = get_midline()
    NF1 = NaturalFrame(X)

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


def scale_checks():
    mc = np.ones(100, dtype=np.complex128) * (1 + 0j) * 2 * np.pi

    NF1 = NaturalFrame(mc, length=1)
    NF2 = NaturalFrame(mc / 2, length=2)
    NF3 = NaturalFrame(mc * 2, length=0.5)

    fig1 = plot_natural_frame_3d(NF1, show_frame_arrows=False, show_pca_arrows=False)
    fig1.suptitle(f'NF1: Length = {NF1.length:.2f}')
    fig2 = plot_natural_frame_3d(NF2, show_frame_arrows=False, show_pca_arrows=False)
    fig2.suptitle(f'NF1: Length = {NF2.length:.2f}')
    fig2 = plot_natural_frame_3d(NF3, show_frame_arrows=False, show_pca_arrows=False)
    fig2.suptitle(f'NF3: Length = {NF3.length:.2f}')

    fig1 = plot_component_comparison(NF1, NF2)
    fig1.suptitle('NF1 vs NF2')
    fig2 = plot_component_comparison(NF1, NF3)
    fig2.suptitle('NF1 vs NF3')
    plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    main()
    # circle_checks()
    # scale_checks()
