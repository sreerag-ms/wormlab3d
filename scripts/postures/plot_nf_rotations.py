import os
import time

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH
from wormlab3d.data.model import Midline3D
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_component_comparison
from wormlab3d.postures.rotations import rotate
from wormlab3d.toolkit.util import parse_target_arguments

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})

save_plots = False
TIMESTAMP = time.strftime('%Y%m%d_%H%M')
np.random.seed(6)


def get_midline() -> Midline3D:
    """
    Find a midline3d by id.
    """
    args = parse_target_arguments()
    if args.midline3d is None:
        raise RuntimeError('This script must be run with the --midline3d=ID argument defined.')
    return Midline3D.objects.get(id=args.midline3d)


def main():
    midline = get_midline()
    NF1 = NaturalFrame(midline.X)

    # 3D frame plot original
    plot_natural_frame_3d(NF1)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_midline={midline.id}_3D_original.svg'
        plt.savefig(fn)
    plt.show()

    # Rotate and translate the midline
    abg = np.random.uniform(low=-np.pi, high=np.pi, size=3)
    T = np.random.uniform(low=-100, high=100, size=3)
    X2 = rotate(midline.X, abg[0], abg[1], abg[2]) + T
    NF2 = NaturalFrame(X2)

    # 3D frame plot rotated and translated
    plot_natural_frame_3d(NF2)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_midline={midline.id}_3D_rotated.svg'
        plt.savefig(fn)
    plt.show()

    # Component comparison plot
    plot_component_comparison(NF1, NF2)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_midline={midline.id}_components_comparison.svg'
        plt.savefig(fn)
    plt.show()


if __name__ == '__main__':
    main()
