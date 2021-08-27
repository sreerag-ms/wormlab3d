import os
import time

import matplotlib.pyplot as plt
from wormlab3d import LOGS_PATH
from wormlab3d.data.model import Midline3D
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_components, plot_natural_frame_3d
from wormlab3d.toolkit.util import parse_target_arguments

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})

save_plots = False
TIMESTAMP = time.strftime('%Y%m%d_%H%M')


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
    NF = NaturalFrame(midline.X)

    plot_natural_frame_3d(NF)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_midline={midline.id}_3D.svg'
        plt.savefig(fn)
    plt.show()

    plot_natural_frame_components(NF)
    if save_plots:
        fn = LOGS_PATH + '/' + TIMESTAMP + f'_midline={midline.id}_components.svg'
        plt.savefig(fn)
    plt.show()


if __name__ == '__main__':
    main()
