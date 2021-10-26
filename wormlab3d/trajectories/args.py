from argparse import ArgumentParser

from simple_worm.material_parameters import MP_DEFAULT_K
from wormlab3d.data.model.midline3d import M3D_SOURCES, M3D_SOURCE_RECONST
from wormlab3d.toolkit.util import str2bool
from wormlab3d.trajectories.displacement import DISPLACEMENT_AGGREGATION_OPTIONS, DISPLACEMENT_AGGREGATION_SQUARED_SUM


def get_args(
        include_trajectory_options: bool = True,
        include_msd_options: bool = True,
        include_K_options: bool = True,
        include_planarity_options: bool = True,
):
    """
    Parse command line arguments for the trajectory scripts.
    Not all arguments are used for all scripts, but this saves duplication.
    """
    parser = ArgumentParser(description='Wormlab3D trajectory script.')

    # Source arguments, trial id is always required
    parser.add_argument('--trial', type=int, help='Trial id.', required=True)
    parser.add_argument('--midline3d-source', type=str, default=M3D_SOURCE_RECONST, choices=M3D_SOURCES,
                        help='Midline3D source.')
    parser.add_argument('--midline3d-source-file', type=str, help='Midline3D source file.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)

    # Trajectory options
    if include_trajectory_options:
        parser.add_argument('--trajectory-point', type=float, default=None,
                            help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass. '
                                 'Leave empty (default) to return full trajectory.')
        parser.add_argument('--projection', type=str, choices=['3D', 'x', 'y', 'z', 'xy', 'yz', 'xz'], default='3D',
                            help='Use a projection of the midline, or not (default=3D).')

    # MSD arguments
    if include_msd_options:
        parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                            help='Time lag sizes.')
        parser.add_argument('--min-delta', type=int, default=1, help='Minimum time lag.')
        parser.add_argument('--max-delta', type=int, default=10000, help='Maximum time lag.')
        parser.add_argument('--delta-step', type=int, default=1, help='Step between deltas.')
        parser.add_argument('--aggregation', type=str, choices=DISPLACEMENT_AGGREGATION_OPTIONS,
                            default=DISPLACEMENT_AGGREGATION_SQUARED_SUM,
                            help='Displacements can be taken as L2 norms or as the squared sum of components.')

    # K estimation arguments
    if include_K_options:
        parser.add_argument('--K0', type=float, default=MP_DEFAULT_K, help='Initial value of K for the optimiser.')
        parser.add_argument('--smoothing-window', type=int, default=5, help='Smooth trajectory with moving average.')
        parser.add_argument('--K-sample-frames', type=int, default=5,
                            help='Number of frames from which to calculate the K estimate.')

    # Planarity arguments
    if include_planarity_options:
        parser.add_argument('--planarity-window', type=int, default=5,
                            help='Number of frames to use when calculating the planarity measure.')

    args = parser.parse_args()

    # Check trajectory point
    if include_trajectory_options and args.trajectory_point is not None:
        assert args.trajectory_point == -1 or 0 <= args.trajectory_point <= 1, 'trajectory-point must be -1 for centre of mass or between 0 and 1.'

    return args
