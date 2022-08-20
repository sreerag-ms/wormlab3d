from argparse import ArgumentParser, Namespace

from simple_worm.material_parameters import MP_DEFAULT_K
from wormlab3d.data.model.midline3d import M3D_SOURCES, M3D_SOURCE_MF
from wormlab3d.particles.particle_explorer import DIST_TYPES
from wormlab3d.toolkit.util import str2bool
from wormlab3d.trajectories.displacement import DISPLACEMENT_AGGREGATION_OPTIONS, DISPLACEMENT_AGGREGATION_SQUARED_SUM


def get_args(
        include_trajectory_options: bool = True,
        include_msd_options: bool = True,
        include_K_options: bool = True,
        include_planarity_options: bool = True,
        include_helicity_options: bool = True,
        include_manoeuvre_options: bool = True,
        include_pe_options: bool = True,
        validate_source: bool = True
) -> Namespace:
    """
    Parse command line arguments for the trajectory scripts.
    Not all arguments are used for all scripts, but this saves duplication.
    """
    parser = ArgumentParser(description='Wormlab3D trajectory script.')

    # Source arguments
    parser.add_argument('--dataset', type=str, help='Dataset id.', required=False)
    parser.add_argument('--reconstruction', type=str, help='Reconstruction id.', required=False)
    parser.add_argument('--trial', type=int, help='Trial id.', required=False)
    parser.add_argument('--trials', type=lambda s: [int(item) for item in s.split(',')], help='Trial ids.',
                        required=False)
    parser.add_argument('--midline3d-source', type=str, default=M3D_SOURCE_MF, choices=M3D_SOURCES,
                        help='Midline3D source.')
    parser.add_argument('--midline3d-source-file', type=str, help='Midline3D source file.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--tracking-only', type=str2bool, help='Use the tracking trajectory only.', default=False)

    # Trajectory options
    if include_trajectory_options:
        parser.add_argument('--smoothing-window', type=int,
                            help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')
        parser.add_argument('--directionality', type=str, choices=['forwards', 'backwards'],
                            help='Use only forwards/backwards frames. Default=both.')
        parser.add_argument('--prune-slowest-ratio', type=float,
                            help='Prune the slowest x% frames from the trajectory, stitching together afterwards.')
        parser.add_argument('--trajectory-point', type=float, default=None,
                            help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass. '
                                 'Leave empty (default) to return full trajectory.')
        parser.add_argument('--projection', type=str, choices=['3D', 'x', 'y', 'z', 'xy', 'yz', 'xz'], default='3D',
                            help='Use a projection of the midline, or not (default=3D).')
        parser.add_argument('--smoothing-window-curvature', type=int,
                            help='Smooth the curvature using average in a sliding window. Size defined in number of frames.')
        parser.add_argument('--eigenworms', type=str,
                            help='Eigenworms by id.')

    # MSD arguments
    if include_msd_options:
        parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                            help='Time lag sizes.')
        parser.add_argument('--min-delta', type=int, default=1, help='Minimum time lag.')
        parser.add_argument('--max-delta', type=int, default=10000, help='Maximum time lag.')
        parser.add_argument('--delta-step', type=float, default=1, help='Step between deltas. -ve=exponential steps.')
        parser.add_argument('--aggregation', type=str, choices=DISPLACEMENT_AGGREGATION_OPTIONS,
                            default=DISPLACEMENT_AGGREGATION_SQUARED_SUM,
                            help='Displacements can be taken as L2 norms or as the squared sum of components.')

    # K estimation arguments
    if include_K_options:
        parser.add_argument('--K0', type=float, default=MP_DEFAULT_K, help='Initial value of K for the optimiser.')
        parser.add_argument('--K-sample-frames', type=int, default=5,
                            help='Number of frames from which to calculate the K estimate.')

    # Planarity arguments
    if include_planarity_options:
        parser.add_argument('--planarity-window', type=int, default=5,
                            help='Number of frames to use when calculating the planarity measure.')

    # Helicity arguments
    if include_helicity_options:
        parser.add_argument('--helicity-window', type=int, default=50,
                            help='Number of frames to use when calculating the helicity measure.')

    # Manoeuvre arguments
    if include_manoeuvre_options:
        parser.add_argument('--min-forward-frames', type=int, default=25,
                            help='Minimum number of forward frames before counting a forward locomotion section.')
        parser.add_argument('--min-reversal-frames', type=int, default=25,
                            help='Minimum number of reversal frames to use to identify a manoeuvre.')
        parser.add_argument('--manoeuvre-window', type=int, default=500,
                            help='Number of frames to include either side of a detected manoeuvre.')

    # Particle explorer arguments
    if include_pe_options:
        parser.add_argument('--batch-size', type=int, help='Batch size.')
        parser.add_argument('--rate-01', type=float, help='Transition rate from slow speed to fast speed.')
        parser.add_argument('--rate-10', type=float, help='Transition rate from fast speed to slow speed.')
        parser.add_argument('--rate-02', type=float, help='Transition rate from slow speed to turn.')
        parser.add_argument('--rate-20', type=float, help='Transition rate from turn to slow speed.')
        parser.add_argument('--speeds-0-mu', type=float, help='Slow speed average.')
        parser.add_argument('--speeds-0-sig', type=float, help='Slow speed standard deviation.')
        parser.add_argument('--speeds-1-mu', type=float, help='Fast speed average.')
        parser.add_argument('--speeds-1-sig', type=float, help='Fast speed standard deviation.')
        parser.add_argument('--theta-dist-type', type=str, choices=DIST_TYPES, help='Planar angle distribution type.')
        parser.add_argument('--theta-dist-params', type=lambda s: [float(item) for item in s.split(',')],
                            help='Planar angle distribution parameters.')
        parser.add_argument('--phi-dist-type', type=str, choices=DIST_TYPES, help='Non-planar angle distribution type.')
        parser.add_argument('--phi-dist-params', type=lambda s: [float(item) for item in s.split(',')],
                            help='Non-planar angle distribution parameters.')
        parser.add_argument('--nonp-pause-type', type=str, choices=[None, 'linear', 'quadratic'],
                            help='Non-planar turn pause penalty type.')
        parser.add_argument('--nonp-pause-max', type=float, help='Maximum non-planar turn pause penalty.')
        parser.add_argument('--sim-duration', type=float, help='Simulation time.')
        parser.add_argument('--sim-dt', type=float, help='Simulation timestep.')

    args = parser.parse_args()

    # Check trajectory point
    if include_trajectory_options and args.trajectory_point is not None:
        assert args.trajectory_point == -1 or 0 <= args.trajectory_point <= 1, 'trajectory-point must be -1 for centre of mass or between 0 and 1.'

    # Dataset, reconstruction id or trial id is usually required
    if validate_source:
        assert sum([getattr(args, k) is not None for k in ['dataset', 'reconstruction', 'trial', 'trials']]) == 1, \
            'Specify just one of dataset, reconstruction, trial OR trials.'

    return args
