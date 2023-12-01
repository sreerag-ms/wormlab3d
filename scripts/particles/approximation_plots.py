import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from matplotlib.figure import Figure
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tvtk.tools import visual

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Trial
from wormlab3d.particles.tumble_run import calculate_curvature, find_approximation
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.plot_utils import to_rgb
from wormlab3d.toolkit.util import print_args, normalise, to_dict
from wormlab3d.trajectories.cache import get_trajectory

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
interactive_plots = False
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot tumble and run approximations.')

    # Trajectory args
    parser.add_argument('--trial', type=str, required=True,
                        help='Trial by id.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    # Approximation args
    parser.add_argument('--error-limits', type=lambda s: [float(item) for item in s.split(',')], required=True,
                        help='Error limits.')
    parser.add_argument('--alpha0', type=int, default=201,
                        help='Initial smoothing window for calculating curvature (in frames).')
    parser.add_argument('--beta0', type=int, default=100,
                        help='Initial minimum peak height.')
    parser.add_argument('--gamma0', type=int, default=500,
                        help='Initial minimum distance between peaks.')
    parser.add_argument('--planarity-window', type=int, default=3,
                        help='PCA window (number of vertices).')

    # 3D plots
    parser.add_argument('--width-3d', type=int, default=1000, help='Width of 3D plot (in pixels).')
    parser.add_argument('--height-3d', type=int, default=1000, help='Height of 3D plot (in pixels).')
    parser.add_argument('--distance', type=float, default=4., help='Camera distance (in mm).')
    parser.add_argument('--azimuth', type=int, default=70, help='Azimuth.')
    parser.add_argument('--elevation', type=int, default=45, help='Elevation.')
    parser.add_argument('--roll', type=int, default=45, help='Roll.')

    args = parser.parse_args()

    print_args(args)

    return args


def _plot_curvature(
        trial: Trial,
        k: np.ndarray,
        tumble_idxs: np.ndarray
) -> Figure:
    """
    Plot the curvature.
    """
    dt = 1 / trial.fps
    ts = np.arange(len(k)) * dt
    tumble_ts = tumble_idxs * dt

    # Set up plot
    plt.rc('axes', labelsize=7)  # fontsize of the X label
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('xtick.major', pad=2, size=3)

    fig, ax = plt.subplots(1, figsize=(1.6, 1.2), gridspec_kw={
        'bottom': 0.15,
        'top': 0.99,
        'left': 0.02,
        'right': 0.99
    })

    # Plot curvature
    ax.plot(ts, k, linewidth=1, color='darkorange')

    # Add vertical lines at the vertices
    vline_args = dict(color='blue', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, **vline_args)
    for d in tumble_ts:
        ax.axvline(x=d, **vline_args)
    ax.axvline(x=ts[-1], **vline_args)

    # Finish plot
    # ax.set_xlabel('Time ()')
    # ax.set_ylabel('Curvature (mm$^{-1}$)')
    ax.set_ylabel('Curvature', labelpad=1)
    ax.set_xticks([0, 300])
    ax.set_xticklabels([0, '5 min'])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


def _plot_3d_matplotlib(
        args: Namespace,
        X: np.ndarray,
        k: np.ndarray,
        vertices: np.ndarray,
        show_real: bool,
        show_approx: bool,
        show_markers: bool = True,
):
    """
    Plot 3D trajectory and approximation.
    """

    fig = plt.figure(figsize=(4, 4))

    # 3D trajectory of approximation
    ax = fig.add_subplot(111, projection='3d')

    # Add vertex markers
    if show_markers:
        ax.scatter(*vertices.T, color='blue', marker='x', s=100, alpha=0.6, zorder=1)

    # Add approximation trajectory
    if show_approx:
        points = vertices[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, color='blue', zorder=5, linewidth=2, linestyle=':', alpha=0.5)
        ax.add_collection(lc)

    # Actual 3D trajectory
    if show_real:
        ax.scatter(*X.T, c=k, cmap='Reds', s=10, alpha=0.4, zorder=-1)

    ax.axis('off')
    equal_aspect_ratio(ax)
    fig.tight_layout()

    return fig


def _plot_3d(
        args: Namespace,
        X: np.ndarray,
        k: np.ndarray,
        vertices: np.ndarray,
        show_real: bool,
        show_approx: bool,
):
    """
    Plot 3D trajectory and approximation.
    """

    # Set up mlab figure
    fig = mlab.figure(size=(args.width_3d * 2, args.height_3d * 2), bgcolor=(1, 1, 1))
    visual.set_viewer(fig)

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 64
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add vertex markers
    mlab.points3d(*vertices.T, scale_factor=0.5, color=to_rgb('blue'), mode='axes', line_width=3, figure=fig)

    # Connect vertices with straight lines
    if show_approx:
        mlab.plot3d(*vertices.T, color=to_rgb('blue'), tube_radius=0.1, figure=fig)

    # Actual 3D trajectory
    if show_real:
        cmap = plt.get_cmap('Reds')
        cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
        path = mlab.plot3d(*X.T, k, opacity=0.7, tube_radius=0.1)
        path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Focus on the middle of the manoeuvre
    centre = X.min(axis=0) + np.ptp(X, axis=0) / 2

    # Draw plot
    mlab.view(
        figure=fig,
        distance=args.distance,
        focalpoint=centre,
        azimuth=args.azimuth,
        elevation=args.elevation,
        roll=args.roll,
    )

    # # Useful for getting the view parameters when recording from the gui:
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [10.884122637407016, -14.764964514351991, 12.077723301604369]
    # scene.scene.camera.focal_point = [1.3738853055960125, 0.16831670636757678, -0.007383090614838661]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.0931855734593002, 0.5898109392083042, 0.802146810059904]
    # scene.scene.camera.clipping_range = [10.351943608887044, 35.481150601253425]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    return fig


def plot_approximations(
        plot_curvature: bool = True,
        plot_3d: bool = True,
        use_mayavi: bool = True
):
    """
    Plot trial approximations.
    """
    args = parse_args()
    trial = Trial.objects.get(id=args.trial)

    # Make output dir and save args
    output_dir = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id:03d}'
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / 'args.yml', 'w') as f:
            yaml.dump(to_dict(args), f)

    # Fetch trajectory and centre
    X, _ = get_trajectory(trial_id=trial.id, tracking_only=True)
    if X.ndim == 3:
        X = X.mean(axis=1)
    X -= X.mean(axis=0)

    # Calculate tangent
    e0 = normalise(np.gradient(X, axis=0))

    # Generate approximations
    results = {}
    for error_limit in args.error_limits:
        logger.info(f'Finding approximation for error limit = {error_limit:.3f}.')

        # Find approximation
        approx, distance, height, smooth_e0, smooth_K = find_approximation(
            X=X,
            e0=e0,
            error_limit=error_limit,
            planarity_window_vertices=args.planarity_window,
            distance_first=args.gamma0,
            height_first=args.beta0,
            smooth_e0_first=args.alpha0,
            smooth_K_first=args.alpha0,
            max_iterations=100
        )
        vertices = approx[1]
        tumble_idxs = approx[2]

        # Store result values
        results[error_limit] = {
            'alpha': smooth_e0,
            'beta': height,
            'gamma': distance,
            'eps': float(np.mean(np.sum((X - approx[0])**2, axis=-1)))
        }

        # Recalculate the curvature
        assert smooth_e0 == smooth_K
        k = calculate_curvature(e0, smooth_e0=smooth_e0, smooth_K=smooth_K)

        # Plot the curvature
        if plot_curvature:
            fig = _plot_curvature(
                trial=trial,
                k=k,
                tumble_idxs=tumble_idxs
            )
            if save_plots:
                path = output_dir / f'curvature_{error_limit:.3f}.{img_extension}'
                logger.info(f'Saving plot to {path}.')
                plt.savefig(path, transparent=True)
            if show_plots:
                plt.show()
            plt.close(fig)

        # Plot 3D
        if plot_3d:
            for variant in ['real', 'approx']:
                plot_args = dict(
                    args=args,
                    X=X,
                    k=k,
                    vertices=vertices,
                    show_real=variant == 'real',
                    show_approx=variant == 'approx',
                )

                if use_mayavi:
                    fig = _plot_3d(**plot_args)
                    if save_plots:
                        path = output_dir / f'3D_{variant}_{error_limit:.3f}.png'
                        logger.info(f'Saving 3D plot to {path}.')
                        fig.scene._lift()
                        img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                        img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                        img.save(path)
                        mlab.clf(fig)
                        mlab.close()
                        logger.info(f'Saving plot to {path}.')
                    if show_plots:
                        mlab.show()
                else:
                    if variant == 'real':
                        fig = _plot_3d_matplotlib(**plot_args, show_markers=False)
                    else:
                        fig = _plot_3d_matplotlib(**plot_args, show_markers=True)
                    if save_plots:
                        path = output_dir / f'3D_{variant}_{error_limit:.3f}.png'
                        logger.info(f'Saving 3D plot to {path}.')
                        plt.savefig(path, transparent=True)
                    if show_plots:
                        plt.show()
                    plt.close(fig)

                    # Make a second plot showing only the markers
                    if variant == 'real':
                        plot_args['show_real'] = False
                        fig = _plot_3d_matplotlib(**plot_args, show_markers=True)
                        if save_plots:
                            path = output_dir / f'3D_markers_{error_limit:.3f}.png'
                            logger.info(f'Saving 3D plot to {path}.')
                            plt.savefig(path, transparent=True)
                        if show_plots:
                            plt.show()
                        plt.close(fig)

    # Save result values
    if save_plots:
        with open(output_dir / 'results.yml', 'w') as f:
            yaml.dump(results, f)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    if not show_plots:
        mlab.options.offscreen = True

    plot_approximations(
        plot_curvature=True,
        plot_3d=True,
        use_mayavi=False
    )
