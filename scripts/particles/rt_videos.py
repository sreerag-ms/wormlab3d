import os
import shutil
import time
from typing import Callable, List, Tuple

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from matplotlib.figure import Figure
from mayavi import mlab

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.particles.args.parameter_args import ParameterArgs
from wormlab3d.particles.cache import get_durations_from_args, get_npas_from_args, get_sim_state_from_args
from wormlab3d.toolkit.plot_utils import make_box_faces_from_dims, make_box_outline, make_cuboid, overlay_image
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import calculate_speeds, smooth_trajectory

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
factor_idxs = [0, 9, 19]


def _init():
    """
    Initialise the arguments and save dir.
    """
    args = get_args(
        include_trajectory_options=True,
        include_msd_options=False,
        include_K_options=False,
        include_planarity_options=True,
        include_helicity_options=False,
        include_manoeuvre_options=True,
        include_approximation_options=True,
        include_pe_options=True,
        include_fractal_dim_options=False,
        include_video_options=True,
        include_evolution_options=False,
        validate_source=False,
    )

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            setattr(args, k, v)
    print_args(args)

    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Ensure that the approx_args is set for the run-tumble model if required
    ParameterArgs.from_args(args)

    # Create output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    return save_dir, args


def _make_label_overlay(
        width: int,
        height: int,
        text: str
) -> np.ndarray:
    """
    Label overlay.
    """
    logger.info('Building label overlay plot.')
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.text(0.1, 0.9, text, ha='left', va='top', fontsize=14, linespacing=1.5)
    fig.tight_layout()
    fig.canvas.draw()
    overlay = np.asarray(fig.canvas.renderer._renderer).take([0, 1, 2], axis=2)
    plt.close(fig)
    return overlay


def _make_traces_plots(
        width: int,
        height: int,
        k_labels: List[str],
        r0s: np.ndarray,
        r1s: np.ndarray,
        r2s: np.ndarray,
        vs: np.ndarray,
        fps: int
) -> Tuple[Figure, Callable]:
    """
    Build a traces plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building traces plot.')
    N = r0s.shape[1]
    ts = np.linspace(0, N / fps, N)
    p_vals = np.sqrt(r0s**2 + r1s**2)
    np_vals = r2s
    vol_vals = vs

    plt.rc('axes', titlesize=14)  # fontsize of the title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=11)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=11)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=13)  # fontsize of the legend
    line_width = 2

    # Plot
    fig, axes = plt.subplots(3, figsize=(width / 100, height / 100), gridspec_kw={
        'hspace': 0.6,
        'top': 0.95,
        'bottom': 0.07,
        'left': 0.1,
        'right': 0.9,
    })

    p_plots = []
    ax_p = axes[0]
    ax_p.set_title('Planar distance $\left(\sqrt{x^2 + y^2}~\\right)$')
    for i, k in enumerate(k_labels):
        p = ax_p.plot(p_vals[i][0], label=f'k = {k}', linewidth=line_width)
        p_plots.append(p[0])
    # ax_r.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_r.transAxes, fontsize=18)
    ax_p.legend(loc='upper left')
    ax_p.set_xlabel('Time (s)')

    np_plots = []
    ax_np = axes[1]
    ax_np.set_title('Non-planar distance $\left(z\\right)$')
    for i, k in enumerate(k_labels):
        nonp = ax_np.plot(np_vals[i][0], label=f'k = {k}', linewidth=line_width)
        np_plots.append(nonp[0])
    # ax_z.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_z.transAxes)
    ax_np.set_xlabel('Time (s)')

    v_plots = []
    ax_v = axes[2]
    ax_v.set_title('Volume explored $\left(x \star y \star z\\right)$')
    for i, k in enumerate(k_labels):
        v = ax_v.plot(vol_vals[i][0], label=f'k = {k}', linewidth=line_width)
        v_plots.append(v[0])
    # ax_v.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=ax_v.transAxes)
    ax_v.set_xlabel('Time (s)')

    def update(frame_idx: int):
        # Update the data
        for i in range(len(k_labels)):
            p_plots[i].set_data(ts[:frame_idx], p_vals[i, :frame_idx])
            np_plots[i].set_data(ts[:frame_idx], np_vals[i, :frame_idx])
            v_plots[i].set_data(ts[:frame_idx], vol_vals[i, :frame_idx])

        # Update the limits
        for ax in axes:
            ax.set_xlim(left=0, right=ts[frame_idx] + 5)
        ax_p.set_ylim(bottom=0, top=np.max(p_vals[:, :frame_idx]) * 1.5)
        ax_np.set_ylim(bottom=0, top=np.max(np_vals[:, :frame_idx]) * 1.5)
        ax_v.set_ylim(bottom=0, top=np.max(vol_vals[:, :frame_idx]) * 1.5)

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def stacked_animation():
    """
    3D stacked animation of cuboids explored by different trajectories.
    """
    save_dir, args = _init()
    args.sim_durations = get_durations_from_args(args)
    args.pauses = [args.nonp_pause_max]  # Fix the pause duration
    args.npas = get_npas_from_args(args)
    args.phi_factor_rt = 1.
    args.pauses = [args.nonp_pause_max]

    duration_idx = 4
    args.sim_duration = args.sim_durations[duration_idx]

    k_factors = args.npas[factor_idxs]
    k_labels = [f'{k:.0E}' for k in k_factors]

    width, height = args.video_width, args.video_height
    traj_anim_rate = 100
    args.traj_anim_rate = traj_anim_rate
    T = int(args.sim_duration / args.sim_dt)

    # Set up plot
    logger.info('Instantiating renderer.')
    mlab.options.offscreen = save_plots

    # Select trajectories
    ptps = []
    trajectories = np.zeros((3, T, 3))
    for i in range(3):
        if i == 0:
            args.phi_factor_rt = k_factors[0]
        elif i == 1:
            args.phi_factor_rt = k_factors[2]
        elif i == 2:
            args.phi_factor_rt = k_factors[1]
        SS = get_sim_state_from_args(args)

        # Select an exemplar trajectory
        Xt = SS.get_Xt()
        ptp = np.ptp(Xt, axis=1)

        # Pick the trajectory with the largest ptp in the x and y-dimensions
        if i == 0:
            traj_idx = np.argmax(ptp[:, 0] * ptp[:, 1])
            trajectories[0] = Xt[traj_idx]

        # Pick the trajectory with the largest ptp in the z-dimension
        elif i == 1:
            traj_idx = np.argmax(ptp[:, 2])
            trajectories[2] = Xt[traj_idx]

        # Pick the trajectory with the largest volume, subject to the ptp values being in the middle
        else:
            max_xy = ptps[0][0] * ptps[0][1]
            max_z = ptps[1][2]
            exclude_xy = ptp[:, 0] * ptp[:, 1] > max_xy
            exclude_z = ptp[:, 2] > max_z
            exclude = exclude_xy | exclude_z
            vols = np.prod(ptp, axis=1)
            vols[exclude] = 0
            traj_idx = np.argmax(vols)
            trajectories[1] = Xt[traj_idx]

        ptps.append(ptp[traj_idx])

    # Cuboid and trajectory colours
    colours = np.array([
        [31, 119, 180, 1],
        [255, 127, 13, 1],
        [86, 179, 86, 1],
    ]) / 255
    alphas = np.array([0.2, 0.4, 0.5])
    cuboid_args = {
        i: dict(
            colour=tuple(colours[i][:3]),
            opacity=alphas[i],
            draw_outline=True,
            outline_colour=tuple(colours[i][:3]),
            outline_opacity=alphas[i] + 0.2,
            outline_tube_radius=0.02,
        )
        for i in range(3)
    }

    # Generate separate figures for each k-factor
    figs = {}
    paths = {}
    cuboids = {}
    label_overlays = {}
    for i, k_factor in enumerate(k_factors):
        colour = tuple(colours[i][:3])

        # Setup figure
        fig = mlab.figure(size=(width, height / 3 * 2), bgcolor=(1, 1, 1))
        fig.scene.renderer.use_depth_peeling = True
        fig.scene.renderer.maximum_number_of_peels = 32
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20
        figs[i] = fig

        # Add trajectory path
        x, y, z = trajectories[i, 0].T
        paths[i] = mlab.plot3d(x, y, z, color=colour, tube_radius=0.08, figure=figs[i])

        # Add cuboid
        meshes, outline = make_cuboid(
            dims=np.array([0.1, 0.1, 0.1]),
            fig=figs[i],
            **cuboid_args[i]
        )
        cuboids[i] = {'meshes': meshes, 'outline': outline}

        # Build label overlays
        label_overlays[i] = _make_label_overlay(width / 2, height / 3, f'k = {k_labels[i]}')

    # Calculate the max r values over time
    logger.info('Calculating r values over time.')
    r0s = np.maximum.accumulate(np.abs(trajectories[..., 0]), axis=1)
    r1s = np.maximum.accumulate(np.abs(trajectories[..., 1]), axis=1)
    r2s = np.maximum.accumulate(np.abs(trajectories[..., 2]), axis=1)
    vs = r0s * r1s * r2s

    fig_traces, update_traces_plot = _make_traces_plots(
        width=int(width / 2),
        height=height,
        k_labels=k_labels,
        r0s=r0s,
        r1s=r1s,
        r2s=r2s,
        vs=vs,
        fps=25
    )

    def update_scenes(t):
        # Update paths
        max_dist = 0
        for i, npas in enumerate(k_factors):
            figs[i].scene.disable_render = True

            # Update trajectory path
            X = trajectories[i][:t]
            dims = np.ptp(X, axis=0)
            X = X - X.min(axis=0) - dims / 2
            x, y, z = X.T
            paths[i].mlab_source.reset(x=x, y=y, z=z)

            # Update cuboid
            faces = make_box_faces_from_dims(dims=dims)
            for m, face in zip(cuboids[i]['meshes'], faces):
                x, y, z = face.T
                x = [[x[0], x[3]], [x[1], x[2]]]
                y = [[y[0], y[3]], [y[1], y[2]]]
                z = [[z[0], z[3]], [z[1], z[2]]]
                m.mlab_source.reset(x=x, y=y, z=z)
            lines = make_box_outline(dims=dims)
            for l, line in zip(cuboids[i]['outline'], lines):
                x, y, z = line.T
                l.mlab_source.reset(x=x, y=y, z=z)

            # Update view and render
            max_dist = max(max_dist, np.max(np.linalg.norm(X, axis=-1)))
            mlab.view(
                figure=figs[i],
                azimuth=np.fmod(t / traj_anim_rate / 2, 360),
                elevation=np.sin(t / traj_anim_rate / 50) * 20 + 90,
                distance=10 + max_dist * 3,
            )
            figs[i].scene.disable_render = False
            figs[i].scene.render()

    if save_plots:
        output_path = save_dir / 'sim_animation_stacked'

        # Write meta data
        meta = to_dict(args)
        meta['date'] = START_TIMESTAMP
        with open(output_path.with_suffix('.yml'), 'w') as f:
            yaml.dump(meta, f)

        # Initialise ffmpeg process
        output_args = {
            'pix_fmt': 'yuv444p',
            'vcodec': 'libx264',
            'r': 25,
            'metadata:g:0': 'title=Simulation output.',
            'metadata:g:1': 'artist=Leeds Wormlab',
            'metadata:g:2': f'year={time.strftime("%Y")}',
        }

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .output(str(output_path) + '.mp4', **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # Overlay plot info on top of images panel
        fig.scene._lift()
        n_frames = int(T / traj_anim_rate)
        for t in range(1, n_frames):
            if t > 1 and t % 50 == 0:
                logger.info(f'Rendering frame {t}/{n_frames}.')
            update_scenes(t * traj_anim_rate)
            update_traces_plot(t * traj_anim_rate)

            # Take the traces render
            plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)

            # Collate the 3D views with overlaid labels
            frames_t = []
            for i, fig in figs.items():
                screenshot = mlab.screenshot(mode='rgb', antialiased=True, figure=figs[i])
                screenshot = cv2.resize(screenshot, dsize=(int(width / 2), int(height / 3)),
                                        interpolation=cv2.INTER_AREA)
                screenshot = overlay_image(screenshot, label_overlays[i], x_offset=0, y_offset=0)
                frames_t.append(screenshot)

            # Stitch together the data
            frame_data = np.concatenate([np.concatenate(frames_t), plot_traces], axis=1)
            if frame_data.shape[0] < height:
                frame_data = np.concatenate([
                    frame_data,
                    np.ones((height - frame_data.shape[0], width, 3), dtype=np.uint8) * 255
                ], axis=0)
            if frame_data.shape[1] < width:
                frame_data = np.concatenate([
                    frame_data,
                    np.ones((height, width - frame_data.shape[1], 3), dtype=np.uint8) * 255
                ], axis=1)
            if frame_data.shape[0] != height or frame_data.shape[1] != width:
                raise RuntimeError('Frame is the wrong shape!')

            frame = Image.fromarray(frame_data, 'RGB')
            process.stdin.write(frame.tobytes())

        # Flush video
        process.stdin.close()
        process.wait()

    if show_plots:
        @mlab.animate(delay=50)
        def animate():
            for t in range(1, T):
                update_scenes(t)
                yield

        animate()
        mlab.show()


def single_trajectory_animation():
    """
    3D animation of a single trajectory.
    """
    save_dir, args = _init()
    args.sim_durations = get_durations_from_args(args)
    args.pauses = [args.nonp_pause_max]  # Fix the pause duration
    args.npas = get_npas_from_args(args)  # Propagate the npas (phi factors)
    args.phi_factor_rt = 1.
    args.pauses = [args.nonp_pause_max]
    args.sim_durations = [args.sim_duration]

    width = args.video_width
    height = args.video_height
    traj_anim_rate = 100
    args.traj_anim_rate = traj_anim_rate
    T = int(args.sim_duration / args.sim_dt)

    # Set up plot
    logger.info('Instantiating renderer.')
    mlab.options.offscreen = save_plots
    fig = mlab.figure(size=(width, height), bgcolor=(1, 1, 1))
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Fetch trajectories
    SS = get_sim_state_from_args(args)
    Xt = SS.get_Xt()

    # Pick a typical trajectory
    ptp = np.ptp(Xt, axis=1)
    avg_ptp = ptp.mean(axis=0)
    best_fit_idx = np.argmin(np.linalg.norm(ptp - avg_ptp, axis=-1))
    traj = Xt[best_fit_idx]

    # Calculate speeds
    s = calculate_speeds(Xt[best_fit_idx])
    cmap = plt.get_cmap('plasma')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Smooth the trajectory for nice camera tracking
    mps = smooth_trajectory(traj, window_len=3001)

    # Draw start of path
    x, y, z = traj[0].T
    path = mlab.plot3d(x, y, z, s[0], tube_radius=0.08, vmin=s.min() * 0.9, vmax=s.max() * 1.1)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    def update_scene(t):
        fig.scene.disable_render = True
        X = traj[:t * traj_anim_rate]
        max_dist = np.log(1 + np.max(np.linalg.norm(X, axis=-1)))
        x, y, z = X.T
        path.mlab_source.scalars = s[:t * traj_anim_rate]
        path.mlab_source.reset(x=x, y=y, z=z)
        fig.scene.disable_render = False

        mlab.view(
            azimuth=np.fmod(t / 2, 360),
            elevation=np.sin(t / 50) * 20 + 90,
            distance=10 + max_dist * 3,
            focalpoint=mps[t * traj_anim_rate]
        )

        fig.scene.render()

    # Initialise ffmpeg process
    output_path = save_dir / 'sim_single_trajectory'
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': 25,
        'metadata:g:0': 'title=Simulation output.',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(str(output_path) + '.mp4', **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    fig.scene._lift()
    for t in range(1, int(T / traj_anim_rate)):
        update_scene(t)
        screenshot = mlab.screenshot(mode='rgb', antialiased=True)
        frame = Image.fromarray(screenshot, 'RGB')
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    # Write meta data
    meta = to_dict(args)
    meta['date'] = START_TIMESTAMP
    with open(output_path.with_suffix(output_path.suffix + '.yml'), 'w') as f:
        yaml.dump(meta, f)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    stacked_animation()
    # single_trajectory_animation()
