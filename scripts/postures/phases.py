import os
from argparse import Namespace, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mayavi import mlab
from scipy.stats import gaussian_kde

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT, interactive
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Eigenworms, Reconstruction
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d_mlab
from wormlab3d.toolkit.plot_utils import to_rgb
from wormlab3d.toolkit.util import print_args, str2bool
from wormlab3d.trajectories.cache import get_trajectory

interactive_plots = False
show_plots = True
save_plots = True
img_extension = 'svg'
eigenworm_length = 1
eigenworm_scale = 64
cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot example curves from eigenworms.')
    parser.add_argument('--ew', type=str, required=True, help='Eigenworms by id.')
    parser.add_argument('--plot-components', type=lambda s: [item for item in s.split(',')],
                        default='0,1,0+1', help='Comma delimited list of component idxs to plot.')

    # Eigenworm rotation plots
    parser.add_argument('--spiral-scale', type=str2bool, default=False, help='Spiral the scale.')
    parser.add_argument('--n-examples', type=int, default=2, help='Highlight n points and draw 3d plots for each.')
    parser.add_argument('--max-scale', type=int, default=128, help='Maximum scale factor for components.')

    # Coefficient phases plot args
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--frame-offset', type=int, help='Frame number to start numbering from.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')

    args = parser.parse_args()

    print_args(args)

    return args


def plot_phases():
    """
    Plot phases and examples.
    """
    args = parse_args()
    ew = Eigenworms.objects.get(id=args.ew)

    # Make the output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_modes={",".join([str(c_) for c_ in args.plot_components])}' \
                           f'_spiral={int(args.spiral_scale)}' \
                           f'_egs={args.n_examples}'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Make the scale and phases
    k = 1000
    thetas = np.linspace(0, -2 * np.pi, k)
    if args.spiral_scale:
        rs = np.linspace(0, args.max_scale, k)
        r_egs = np.linspace(0, args.max_scale, args.n_examples)
        theta_egs = np.linspace(0, -2 * np.pi, args.n_examples)
    else:
        rs = np.ones(k) * args.max_scale
        r_egs = np.ones(args.n_examples) * args.max_scale
        theta_egs = np.linspace(0, -2 * np.pi, args.n_examples + 1)[:-1]

    # Plot the phase on a polar plot highlighting the examples
    plt.rc('axes', labelsize=6)  # fontsize of the X label
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=0.5, size=1)
    plt.rc('ytick.major', pad=0.5, size=1)
    fig = plt.figure(figsize=(1, 1))
    gs = GridSpec(1, 1, left=0.11, right=0.89, top=0.89, bottom=0.11)
    ax = fig.add_subplot(gs[0, 0], projection='polar')
    ax.plot(thetas, rs, linewidth=2)
    ax.scatter(theta_egs, r_egs, color='red', s=50, zorder=100)
    ax.set_rlim(top=args.max_scale * 1.4)
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)
    ax.spines['polar'].set_visible(False)
    if save_plots:
        path = save_dir / f'polar_plot.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()
    plt.close(fig)

    for c in args.plot_components:
        logger.info(f'Making worms for component {c}.')
        cs = c.split('+')
        NFs = []

        if len(cs) == 1:
            component = ew.components[int(cs[0])]
            component = component * np.exp(1j * np.angle(component[0]))
        elif len(cs) == 2:
            c1 = ew.components[int(cs[0])]
            c1 = c1 * np.exp(1j * np.angle(c1[0]))
            c2 = ew.components[int(cs[1])]
            c2 = c2 * np.exp(1j * np.angle(c2[0]))
        else:
            raise RuntimeError(f'Unrecognised component {c}')

        # Plot the example shapes
        for i in range(args.n_examples):
            logger.info(f'Making example {i + 1}/{args.n_examples}')

            if len(cs) == 1:
                a = r_egs[i] * np.exp(1j * theta_egs[i])
                NF = NaturalFrame(component * a)
            else:
                a = r_egs[i] / np.sqrt(2) * c1 + r_egs[i] / np.sqrt(2) * np.exp(1j * theta_egs[i]) * c2
                NF = NaturalFrame(a)
            NFs.append(NF)

            # 3D plot of eigenworm
            fig = plot_natural_frame_3d_mlab(
                NF,
                show_frame_arrows=True,
                n_frame_arrows=16,
                show_pca_arrows=False,
                show_outline=False,
                show_axis=False,
                offscreen=not interactive,
                azimuth=90,
                elevation=160,
                roll=-50,
                distance=1.8,
                arrow_scale=0.12,
                arrow_opts={
                    'radius_shaft': 0.02,
                    'radius_cone': 0.1,
                    'length_cone': 0.2,
                },
                midline_opts={
                    'line_width': 8
                },
                outline_opts={
                    'color': to_rgb('red')
                }
            )

            # orientation_axes = mlab.orientation_axes(xlabel='x', ylabel='y', zlabel='z')
            # orientation_axes.axes.normalized_label_position = [1.7, 1.7, 1.5]
            # ax_col = 'darkgrey'
            # orientation_axes.axes.x_axis_caption_actor2d.caption_text_property.color = to_rgb('black')
            # orientation_axes.axes.x_axis_tip_property.color = to_rgb(ax_col)
            # orientation_axes.axes.x_axis_shaft_property.color = to_rgb(ax_col)
            # orientation_axes.axes.y_axis_caption_actor2d.caption_text_property.color = to_rgb('black')
            # orientation_axes.axes.y_axis_tip_property.color = to_rgb(ax_col)
            # orientation_axes.axes.y_axis_shaft_property.color = to_rgb(ax_col)
            # orientation_axes.axes.z_axis_caption_actor2d.caption_text_property.color = to_rgb('black')
            # orientation_axes.axes.z_axis_tip_property.color = to_rgb(ax_col)
            # orientation_axes.axes.z_axis_shaft_property.color = to_rgb(ax_col)

            if save_plots:
                path = save_dir / f'c={c}_eg={i:02d}_r={r_egs[i]:.1f}_theta={theta_egs[i] / np.pi:.1f}pi.png'
                logger.info(f'Saving plot to {path}.')
                fig.scene._lift()
                img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                img.save(path)
                mlab.clf(fig)
                mlab.close()

            if show_plots:
                if interactive:
                    mlab.show()
                else:
                    fig.scene._lift()
                    img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                    mlab.clf(fig)
                    mlab.close()
                    fig_mpl = plt.figure(figsize=(10, 10))
                    ax = fig_mpl.add_subplot()
                    ax.imshow(img)
                    ax.axis('off')
                    fig_mpl.tight_layout()
                    plt.show()
                    plt.close(fig_mpl)

        # Plot curvatures
        plt.rc('axes', labelsize=6)  # fontsize of the axis labels
        plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2)
        plt.rc('ytick.major', pad=1, size=2)
        fig, ax = plt.subplots(1, figsize=(0.85, 0.7), gridspec_kw={
            'left': 0.28,
            'right': 0.95,
            'top': 0.95,
            'bottom': 0.27,
        })
        ax.spines['top'].set_visible(False)

        trim_start = 3
        trim_end = 5
        if len(cs) == 1:
            ax.plot(NF.kappa[trim_start:-trim_end], color='blue')
        else:
            cmap = plt.get_cmap('CMRmap')
            colours = cmap(np.linspace(0, 1, len(NFs) + 2)[1:-1])
            for i, NF in enumerate(NFs):
                ax.plot(NF.kappa[:-1], color=colours[i])

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=trim_start, right=NF.N - trim_end - 1)
        ax.set_xticks([trim_start, NF.N - trim_end - 1])
        ax.set_xticklabels(['H', 'T'])

        # Set up y-axis
        ax.set_ylim(bottom=0, top=12)
        ax.set_yticks([0, 5, 10])
        ax.set_ylabel('Curvature (mm$^{-1}$)', labelpad=-3)

        if save_plots:
            path = save_dir / f'curvature_{c}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()
        plt.close(fig)
        plt.show()


def coefficient_phases(style: str = 'paper'):
    """
    Coefficient phase plots.
    """
    args = parse_args()
    ew = Eigenworms.objects.get(id=args.ew)
    assert args.reconstruction is not None, '--reconstruction must be set!'
    assert args.start_frame is not None, '--start-frame must be set!'
    assert args.end_frame is not None, '--end-frame must be set!'
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    n_mesh_points = 100
    plot_components = [int(c) for c in args.plot_components if '+' not in c]
    if len(plot_components) == 0:
        raise RuntimeError('No components to plot!')

    common_args = {
        'reconstruction_id': args.reconstruction,
        'smoothing_window': 3,
        'natural_frame': True,
    }

    # Eigenworms embeddings for entire clip
    logger.info('Fetching eigenworms embeddings for entire clip.')
    Z_all, meta = get_trajectory(**common_args)
    # Z_all, meta = get_trajectory(**common_args, start_frame=args.start_frame, end_frame=args.end_frame)
    X_ew_all = ew.transform(np.array(Z_all))

    # Eigenworms embeddings just for requested frames
    logger.info('Fetching eigenworms embeddings for requested frames.')
    Z_traj, meta = get_trajectory(**common_args, start_frame=args.start_frame, end_frame=args.end_frame)
    X_ew_traj = ew.transform(np.array(Z_traj))

    # Rotate the complex numbers so the phase of the first component is zero
    X_ew_all *= np.exp(-1j * np.angle(X_ew_all[:, 0]))[:, None]
    X_ew_traj *= np.exp(-1j * np.angle(X_ew_traj[:, 0]))[:, None]

    # Collate and get limits
    x_all = np.stack([np.real(X_ew_all[:, c]) for c in plot_components])
    y_all = np.stack([np.imag(X_ew_all[:, c]) for c in plot_components])
    x_traj = np.stack([np.real(X_ew_traj[:, c]) for c in plot_components])
    y_traj = np.stack([np.imag(X_ew_traj[:, c]) for c in plot_components])
    x_max = np.abs(x_all.flatten()).max()
    y_max = np.abs(y_all.flatten()).max()
    x_max = np.abs(x_traj.flatten()).max()
    y_max = np.abs(y_traj.flatten()).max()
    x_max = max(x_max, y_max)
    y_max = x_max

    # Construct colours
    colours = np.linspace(0, 1, len(X_ew_traj))
    cmap = plt.get_cmap('winter_r')

    # Axis lines
    ax_args = {
        'color': 'grey',
        'linestyle': '--',
        'linewidth': 0.5,
    }

    # pads
    style_vars = {
        'paper': {
            'tr_padx': 2,
            'tr_pady': 1,
            'heat_padx': 4,
            'heat_pady': 2,
            'heat_lw': 1,
        },
        'thesis': {
            'tr_padx': 2,
            'tr_pady': 4,
            'heat_padx': 4,
            'heat_pady': 3,
            'heat_lw': 1.2,
        }
    }[style]

    # Interpolate data to make a surface
    def make_surface(x_, y_):
        X, Y = np.mgrid[-x_max:x_max:complex(n_mesh_points), -y_max:y_max:complex(n_mesh_points)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x_, y_])
        kernel = gaussian_kde(values)  # , bw_method=0.2)
        Z = np.reshape(kernel(positions).T, X.shape)
        return Z

    def plot_heatmap(ax, c, x_all_, y_all_, x_traj_, y_traj_):
        Z = make_surface(x_all_, y_all_)
        ax.set_xlabel(f'Re($\lambda_{c}$)', labelpad=style_vars['heat_padx'])
        ax.set_ylabel(f'Im($\lambda_{c}$)', labelpad=style_vars['heat_pady'])

        # Add the heatmap
        ax.axvline(x=0, **ax_args)
        ax.axhline(y=0, **ax_args)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[-x_max, x_max, -y_max, y_max])

        # Overlay the trajectory
        X = np.stack([x_traj_, y_traj_], axis=1)
        points = X[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=colours, cmap=cmap, alpha=colours, linewidths=style_vars['heat_lw'])
        ax.add_collection(lc)
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([-y_max, y_max])
        ax.set_yticks([])
        ax.set_xticks([])

    # Plot
    if style == 'paper':
        plt.rc('axes', labelsize=7, labelpad=1)  # fontsize of the labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2, size=2)
        plt.rc('ytick.major', pad=2, size=2)
        fig, axes = plt.subplots(1, len(plot_components), figsize=(len(plot_components) * 1.4, 1.4), gridspec_kw={
            'wspace': 0.25,
            'left': 0.05,
            'right': 0.99,
            'top': 0.96,
            'bottom': 0.2,
        })

    else:
        plt.rc('axes', labelsize=9)  # fontsize of the X label
        plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=8)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)
        fig, axes = plt.subplots(1, len(plot_components), figsize=(len(plot_components) * 1.16, 1.4), gridspec_kw={
            'wspace': 0.3,
            'left': 0.07,
            'right': 0.99,
            'top': 0.86,
            'bottom': 0.24,
        })

    for i, c in enumerate(plot_components):
        logger.info(f'Plotting component {c}.')
        ci = c if style == 'thesis' else c + 1

        # First component is only real so just draw a normal plot
        if c == 0:
            N = len(Z_traj)
            ts = np.arange(N, dtype=np.float32) + args.start_frame
            if args.frame_offset is not None:
                ts -= args.frame_offset
            if args.x_label == 'time':
                ts /= reconstruction.trial.fps

            ax = axes[i]
            for j in range(N - 1):
                ax.plot(ts[j:j + 2], x_traj[i, j:j + 2], c=cmap(colours[j]), alpha=colours[j])
            ax.set_yticks([0, 50, 100])
            asp_sqr = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
            asp_targ = 8 / 100
            ax.set_aspect(asp_targ, anchor='C', adjustable='box' if asp_targ > asp_sqr else 'datalim')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.set_ylabel(f'Re($\lambda_{ci}$)', labelpad=style_vars['tr_pady'])
            if args.x_label == 'time':
                ax.set_xlabel('Time (s)', labelpad=style_vars['tr_padx'])
            else:
                ax.set_xlabel('Frame #', labelpad=style_vars['tr_padx'])

        # Draw others as trajectories on top of heatmaps
        else:
            plot_heatmap(axes[i], ci, x_all[i], y_all[i], x_traj[i], y_traj[i])

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_coefficient_phases' \
                           f'_r={reconstruction.id}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
                           f'_ew={ew.id}' \
                           f'_c={",".join([str(c) for c in plot_components])}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if interactive_plots:
        interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # plot_phases()
    coefficient_phases(style='thesis')
