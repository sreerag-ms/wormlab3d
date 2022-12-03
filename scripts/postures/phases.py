import os
from argparse import Namespace, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mayavi import mlab

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Eigenworms
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d_mlab
from wormlab3d.toolkit.util import print_args, str2bool

interactive = False
show_plots = False
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
    parser.add_argument('--spiral-scale', type=str2bool, default=False, help='Spiral the scale.')
    parser.add_argument('--n-examples', type=int, default=2, help='Highlight n points and draw 3d plots for each.')
    parser.add_argument('--max-scale', type=int, default=128, help='Maximum scale factor for components.')
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
                a = r_egs[i] / 2 * c1 + r_egs[i]/2 * np.exp(1j * theta_egs[i]) * c2
                NF = NaturalFrame(a)

            # Align main PCA component to z-axis
            # R, _ = Rotation.align_vectors(NF.pca.components_, np.eye(3))
            # R = R.as_matrix()
            # R = calculate_rotation_matrix(NF.pca.components_[0], np.array([0,0,1]))
            # NF = NaturalFrame(NF.X_pos @ R.T)

            # 3D plot of eigenworm
            fig = plot_natural_frame_3d_mlab(
                NF,
                show_frame_arrows=True,
                n_frame_arrows=16,
                show_pca_arrows=False,
                show_outline=True,
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
                }
            )

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


if __name__ == '__main__':
    if interactive:
        from simple_worm.plot3d import interactive

        interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    plot_phases()
