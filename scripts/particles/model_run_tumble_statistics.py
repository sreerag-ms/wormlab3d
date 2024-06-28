import os
import shutil

import numpy as np
import yaml
from copulas.multivariate import GaussianMultivariate
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde, norm, pearsonr

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset
from wormlab3d.particles.cache import get_sim_state_from_args
from wormlab3d.particles.tumble_run import generate_or_load_ds_statistics
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.util import smooth_trajectory

# show_plots = True
# save_plots = False
show_plots = False
save_plots = True
interactive_plots = False

DATA_KEYS = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles', 'tumble_idxs']


def _init(include_all_runs: bool = False, return_args: bool = False):
    """
    Initialise the arguments, save dir and load the dataset statistics.
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
        include_video_options=False,
        include_evolution_options=True,
        validate_source=False,
    )

    # Load arguments from spec file
    if (LOGS_PATH / 'spec.yml').exists():
        with open(LOGS_PATH / 'spec.yml') as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in spec.items():
            assert hasattr(args, k), f'{k} is not a valid argument!'
            if k in ['theta_dist_params', 'phi_dist_params']:
                v = [float(vv) for vv in v.split(',')]
            setattr(args, k, v)
    print_args(args)

    assert args.batch_size is not None, 'Must provide a batch size!'
    assert args.sim_duration is not None, 'Must provide a simulation duration!'
    assert args.sim_dt is not None, 'Must provide a simulation time step!'
    assert args.approx_noise is not None, 'Must provide an approximation noise level!'

    # Create output directory
    if save_plots:
        save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the arguments
        if (LOGS_PATH / 'spec.yml').exists():
            shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
        with open(save_dir / 'args.yml', 'w') as f:
            yaml.dump(to_dict(args), f)
    else:
        save_dir = None

    approx_args = dict(
        approx_method=args.approx_method,
        error_limits=[args.approx_error_limit],
        planarity_window=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.smoothing_window_K,
        smooth_K_first=args.smoothing_window_K,
        use_euler_angles=args.approx_use_euler_angles,
    )
    if include_all_runs:
        approx_args['min_run_speed_duration'] = (0, 10000)

    # Fetch dataset
    ds = Dataset.objects.get(id=args.dataset)

    # Generate or load tumble/run values
    ds_stats = generate_or_load_ds_statistics(
        ds=ds,
        rebuild_cache=args.regenerate,
        **approx_args
    )
    # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles, tumble_idxs
    data_values = {k: ds_stats[i + 1][0] for i, k in enumerate(DATA_KEYS)}

    if return_args:
        return save_dir, data_values, args
    return save_dir, data_values


def _make_surface(x, y, x_min, x_max, y_min, y_max, n_mesh_points, bw_method=None):
    """
    Interpolate data to make a surface.
    """
    X, Y = np.mgrid[x_min:x_max:complex(n_mesh_points), y_min:y_max:complex(n_mesh_points)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, bw_method=bw_method)
    Z = np.reshape(kernel(positions).T, X.shape)
    return Z


def model_runs(
        show_heatmap: bool = False,
        show_real: bool = True,
        show_synth: bool = True,
        layout: str = 'default',
):
    """
    Build a bivariate probability distribution for the runs based on speeds and durations.
    """
    save_dir, data_values = _init()
    durations = data_values['durations']
    speeds = data_values['speeds']

    # Fit a copula to the durations and speeds
    data = np.column_stack((durations, speeds))
    copula = GaussianMultivariate()
    copula.fit(data)

    # Generate synthetic samples from the joint distribution
    synth = copula.sample(len(data))

    # Plot the results
    if layout == 'paper':
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
        plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
        plt.rc('axes', labelsize=6, labelpad=0)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=6)  # fontsize of the legend
        plt.rc('ytick.major', pad=1, size=2)
        plt.rc('xtick.major', pad=1, size=2)
        plt.rc('xtick.minor', size=1)
        fig, ax = plt.subplots(1, 1, figsize=(1.39, 0.8), gridspec_kw={
            'top': 0.98,
            'bottom': 0.22,
            'left': 0.18,
            'right': 0.98,
        })
        x_label = 'Duration (s)'
        y_label = 'Speed (mm/s)'
        scatter_args = dict(alpha=0.3, s=0.1)
        legend_args = dict(markerscale=10, handlelength=1, handletextpad=0.2,
                           labelspacing=0, borderpad=0.3)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        scatter_args = dict(alpha=0.5, s=20)
        x_label = 'Run duration'
        y_label = 'Run speed'
        legend_args = {}

    if show_heatmap:
        x_min, x_max = min(durations), max(durations)
        y_min, y_max = min(speeds), max(speeds)
        Z = _make_surface(durations, speeds, x_min, x_max, y_min, y_max, 1000)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')

    if show_real:
        ax.scatter(durations, speeds, marker='o', label='Data', **scatter_args)
    if show_synth:
        ax.scatter(synth[0], synth[1], marker='x', label='Synthetic', **scatter_args)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(**legend_args)
    if layout == 'default':
        fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'runs_scatter.svg', transparent=layout != 'default')
    if show_plots:
        plt.show()


def _fit_dual_cdf_model(
        data: np.ndarray,
        n_angle_bins: int = 5,
):
    # Put all data in the top right quadrant
    data = np.abs(data)

    # Convert data to polar coordinates
    r = np.linalg.norm(data, axis=1)
    psi = np.arctan2(data[:, 1], data[:, 0])
    assert np.all(psi >= 0) and np.all(psi <= np.pi / 2), 'psi must be between 0 and pi/2!'
    data_p = np.array([r, psi]).T

    # Split the data into wedges based on the angles
    angle_bins = np.linspace(0, np.pi / 2, n_angle_bins + 1)
    psi_bin_idxs = np.digitize(psi, angle_bins)
    psi_bin_idxs -= 1
    data_slices_by_psi = [data_p[psi_bin_idxs == i] for i in range(n_angle_bins)]

    def make_cdf(vals):
        vals = np.sort(vals)
        k = len(vals)
        if k == 0:
            vals = np.array([0, 1e-4])
        elif k == 1:
            vals = np.concatenate(([np.min(vals) * 0.9], vals))
        k = len(vals)
        y = np.arange(0, k) / (k - 1)
        interp = PchipInterpolator(vals, y, extrapolate=False)
        interp_inv = PchipInterpolator(y, vals, extrapolate=False)

        def interp_func(x, inverse=False):
            if inverse:
                return interp_inv(x)
            y = interp(x)
            y = np.where(x < vals[0], 0, np.where(x > vals[-1], 1, y))
            return y

        return interp_func

    # Calculate the psi distribution for all r values
    cdfs_psi_all = make_cdf(data_p[:, 1])

    # For each of the psi data slices, make a cdf for the distribution of r values
    cdfs_r = []
    for i, data_slice in enumerate(data_slices_by_psi):
        cdfs_r.append(make_cdf(data_slice[:, 0]))

    def sample_fake_data(n_fake_samples):
        # Sample some new data
        z1 = np.random.rand(n_fake_samples)
        z2 = np.random.rand(n_fake_samples)
        psi_vals = cdfs_psi_all(z1, inverse=True)
        r_vals_all = np.zeros((n_angle_bins, n_fake_samples))
        for i, cdf in enumerate(cdfs_r):
            r_vals_all[i] = cdf(z2, inverse=True)
        pbi = np.digitize(psi_vals, angle_bins) - 1

        # Interpolate the r values across neighbouring bins depending on their position in the bin
        psi_bin_positions = (psi_vals - angle_bins[pbi]) / (angle_bins[pbi + 1] - angle_bins[pbi])
        pos_idx = np.arange(n_fake_samples)
        r_vals_left = r_vals_all[np.clip(pbi - 1, a_min=0, a_max=None), pos_idx]
        r_vals_middle = r_vals_all[pbi, pos_idx]
        r_vals_right = r_vals_all[np.clip(pbi + 1, a_min=0, a_max=n_angle_bins - 1), pos_idx]
        r_vals = (r_vals_middle * (1 - np.abs(psi_bin_positions - 0.5))
                  + r_vals_left * np.clip(0.5 - psi_bin_positions, a_min=0, a_max=None)
                  + r_vals_right * np.clip(psi_bin_positions - 0.5, a_min=0, a_max=None))

        # Turn the data back into cartesian
        fake_data = np.array([r_vals * np.cos(psi_vals), r_vals * np.sin(psi_vals)]).T

        # Randomly flip the x and y values
        flip_x = np.random.rand(n_fake_samples) > 0.5
        flip_y = np.random.rand(n_fake_samples) > 0.5
        fake_data[:, 0] *= np.where(flip_x, -1, 1)
        fake_data[:, 1] *= np.where(flip_y, -1, 1)

        return fake_data

    return data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data


def _generate_noisy_donut_data(n_samples, radius, noise_std):
    psi = np.linspace(0, 2 * np.pi, n_samples)
    data = np.array([radius * np.cos(psi), radius * np.sin(psi)]).T
    data += np.random.normal(0, noise_std, size=(n_samples, 2))
    return data


def donut_test():
    """
    Try to fit a synthetic donut distribution.
    """
    n_data_samples = 200
    n_fake_samples = 300
    radius = 0.5
    noise_std = 0.1
    n_angle_bins = 5

    # Generate donut-distributed data
    data = _generate_noisy_donut_data(n_data_samples, radius, noise_std)

    # Fit the model
    data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data = _fit_dual_cdf_model(data, n_angle_bins)
    max_psi = np.max(data_p[:, 1])
    max_r = np.max(data_p[:, 0])

    # Sample some fake data
    fake_data = sample_fake_data(n_fake_samples)

    # Plot the data, distribution functions and sampled data
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax = axes[0, 0]
    ax.set_title('All data')
    probs = cdfs_psi_all(data_p[:, 1])
    ax.scatter(*data.T, c=probs, cmap='viridis')
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\phi$')

    ax = axes[1, 0]
    x = np.linspace(0, max_psi * 1.01, 1000)
    y = cdfs_psi_all(x)
    y[np.isnan(y)] = 1
    cmap = plt.get_cmap('viridis')
    for i in range(len(x)):
        ax.plot(x[i:i + 2], y[i:i + 2], c=cmap(y[i]))
    ax.set_xticks([0, np.pi / 4, np.pi / 2])
    ax.set_xticklabels(['0', '$\pi/4$', '$\pi/2$'])
    ax.set_xlabel('$\psi$')
    ax.set_ylabel('Probability')
    ax.grid(True)

    ax = axes[0, 1]
    ax.set_title('Split by $\psi$')
    for i in range(n_angle_bins):
        data_i = data[psi_bin_idxs == i]
        ax.scatter(*data_i.T)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\phi$')

    ax = axes[1, 1]
    x = np.linspace(0, max_r * 1.01, 1000)
    for i, cdf in enumerate(cdfs_r):
        y = cdf(x)
        y[np.isnan(y)] = 1
        ax.plot(x, y,
                label=f'$\\psi \in \lbrace{angle_bins[i] / np.pi:.2f}\pi,{angle_bins[i + 1] / np.pi:.2f}\pi\\rbrace$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    ax.set_title('Sampled data')
    ax.scatter(*fake_data.T)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\phi$')

    axes[1, 2].axis('off')

    fig.tight_layout()
    plt.show()


def model_tumbles(show_heatmap: bool = False):
    """
    Build a probability distribution for the tumbles based on planar and non-planar angles.
    """
    n_fake_samples = 1000
    n_angle_bins = 8
    save_dir, data_values = _init()
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']
    data = np.array([thetas, phis]).T

    # Fit the model
    data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data = _fit_dual_cdf_model(data, n_angle_bins)
    max_psi = np.max(data_p[:, 1])
    max_r = np.max(data_p[:, 0])

    # Sample some fake data
    fake_data = sample_fake_data(n_fake_samples)

    # Plot the data, distribution functions and sampled data
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def setup_angle_axes(ax_):
        ax_.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
        ax_.set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax_.set_yticklabels(['$-\pi/2$', '0', '$\pi/2$'])
        ax_.set_xlabel('$\\theta$')
        ax_.set_ylabel('$\\phi$')

    ax = axes[0, 0]
    ax.set_title('All data')
    probs = cdfs_psi_all(data_p[:, 1])
    if show_heatmap:
        x_min, x_max = min(thetas), max(thetas)
        y_min, y_max = min(phis), max(phis)
        Z = _make_surface(thetas, phis, x_min, x_max, y_min, y_max, 1000)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')
    ax.scatter(*data.T, c=probs, cmap='viridis', s=5)
    setup_angle_axes(ax)

    ax = axes[1, 0]
    x = np.linspace(0, max_psi * 1.01, 1000)
    y = cdfs_psi_all(x)
    y[np.isnan(y)] = 1
    cmap = plt.get_cmap('viridis')
    for i in range(len(x)):
        ax.plot(x[i:i + 2], y[i:i + 2], c=cmap(y[i]))
    ax.set_xticks([0, np.pi / 4, np.pi / 2])
    ax.set_xticklabels(['0', '$\pi/4$', '$\pi/2$'])
    ax.set_xlabel('$\psi$')
    ax.set_ylabel('cdf')
    ax.grid(True)

    ax = axes[0, 1]
    ax.set_title('Split by $\psi$')
    for i in range(n_angle_bins):
        data_i = data[psi_bin_idxs == i]
        ax.scatter(*data_i.T, s=10)
    setup_angle_axes(ax)

    ax = axes[1, 1]
    x = np.linspace(0, max_r * 1.01, 1000)
    for i, cdf in enumerate(cdfs_r):
        y = cdf(x)
        y[np.isnan(y)] = 1
        ax.plot(x, y,
                label=f'$\\psi \in \lbrace{angle_bins[i] / np.pi:.2f}\pi,{angle_bins[i + 1] / np.pi:.2f}\pi\\rbrace$')
    ax.set_xlabel('$r$')
    ax.set_ylabel('cdf')
    ax.grid(True)

    # Legend - show in empty axis
    legend = ax.legend()
    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    axes[1, 2].legend(handles, labels, loc='upper left', bbox_to_anchor=(-0.15, 1),
                      bbox_transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    legend.remove()

    ax = axes[0, 2]
    ax.set_title('Sampled data')
    if show_heatmap:
        fake_thetas, fake_phis = fake_data.T
        x_min, x_max = min(fake_thetas), max(fake_thetas)
        y_min, y_max = min(fake_phis), max(fake_phis)
        Z = _make_surface(fake_thetas, fake_phis, x_min, x_max, y_min, y_max, 1000)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')
    ax.scatter(*fake_data.T, s=2)
    setup_angle_axes(ax)

    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'tumbles_dual_cdf_model.svg')
    if show_plots:
        plt.show()


def model_tumbles_basic(
        show_heatmap: bool = False,
        show_real: bool = True,
        show_synth: bool = True,
        layout: str = 'default',
):
    """
    Build a probability distribution for the tumbles based on planar and non-planar angles.
    """
    n_fake_samples = 2000
    n_angle_bins = 20
    save_dir, data_values = _init()
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']
    data = np.array([thetas, phis]).T

    # Fit the model
    data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data = _fit_dual_cdf_model(data, n_angle_bins)

    # Sample some fake data
    fake_data = sample_fake_data(n_fake_samples)

    # Plot the results
    if layout == 'paper':
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
        plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
        plt.rc('axes', labelsize=6, labelpad=0)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=6)  # fontsize of the legend
        plt.rc('ytick.major', pad=1, size=2)
        plt.rc('xtick.major', pad=1, size=2)
        plt.rc('xtick.minor', size=1)
        fig, ax = plt.subplots(1, 1, figsize=(1.39, 0.8), gridspec_kw={
            'top': 0.98,
            'bottom': 0.22,
            'left': 0.18,
            'right': 0.98,
        })
        scatter_args = dict(alpha=0.3, s=0.1)
        legend_args = dict(loc='upper right', markerscale=10, handlelength=1, handletextpad=0.2,
                           labelspacing=0, borderpad=0.3)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        scatter_args = dict(alpha=0.5, s=20)
        legend_args = {}

    def setup_angle_axes(ax_):
        ax_.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_.set_xticklabels(['$-\pi$', '$-\pi/2$', '', '$\pi/2$', '$\pi$'])
        ax_.set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax_.set_yticklabels(['$-\pi/2$', '', '$\pi/2$'])
        # ax_.set_xlabel('$\\theta$', labelpad=-5)
        # ax_.set_ylabel('$\\phi$', labelpad=-8)
        ax_.set_xlabel('In-plane angle')
        ax_.set_ylabel('Out-of-plane angle')
        ax_.set_xlim(-np.pi, np.pi)
        ax_.set_ylim(-np.pi / 2, np.pi / 2)

    if show_heatmap:
        fake_thetas, fake_phis = fake_data.T
        x_min, x_max = -np.pi, np.pi
        y_min, y_max = -np.pi / 2, np.pi / 2
        Z = _make_surface(fake_thetas, fake_phis, x_min, x_max, y_min, y_max, 1000, bw_method=0.3)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')

    if show_real:
        ax.scatter(thetas, phis, marker='o', label='Data', **scatter_args)
    if show_synth:
        ax.scatter(*fake_data.T, marker='x', label='Synthetic', **scatter_args)

    setup_angle_axes(ax)
    ax.legend(**legend_args)
    if layout == 'default':
        fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'tumbles_scatter.svg', transparent=layout != 'default')
    if show_plots:
        plt.show()


def plot_phi_factors(
        layout: str = 'default',
):
    """
    Build a probability distribution for the tumbles based on planar and non-planar angles.
    """
    n_fake_samples = 2000
    n_angle_bins = 20
    save_dir, data_values = _init()
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']
    data = np.array([thetas, phis]).T

    # Fit the model
    data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data = _fit_dual_cdf_model(data, n_angle_bins)

    # Sample some fake data
    fake_data = sample_fake_data(n_fake_samples)

    # Set up plot parameters
    if layout == 'paper':
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('ytick.major', pad=1, size=2)
        plt.rc('xtick.major', pad=1, size=2)
        plt.rc('xtick.minor', size=1)

    def setup_angle_axes(ax_):
        ax_.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_.set_xticklabels(['$-\pi$', '', '', '', '$\pi$'])
        ax_.set_yticks([-np.pi / 2, 0, np.pi / 2])
        ax_.set_yticklabels(['$-\pi/2$', '', '$\pi/2$'])
        ax_.set_xlim(-np.pi, np.pi)
        ax_.set_ylim(-np.pi / 2, np.pi / 2)

    fake_thetas, fake_phis = fake_data.T
    x_min, x_max = -np.pi, np.pi
    y_min, y_max = -np.pi / 2, np.pi / 2
    phi_factors = [0.2, 1, 5]

    for factor in phi_factors:
        logger.info(f'Plotting phi factor: {factor}')
        fake_phis_adj = fake_phis * factor
        Z = _make_surface(fake_thetas, fake_phis_adj, x_min, x_max, y_min, y_max, 100, bw_method=0.3)

        # Plot the results
        if layout == 'paper':
            fig, ax = plt.subplots(1, 1, figsize=(0.6, 0.45), gridspec_kw={
                'top': 0.96,
                'bottom': 0.22,
                'left': 0.24,
                'right': 0.95,
            })
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')
        setup_angle_axes(ax)
        if layout == 'default':
            fig.tight_layout()
        if save_plots:
            plt.savefig(save_dir / f'phi_factor={factor}.svg', transparent=layout != 'default')
        if show_plots:
            plt.show()


def tumble_heatmaps():
    """
    Plot heatmaps from the data and the modelled pdfs.
    """
    save_dir, data_values = _init()
    n_samples = 2000
    n_angle_bins = 8

    # Fit the model
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']
    data = np.array([thetas, phis]).T
    data_p, angle_bins, psi_bin_idxs, cdfs_psi_all, cdfs_r, sample_fake_data = _fit_dual_cdf_model(data, n_angle_bins)

    # Sample from the model
    thetas_synth, phis_synth = sample_fake_data(n_samples).T

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    x_min, x_max = -np.pi, np.pi
    y_min, y_max = -np.pi / 2, np.pi / 2
    Z_data = _make_surface(thetas, phis, x_min, x_max, y_min, y_max, 1000)
    Z_synth = _make_surface(thetas_synth, phis_synth, x_min, x_max, y_min, y_max, 1000)
    for i, (Z, title) in enumerate(zip([Z_data, Z_synth], ['Data', 'Synthetic'])):
        ax = axes[i]
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max], aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('$\\theta$')
        ax.set_ylabel('$\phi$')
    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'tumbles_heatmaps.svg')
    if show_plots:
        plt.show()


def plot_correlations():
    """
    Check for correlations between the run durations/speeds and the angles before and after each run.
    """
    save_dir, data_values = _init(include_all_runs=True)
    durations = data_values['durations']
    speeds = data_values['speeds']
    thetas = data_values['planar_angles']
    phis = data_values['nonplanar_angles']
    tumble_idxs = data_values['tumble_idxs']
    thetas_pre = []
    thetas_post = []
    phis_pre = []
    phis_post = []
    idx = 0
    for i in range(len(tumble_idxs)):
        n_tumbles = len(tumble_idxs[i])
        if n_tumbles == 0:
            continue
        thetas_i = thetas[idx:idx + n_tumbles]
        phis_i = phis[idx:idx + n_tumbles]
        thetas_pre.extend(thetas_i[:-1])
        thetas_post.extend(thetas_i[1:])
        phis_pre.extend(phis_i[:-1])
        phis_post.extend(phis_i[1:])
        idx += n_tumbles
    thetas_pre = np.array(thetas_pre)
    thetas_post = np.array(thetas_post)
    phis_pre = np.array(phis_pre)
    phis_post = np.array(phis_post)

    # Plot the angles against the durations and speeds to check for correlations
    scatter_args = dict(alpha=0.5, s=20, marker='o')
    fig, axes = plt.subplots(2, 4, figsize=(14, 10))

    for i, (x, x_lbl) in enumerate(zip([durations, speeds], ['Duration', 'Speed'])):
        for j, (y, y_lbl) in enumerate(zip([phis_pre, phis_post, thetas_pre, thetas_post],
                                           ['Phis pre-run', 'Phis post-run', 'Thetas pre-run', 'Thetas post-run'])):
            y = np.abs(y)
            res = pearsonr(x, y)
            ax = axes[i, j]
            ax.set_title(f'R={res[0]:.2E}, p={res[1]:.2E}')
            ax.scatter(x, y, **scatter_args)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)

    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'correlations.svg')
    if show_plots:
        plt.show()


def plot_pause_penalty(
        layout: str = 'default',
):
    """
    Plot the pause penalty relationship.
    """
    save_dir, data_values, args = _init(include_all_runs=True, return_args=True)

    # Fit a single gaussian to the nonplanar angles
    phis = data_values['nonplanar_angles']

    mu, std = norm.fit(phis)
    dist = norm(loc=mu, scale=std)

    # Plot the results
    if layout == 'paper':
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=6)  # fontsize of the legend
        plt.rc('ytick.major', pad=1, size=2)
        plt.rc('xtick.major', pad=1, size=2)
        plt.rc('xtick.minor', size=1)
        fig, ax = plt.subplots(1, 1, figsize=(1.37, 0.62), gridspec_kw={
            'top': 0.98,
            'bottom': 0.18,
            'left': 0.24,
            'right': 0.73,
        })
        legend_args = dict(handlelength=0.5, handletextpad=0.2,
                           labelspacing=0, borderpad=0.3)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        legend_args = {}

    N = 501
    vals = np.zeros(N)
    for i in range(-9, 8, 2):
        x = np.linspace(i * np.pi, (i + 2) * np.pi, N)
        vals += dist.pdf(x)

    x = np.linspace(-np.pi, np.pi, N)
    ax.plot(x, vals, alpha=0.9, color='#9b628dff', zorder=1, label='$\phi$')

    p_samples = dist.rvs(2000).squeeze()
    p_samples = np.arctan(np.tan(p_samples))
    ax.hist(p_samples, bins=25, density=True, facecolor='#fbe6fbff', alpha=1.)

    ax.set_xlim(left=-np.pi / 2 - 0.1, right=np.pi / 2 + 0.1)
    ax.set_xticks([-np.pi / 2, np.pi / 2])
    ax.set_xticklabels(['$-\pi/2$', '$\pi/2$'])
    ax.set_yticks([0.5, ])
    ax.legend(**legend_args, loc='center right', bbox_to_anchor=(-0.15, 0.5),
              bbox_transform=ax.transAxes)

    # Pauses
    phis = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    if args.nonp_pause_type == 'linear':
        pauses = (np.abs(phis) / (np.pi / 2)) * args.nonp_pause_max
    elif args.nonp_pause_type == 'quadratic':
        pauses = (np.abs(phis) / (np.pi / 2))**2 * args.nonp_pause_max
    else:
        raise RuntimeError(f'Unsupported pause type: {args.nonp_pause_type}.')
    ax2 = ax.twinx()
    ax2.plot(phis, pauses, label='$\delta(\phi)$', color='#e91616ff')
    ax2.set_ylim(bottom=0, top=args.nonp_pause_max + 0.1)
    ax2.set_yticks([args.nonp_pause_max])
    ax2.set_yticklabels([args.nonp_pause_max])
    ax2.legend(**legend_args, loc='center left', bbox_to_anchor=(1.05, 0.5),
               bbox_transform=ax2.transAxes)

    if layout == 'default':
        fig.tight_layout()

    if layout == 'default':
        fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'pause_penalty.svg', transparent=layout != 'default')
    if show_plots:
        plt.show()


def plot_simulated_trajectories(plot_n_examples=5):
    """
    Plot some simulation trajectories.
    """
    save_dir, data_values, args = _init(include_all_runs=True, return_args=True)
    SS = get_sim_state_from_args(args)

    # Construct colours
    colours = np.linspace(0, 1, SS.X.shape[1])
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]
    plot_idxs = range(min(SS.parameters.batch_size, plot_n_examples))
    plot_idxs = [2, 16, 23, 28, 48]

    if args.approx_noise is not None and args.approx_noise > 0:
        logger.info(f'Will be adding noise to the trajectories with std={args.approx_noise}.')
    if args.smoothing_window is not None and args.smoothing_window > 0:
        logger.info(f'Will be smoothing the trajectories with a window of {args.smoothing_window} frames.')

    for idx in plot_idxs:
        logger.info(f'Plotting sim run {idx}.')
        X = SS.X[idx].copy().astype(np.float64)

        # Add some noise to the trajectory then smooth
        if args.approx_noise is not None and args.approx_noise > 0:
            X = X + np.random.normal(np.zeros_like(X), args.approx_noise)
        if args.smoothing_window is not None and args.smoothing_window > 0:
            X = smooth_trajectory(X, window_len=args.smoothing_window)

        # Plot the trajectory
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', azim=105, elev=-80)
        ax.scatter(*X.T, c=c, s=100, alpha=1, zorder=1)
        equal_aspect_ratio(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis('off')
        fig.tight_layout()

        if save_plots:
            plt.savefig(save_dir / f'ss_{SS.parameters.id}_sim_{idx}.png',
                        transparent=True)
            np.savez(save_dir / f'ss_{SS.parameters.id}_sim_{idx}', X=X)

        if show_plots:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    # model_runs(show_heatmap=True)
    # model_runs(show_heatmap=True, show_synth=False, layout='paper')
    # donut_test()
    # model_tumbles(show_heatmap=True)
    model_tumbles_basic(show_heatmap=True, show_synth=False, layout='paper')
    # plot_phi_factors(layout='paper')
    # tumble_heatmaps()
    # plot_correlations()
    # plot_pause_penalty(layout='paper')
    # plot_simulated_trajectories(plot_n_examples=50)
