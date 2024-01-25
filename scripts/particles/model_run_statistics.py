import os
import shutil
from argparse import Namespace

import numpy as np
import yaml
from copulas.multivariate import GaussianMultivariate
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Dataset
from wormlab3d.particles.tumble_run import generate_or_load_ds_statistics
from wormlab3d.toolkit.util import hash_data, print_args, to_dict
from wormlab3d.trajectories.args import get_args

show_plots = True
# save_plots = False
# show_plots = False
save_plots = True
interactive_plots = False
img_extension = 'svg'

DATA_CACHE_PATH = LOGS_PATH / 'cache'
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)

DIST_KEYS = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']


def _identifiers(args: Namespace) -> str:
    id_str = (f'_ds={args.dataset}'
              f'_e={args.approx_error_limit}'
              f'_pw={args.planarity_window_vertices}'
              f'_df={args.approx_distance}'
              f'_hf={args.approx_curvature_height}'
              f'_sm={args.smoothing_window_K}')
    return id_str


def model_runs():
    """
    Build a bivariate probability distribution for the runs based on speeds and durations.
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
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{hash_data(to_dict(args))}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments
    if (LOGS_PATH / 'spec.yml').exists():
        shutil.copy(LOGS_PATH / 'spec.yml', save_dir / 'spec.yml')
    with open(save_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    approx_args = dict(
        approx_method=args.approx_method,
        error_limits=[args.approx_error_limit],
        planarity_window=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.smoothing_window_K,
        smooth_K_first=args.smoothing_window_K,
        min_run_speed_duration=(0, 10000)
    )

    # Fetch dataset
    ds = Dataset.objects.get(id=args.dataset)

    # Generate or load tumble/run values
    ds_stats = generate_or_load_ds_statistics(
        ds=ds,
        rebuild_cache=args.regenerate,
        **approx_args
    )
    # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles
    data_values = {k: ds_stats[i + 1] for i, k in enumerate(DIST_KEYS)}
    durations = data_values['durations'][0]
    speeds = data_values['speeds'][0]

    # Get the angles before and after each run
    thetas = data_values['planar_angles'][0]
    phis = data_values['nonplanar_angles'][0]
    tumble_idxs = ds_stats[6]['0']
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

    # Fit a copula to the durations and speeds
    data = np.column_stack((durations, speeds))
    copula = GaussianMultivariate()
    copula.fit(data)

    # Generate synthetic samples from the joint distribution
    synth = copula.sample(len(data))

    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scatter_args = dict(alpha=0.5, s=20)
    ax.scatter(durations, speeds, marker='o', label='Real', **scatter_args)
    ax.scatter(synth[0], synth[1], marker='x', label='Synthetic', **scatter_args)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('Run duration')
    ax.set_ylabel('Run speed')
    ax.legend()
    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'scatter.png')
    if show_plots:
        plt.show()

    # Plot the angles against the durations and speeds to check for correlations
    scatter_args['marker'] = 'o'
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
        plt.savefig(save_dir / 'correlations.png')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()
    model_runs()
