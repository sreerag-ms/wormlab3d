import json
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_1samp

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction, Trial
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

CameraConfig = Dict[int, Tuple[str, str]]  # Camera configuration mapping camera index to x/y image axes

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

REFERENCE_CONFIG = {0: ('x', 'y'), 1: ('x', '-z'), 2: ('z', 'y')}

show_plots = True
save_plots = True
img_extension = 'png'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to check isotropy.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}'


def create_transformation_matrix(
        cam_config0: CameraConfig,
        cam_config1: CameraConfig
) -> np.ndarray:
    """
    Create a transformation matrix that maps cam_config1 to cam_config0.

    Parameters:
    - cam_config0: Dictionary mapping camera axes for config 0.
    - cam_config1: Dictionary mapping camera axes for config 1.

    Returns:
    - transformation_matrix: A 3x3 matrix that aligns cam_config1 with cam_config0.
    """

    # Define a helper to convert axis strings to index and sign
    def axis_to_index_sign(axis):
        if axis == 'x':
            return 0, 1
        elif axis == '-x':
            return 0, -1
        elif axis == 'y':
            return 1, 1
        elif axis == '-y':
            return 1, -1
        elif axis == 'z':
            return 2, 1
        elif axis == '-z':
            return 2, -1
        raise ValueError(f'Unknown axis: {axis}')

    # Initialise the transformation matrix
    T = np.zeros((3, 3))

    # Loop through the axes in cam_config0 and find the corresponding axes in cam_config1
    for i, (axis0_1, axis0_2) in cam_config0.items():
        axis1_1, axis1_2 = cam_config1[i]

        # Map the first axis (e.g., 'x' in cam_config0) to its corresponding axis in cam_config1
        idx0_1, sign0_1 = axis_to_index_sign(axis0_1)
        idx1_1, sign1_1 = axis_to_index_sign(axis1_1)

        # Map the second axis (e.g., 'y' in cam_config0) to its corresponding axis in cam_config1
        idx0_2, sign0_2 = axis_to_index_sign(axis0_2)
        idx1_2, sign1_2 = axis_to_index_sign(axis1_2)

        # Set the matrix elements to align cam_config1 to cam_config0
        T[idx0_1, idx1_1] = sign0_1 * sign1_1
        T[idx0_2, idx1_2] = sign0_2 * sign1_2

    return T


def _calculate_camera_configuration_from_points(
        cam0_points: np.ndarray,
        cam1_points: np.ndarray,
        cam2_points: np.ndarray
) -> CameraConfig:
    """
    Calculate the camera configuration from the projected 2D points.
    """
    assert cam0_points.shape == (3, 2), f'Wrong shape for cam0_points: {cam0_points.shape}'
    assert cam1_points.shape == (3, 2), f'Wrong shape for cam1_points: {cam1_points.shape}'
    assert cam2_points.shape == (3, 2), f'Wrong shape for cam2_points: {cam2_points.shape}'

    # Determine what the image x/y axes are in the world frame
    directions = ['x', 'y', 'z', '-x', '-y', '-z']
    cam_config = {}
    for cam_idx, cam_points in enumerate([cam0_points, cam1_points, cam2_points]):
        rtol = 0.1
        n_attempts = 0
        while n_attempts < 10:
            threshold = np.abs(cam_points).max() * rtol
            pos_axes = cam_points > threshold
            neg_axes = cam_points < -threshold
            axes_directions = np.concatenate([pos_axes, neg_axes])
            if np.allclose(axes_directions.sum(axis=0), np.array([1, 1])):
                cam_config[cam_idx] = (
                    directions[np.where(axes_directions[:, 0])[0][0]],
                    directions[np.where(axes_directions[:, 1])[0][0]]
                )
                break
            rtol *= 1.5
            n_attempts += 1

        if cam_idx not in cam_config:
            raise RuntimeError(f'Could not determine axes for camera {cam_idx}!')

    return cam_config


def _calculate_camera_configuration_from_legacy_cameras(
        trial: Trial, X0: np.ndarray,
        transformation: np.ndarray = None
) -> CameraConfig:
    """
    Calculate the camera configuration using the original camera models.
    Optionally apply a transformation to the 3D test points before projecting.
    """
    assert X0.shape == (3,), f'Wrong shape for X0: {X0.shape}'

    # Move the 3D point in the world x, y and z directions
    dx = np.array([1, 0, 0])
    dy = np.array([0, 1, 0])
    dz = np.array([0, 0, 1])

    # Apply the transformation to the test directions if provided
    if transformation is not None:
        dx = dx @ transformation.T
        dy = dy @ transformation.T
        dz = dz @ transformation.T

    # Calculate the new 3D points
    Xx = X0 + dx
    Xy = X0 + dy
    Xz = X0 + dz

    # Iterate over available cameras - the first camera should be the "best", but if it fails, try the others
    all_cams = trial.get_cameras(best=False)
    cam_config = None
    for cameras in all_cams:
        cams = cameras.get_camera_model_triplet()
        try:
            # Project the initial 3D trajectory point onto the 3 image planes
            ref0, ref1, ref2 = cams.project_to_2d(X0)[0]

            # Project the new points onto the 3 image planes
            x0, x1, x2 = cams.project_to_2d(Xx)[0]
            y0, y1, y2 = cams.project_to_2d(Xy)[0]
            z0, z1, z2 = cams.project_to_2d(Xz)[0]

            # Subtract the reference image points to get the 2D positions moved in each image
            cam0_points = np.array([x0 - ref0, y0 - ref0, z0 - ref0])
            cam1_points = np.array([x1 - ref1, y1 - ref1, z1 - ref1])
            cam2_points = np.array([x2 - ref2, y2 - ref2, z2 - ref2])

            # Determine camera configuration
            cam_config = _calculate_camera_configuration_from_points(cam0_points, cam1_points, cam2_points)

            # If the configuration is valid, return it
            break
        except RuntimeError:
            continue
    if cam_config is None:
        raise RuntimeError(f'Could not calculate camera configuration for trial {trial.id}!')

    return cam_config


def _calculate_camera_configuration_from_mf_cameras(
        reconstruction: Reconstruction,
        transformation: np.ndarray = None
) -> CameraConfig:
    """
    Calculate the camera configuration using the Midline-Finder cameras.
    Optionally apply a transformation to the 3D test points before projecting.
    """
    ts = TrialState(reconstruction)
    points_3d = ts.get('points')
    points_3d_base = ts.get('points_3d_base')
    points_2d_base = ts.get('points_2d_base')
    cam_coeffs = np.concatenate([
        ts.get(f'cam_{k}')
        for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
    ], axis=2)
    prs = ProjectRenderScoreModel(image_size=reconstruction.trial.crop_size)
    frame_num = reconstruction.start_frame_valid + 1
    X0 = points_3d[frame_num].mean(axis=0)

    # Move the 3D point in the world x, y and z directions
    dx = np.array([1, 0, 0])
    dy = np.array([0, 1, 0])
    dz = np.array([0, 0, 1])

    # Apply the transformation to the test directions if provided
    if transformation is not None:
        dx = dx @ transformation.T
        dy = dy @ transformation.T
        dz = dz @ transformation.T

    # Calculate the new 3D points
    Xx = X0 + dx
    Xy = X0 + dy
    Xz = X0 + dz
    test_points = np.stack([X0, Xx, Xy, Xz])[None, ...]

    # Project the new points onto the 3 image planes
    points_2d = prs._project_to_2d(
        cam_coeffs=torch.from_numpy(cam_coeffs[frame_num][None, ...]),
        points_3d=torch.from_numpy(test_points.astype(np.float32)),
        points_3d_base=torch.from_numpy(points_3d_base[frame_num][None, ...].astype(np.float32)),
        points_2d_base=torch.from_numpy(points_2d_base[frame_num][None, ...].astype(np.float32)),
        clamp=False
    ).squeeze().numpy().transpose(1, 0, 2)

    # Subtract the reference image points to get the 2D positions moved in each image
    ref0, ref1, ref2 = points_2d[0]
    x0, x1, x2 = points_2d[1]
    y0, y1, y2 = points_2d[2]
    z0, z1, z2 = points_2d[3]
    cam0_points = np.array([x0 - ref0, y0 - ref0, z0 - ref0])
    cam1_points = np.array([x1 - ref1, y1 - ref1, z1 - ref1])
    cam2_points = np.array([x2 - ref2, y2 - ref2, z2 - ref2])

    # Determine what the image x/y axes are in the world frame
    cam_config = _calculate_camera_configuration_from_points(cam0_points, cam1_points, cam2_points)

    return cam_config


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, List[CameraConfig]], Dict[float, np.ndarray]]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')

    # Use the reconstruction from the dataset where possible
    reconstructions = {}
    for r_ref in ds.reconstructions:
        r = Reconstruction.objects.get(id=r_ref.id)
        reconstructions[r.trial.id] = r

    # Calculate the isotropy for all trials
    end_points = {}
    end_points_transformed = {}
    cam_configs = {}
    transformations = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating isotropy for trial={trial.id}.')

        # Get the camera configuration using the midline-finder reconstruction data and cameras
        if trial.id in reconstructions:
            X, _ = get_trajectory(
                reconstruction_id=reconstructions[trial.id].id,
                trajectory_point=args.trajectory_point,
                smoothing_window=args.smoothing_window
            )
            X = X.squeeze()
            assert X.ndim == 2, f'Wrong shape trajectory: {X.shape}'
            cam_config = _calculate_camera_configuration_from_mf_cameras(reconstructions[trial.id])

            # Check the config matches with the trial cameras if available
            try:
                cam_config_legacy = _calculate_camera_configuration_from_legacy_cameras(trial, X[0])
            except RuntimeError:
                cam_config_legacy = None
            if cam_config_legacy is not None:
                assert cam_config == cam_config_legacy, f'Camera configurations do not match: {cam_config} vs {cam_config_legacy}'

        # Get the camera configuration using the original cameras and basic tracking data
        else:
            X, _ = get_trajectory(
                trial_id=trial.id,
                tracking_only=True,
                smoothing_window=args.smoothing_window
            )
            X = X.squeeze()
            assert X.ndim == 2, f'Wrong shape trajectory: {X.shape}'
            cam_config = _calculate_camera_configuration_from_legacy_cameras(trial, X[0])

        # Calculate the transformation needed to align the camera configuration with the reference configuration
        T = create_transformation_matrix(REFERENCE_CONFIG, cam_config)

        # If the camera configuration is different, transform the trajectory to match
        if cam_config != REFERENCE_CONFIG:
            assert not np.allclose(T, np.eye(3)), \
                f'Transformation matrix should not be identity as the configurations are different! Cam config: {cam_config} T: {T}'

            # Validate that the transformation really does align the camera configurations
            if trial.id in reconstructions:
                cam_config_T = _calculate_camera_configuration_from_mf_cameras(reconstructions[trial.id],
                                                                               transformation=T)
            else:
                cam_config_T = _calculate_camera_configuration_from_legacy_cameras(trial, X[0], transformation=T)
            assert cam_config_T == REFERENCE_CONFIG, \
                f'Camera configurations do not match reference configuration after transformation: {cam_config}'

            # Apply the transformation to the trajectory
            Xt = X @ T.T

        # Otherwise, the trajectory is already in the correct frame
        else:
            assert np.allclose(T, np.eye(3)), \
                f'Transformation matrix should be identity as the configurations are the same! Cam config: {cam_config} T: {T}'
            Xt = X

        # Record the camera setup and end points
        c = trial.experiment.concentration
        if c not in end_points:
            end_points[c] = []
            end_points_transformed[c] = []
            cam_configs[c] = []
            transformations[c] = []
        end_points[c].append(X[-1] - X[0])
        end_points_transformed[c].append(Xt[-1] - Xt[0])
        cam_configs[c].append(cam_config)
        transformations[c].append(T)

    # Sort by concentration
    end_points = {c: np.array(v) for c, v in sorted(list(end_points.items()))}
    end_points_transformed = {c: np.array(v) for c, v in sorted(list(end_points_transformed.items()))}
    cam_configs = {c: v for c, v in sorted(list(cam_configs.items()))}
    transformations = {c: np.array(v) for c, v in sorted(list(transformations.items()))}

    return end_points, end_points_transformed, cam_configs, transformations


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray], Dict[float, List[CameraConfig]], Dict[float, np.ndarray]]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    cache_fn_cc = cache_path.with_suffix(cache_path.suffix + '.json')
    end_points, end_points_transformed, cam_configs, transformations = None, None, None, None
    if not rebuild_cache and cache_fn.exists():
        try:
            # Load data
            data = np.load(cache_fn)
            concs = data['concs']
            end_points, end_points_transformed, transformations = {}, {}, {}
            for c in concs:
                end_points[float(c)] = data[f'end_points_{c}']
                end_points_transformed[float(c)] = data[f'end_points_transformed_{c}']
                transformations[float(c)] = data[f'transformations_{c}']

            # Load camera configs
            with open(cache_fn_cc, 'r') as f:
                cam_configs = json.load(f)

            logger.info(f'Loaded data from cache: {cache_fn} and {cache_fn_cc}.')
        except Exception as e:
            end_points, end_points_transformed, cam_configs, transformations = None, None, None, None
            logger.warning(f'Could not load cache: {e}')

    if end_points is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        end_points, end_points_transformed, cam_configs, transformations = _calculate_data(args, ds)
        data = {'concs': np.array([f'{c:.2f}' for c in end_points.keys()])}
        for c in end_points.keys():
            data[f'end_points_{c:.2f}'] = end_points[c]
            data[f'end_points_transformed_{c:.2f}'] = end_points_transformed[c]
            data[f'transformations_{c:.2f}'] = transformations[c]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)
        logger.info(f'Saving camera configs to {cache_fn_cc}.')
        with open(cache_fn_cc, 'w') as f:
            json.dump(cam_configs, f)

    return end_points, end_points_transformed, cam_configs, transformations


def plot_isotropy_by_concentrations(
        layout: str = 'thesis',
        use_transformed: bool = True
):
    """
    Plot the isotropy across concentrations.
    """
    args = parse_args()
    end_points_og, end_points_transformed, cam_configs, transformations = _generate_or_load_data(
        args, rebuild_cache=False, cache_only=True
    )
    if use_transformed:
        end_points = end_points_transformed
    else:
        end_points = end_points_og

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    # Collate data
    concs = [c for c in end_points.keys() if _is_included(c)]

    # Set up plot
    if layout == 'thesis':
        plt.rc('axes', titlesize=10, titlepad=7)  # fontsize of the title
        plt.rc('axes', labelsize=9, labelpad=0)  # fontsize of the X label
        plt.rc('xtick', labelsize=9)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2, size=5)
        plt.rc('ytick.major', pad=2, size=5)
        fig, axes = plt.subplots(len(concs), figsize=(4, 2 * len(concs)), gridspec_kw=dict(
            top=0.98,
            bottom=0.03,
            left=0.13,
            right=0.97,
            hspace=1
        ))
    else:
        plt.rc('axes', labelsize=7)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)

    # Determine positions
    concs = [c for c in end_points.keys() if _is_included(c)]
    # positions = np.arange(len(concs))
    # lim = np.max([np.max(np.abs(end_points[c])) for c in concs])
    positions = np.arange(3)

    for i, c in enumerate(concs):
        ax = axes[i]

        data = [[], [], []]
        for j in range(3):
            data[j].extend(end_points[c][:, j])
        data = np.array(data).T
        lim = np.abs(data).max()

        # Calculate directional biases
        xtick_labels = []
        for j, xyz in enumerate('xyz'):
            res = ttest_1samp(data[:, j], 0)
            xtick_labels.append(f'{xyz}\nt={res.statistic:.3f}\np={res.pvalue:.3f}')

        ax.set_title(f'Concentration: {c:.2f}%\n'
                     f'{len(data)} trials, {"transformed" if use_transformed else "original"} data.')

        ax.violinplot(
            data,
            positions,
            widths=0.5,
            showmeans=False,
            showmedians=False,
        )

        ax.set_ylabel('Displacement (mm)')
        ax.axhline(y=0, linestyle='--', color='red')
        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylim(bottom=-lim * 1.1, top=lim * 1.1)

    if save_plots:
        if args.restrict_concs is not None and len(args.restrict_concs) > 0:
            c_str = '_c=' + ','.join([f'{c:.2f}' for c in args.restrict_concs])
        else:
            c_str = ''

        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_isotropy'
                            f'_{"transformed" if use_transformed else "original"}'
                            f'_{_identifiers(args)}'
                            f'{c_str}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def plot_isotropy_summary(
        layout: str = 'thesis',
        use_transformed: bool = True
):
    """
    Plot the pauses, durations against activity.
    """
    args = parse_args()
    end_points_og, end_points_transformed, cam_configs, transformations = _generate_or_load_data(
        args, rebuild_cache=False, cache_only=True
    )
    if use_transformed:
        end_points = end_points_transformed
    else:
        end_points = end_points_og

    def _is_included(c) -> bool:
        return not (args.restrict_concs is not None and c not in args.restrict_concs)

    # Collate data
    concs = [c for c in end_points.keys() if _is_included(c)]
    data = [[], [], []]
    for c in concs:
        for i in range(3):
            data[i].extend(end_points[c][:, i])
    data = np.array(data).T
    lim = np.abs(data).max()

    # Calculate directional biases
    xtick_labels = []
    for i, xyz in enumerate('xyz'):
        res = ttest_1samp(data[:, i], 0)
        xtick_labels.append(f'{xyz}\nt={res.statistic:.3f}\np={res.pvalue:.3f}')

    # Set up plot
    if layout == 'thesis':
        plt.rc('axes', titlesize=10, titlepad=7)  # fontsize of the title
        plt.rc('axes', labelsize=9, labelpad=0)  # fontsize of the X label
        plt.rc('xtick', labelsize=9)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=9)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2, size=5)
        plt.rc('ytick.major', pad=2, size=5)
        fig, ax = plt.subplots(1, figsize=(4, 3), gridspec_kw=dict(
            top=0.87,
            bottom=0.17,
            left=0.13,
            right=0.97,
        ))
    else:
        plt.rc('axes', labelsize=7)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('xtick.minor', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)

    ax.set_title(f'Isotropy Summary\n'
                 f'{len(data)} trials, {len(concs)} concentrations, {"transformed" if use_transformed else "original"} data.')
    positions = np.arange(3)
    ax.violinplot(
        data,
        positions,
        widths=0.5,
        showmeans=False,
        showmedians=False,
    )

    ax.set_ylabel('Displacement (mm)')
    ax.axhline(y=0, linestyle='--', color='red')
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(bottom=-lim * 1.1, top=lim * 1.1)

    if save_plots:
        if args.restrict_concs is not None and len(args.restrict_concs) > 0:
            c_str = '_c=' + ','.join([f'{c:.2f}' for c in args.restrict_concs])
        else:
            c_str = ''

        path = LOGS_PATH / (f'{START_TIMESTAMP}'
                            f'_isotropy_summary'
                            f'_{"transformed" if use_transformed else "original"}'
                            f'_{_identifiers(args)}'
                            f'{c_str}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    plot_isotropy_by_concentrations(layout='thesis', use_transformed=True)
    plot_isotropy_summary(layout='thesis', use_transformed=True)
    plot_isotropy_by_concentrations(layout='thesis', use_transformed=False)
    plot_isotropy_summary(layout='thesis', use_transformed=False)
