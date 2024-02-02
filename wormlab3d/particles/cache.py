from argparse import Namespace
from typing import Union

import numpy as np

from wormlab3d import PE_CACHE_PATH, logger
from wormlab3d.data.model import PEParameters
from wormlab3d.data.model.pe_parameters import PE_MODEL_RUNTUMBLE
from wormlab3d.particles.args.parameter_args import ParameterArgs
from wormlab3d.particles.simulation_state import SimulationState
from wormlab3d.toolkit.util import hash_data


def _init_parameters(args: ParameterArgs) -> PEParameters:
    """
    Create or load particle explorer parameters.
    """
    db_params = args.get_db_params()
    parameters = None

    # If we have a model id then load this from the database
    if args.params_id is not None:
        parameters = PEParameters.objects.get(id=args.params_id)
    else:
        # Otherwise, try to find one matching the same parameters
        params_matching = PEParameters.objects(**db_params)
        if params_matching.count() > 0:
            parameters = params_matching[0]
            logger.info(f'Found {len(params_matching)} suitable parameter records in database, using most recent.')
        else:
            logger.info(f'No suitable parameter records found in database.')
    if parameters is not None:
        logger.info(f'Loaded parameters (id={parameters.id}, created={parameters.created}).')

    # Not loaded model, so create one
    if parameters is None:
        parameters = PEParameters(**db_params)
        parameters.save()
        logger.info(f'Saved parameters to database (id={parameters.id})')

    return parameters


def get_sim_state_from_args(args: Union[ParameterArgs, Namespace], no_cache: bool = False) -> SimulationState:
    """
    Generate or load the trajectories from parameters set in an argument namespace.
    """
    if type(args) == Namespace:
        args = ParameterArgs.from_args(args)
    params = _init_parameters(args)
    SS = SimulationState(params, read_only=False, regenerate=args.regenerate, no_cache=no_cache)
    return SS


def get_npas_from_args(args: Namespace) -> np.ndarray:
    return np.exp(-np.linspace(np.log(1 / args.npas_min), np.log(1 / args.npas_max), args.npas_num))


def get_voxel_sizes_from_args(args: Namespace) -> np.ndarray:
    return np.exp(-np.linspace(np.log(1 / args.vxs_max), np.log(1 / args.vxs_min), args.vxs_num))


def get_durations_from_args(args: Namespace) -> np.ndarray:
    if args.durations_intervals == 'exponential':
        durations = np.exp(-np.linspace(np.log(1 / (args.durations_min * 60)), np.log(1 / (args.durations_min * 60)),
                                        args.durations_num))
    else:
        durations = np.arange(args.durations_min, args.durations_min + args.durations_num)**2 * 60
    durations = np.round(durations / args.sim_dt) * args.sim_dt
    return durations


def get_pauses_from_args(args: Namespace) -> np.ndarray:
    if args.pauses_intervals == 'exponential':
        if args.pauses_min == 0:
            pauses = np.r_[[0, ], np.exp(-np.linspace(np.log(1 / 1), np.log(1 / args.pauses_max), args.pauses_num - 1))]
        else:
            pauses = np.exp(-np.linspace(np.log(1 / args.pauses_min), np.log(1 / args.pauses_max), args.pauses_num))
    else:
        pauses = np.arange(args.pauses_min, args.pauses_min + args.pauses_num)**2
    return pauses


def _calculate_r_values(
        args: Namespace
) -> np.ndarray:
    """
    Calculate the r values across a range of sigmas, durations and pauses.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)

    # Outputs
    r_values = np.zeros((n_sigmas, n_durations, n_pauses, 3, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'Simulating exploration with sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}. '
                    f'({sim_idx + 1}/{n_sims}).'
                )

                if args.model_type == PE_MODEL_RUNTUMBLE:
                    args.phi_factor_rt = npas
                else:
                    args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Find the maximums in each relative directions
                Xt = SS.get_Xt()
                Xt_max = np.abs(Xt).max(axis=1)

                # Collect the batch-mean, min, max and std.
                r_values[i, j, k] = np.array([
                    Xt_max.mean(axis=0),
                    Xt_max.min(axis=0),
                    Xt_max.max(axis=0),
                    Xt_max.std(axis=0)
                ]).T

                if SS.needs_save:
                    SS.save()
                sim_idx += 1

    return r_values


def generate_or_load_r_values(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the r values - distances travelled along each of the principal component axes.
    """
    if not hasattr(args, 'npas'):
        npas = get_npas_from_args(args)
    else:
        npas = args.npas
    if not hasattr(args, 'sim_durations'):
        durations = get_durations_from_args(args)
    else:
        durations = args.sim_durations
    if not hasattr(args, 'pauses'):
        pauses = get_pauses_from_args(args)
    else:
        pauses = args.pauses

    keys = {
        'npas': [f'{s:.3E}' for s in npas],
        'durations': [f'{d:.4f}' for d in durations],
        'pauses': [f'{p:.4f}' for p in pauses],
    }

    cache_id = 'r_vals_'
    if args.model_type == PE_MODEL_RUNTUMBLE:
        assert args.approx_args is not None, 'Run and tumble model requires approx_args!'
        keys = {**keys, **{
            'ds': args.dataset,
            'approx_args': args.approx_args,
            'batch_size': args.batch_size,
            'nonp_pause_type': args.nonp_pause_type,
            'nonp_pause_max': args.nonp_pause_max,
        }}
        cache_id += 'rt_'

    cache_id += hash_data(keys)
    cache_path = PE_CACHE_PATH / cache_id
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    r_values = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            r_values = data['r_values']
            assert r_values.shape == (len(args.npas), len(args.sim_durations), len(args.pauses), 3, 4), \
                'Invalid r_values shape.'
            logger.info(f'Loaded r values from cache: {cache_fn}')
        except Exception as e:
            r_values = None
            logger.warning(f'Could not load cache: {e}')

    if r_values is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating r values.')
        r_values = _calculate_r_values(args)
        save_arrs = {
            'r_values': r_values,
        }
        logger.info(f'Saving r values to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return r_values


def _calculate_volumes(
        args: Namespace
) -> np.ndarray:
    """
    Calculate the volumes across a range of sigmas, durations and pauses.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)

    def _calculate_volumes(r_):
        if args.volume_metric == 'disks':
            radius = r_[:, 0]
            height = r_[:, 2]
            sphere_vols = 4 / 3 * np.pi * radius**3
            cap_vols = 1 / 3 * np.pi * (radius - height)**2 * (2 * radius + height)
            return sphere_vols - 2 * cap_vols
        elif args.volume_metric == 'cuboids':
            r1 = r_[:, 0]
            r2 = r_[:, 1]
            r3 = r_[:, 2]
            return r1 * r2 * r3

    # Outputs
    vols = np.zeros((n_sigmas, n_durations, n_pauses, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'Simulating exploration with sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}. '
                    f'({sim_idx + 1}/{n_sims}).'
                )

                if args.model_type == PE_MODEL_RUNTUMBLE:
                    args.phi_factor_rt = npas
                else:
                    args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Find the maximums in each relative directions
                Xt = SS.get_Xt()
                Xt_max = np.abs(Xt).max(axis=1)

                # Collect the batch-mean, min, max and std.
                vols_ijk = _calculate_volumes(Xt_max)

                vols[i, j, k] = np.array([
                    vols_ijk.mean(),
                    vols_ijk.min(),
                    vols_ijk.max(),
                    vols_ijk.std(),
                ]).T

                if SS.needs_save:
                    SS.save()
                sim_idx += 1

    return vols


def generate_or_load_volumes(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the volumes.
    """
    if not hasattr(args, 'npas'):
        npas = get_npas_from_args(args)
    else:
        npas = args.npas
    if not hasattr(args, 'sim_durations'):
        durations = get_durations_from_args(args)
    else:
        durations = args.sim_durations
    if not hasattr(args, 'pauses'):
        pauses = get_pauses_from_args(args)
    else:
        pauses = args.pauses

    keys = {
        'npas': [f'{s:.3E}' for s in npas],
        'durations': [f'{d:.4f}' for d in durations],
        'pauses': [f'{p:.4f}' for p in pauses],
    }
    cache_id = f'vols_{args.volume_metric}_'

    if args.model_type == PE_MODEL_RUNTUMBLE:
        assert args.approx_args is not None, 'Run and tumble model requires approx_args!'
        keys = {**keys, **{
            'ds': args.dataset,
            'approx_args': args.approx_args,
            'batch_size': args.batch_size,
            'nonp_pause_type': args.nonp_pause_type,
            'nonp_pause_max': args.nonp_pause_max,
        }}
        cache_id += 'rt_'

    cache_id += hash_data(keys)
    cache_path = PE_CACHE_PATH / cache_id
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    vols = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            vols = data['vols']
            assert vols.shape == (len(args.npas), len(args.sim_durations), len(args.pauses), 4), \
                'Invalid vols shape.'
            logger.info(f'Loaded volume values from cache: {cache_fn}')
        except Exception as e:
            vols = None
            logger.warning(f'Could not load cache: {e}')

    if vols is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Calculating volumes.')
        vols = _calculate_volumes(args)
        save_arrs = {
            'vols': vols,
        }
        logger.info(f'Saving volume values to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return vols


def _calculate_voxel_scores(
        args: Namespace
) -> np.ndarray:
    """
    Calculate the voxel scores across a range of sigmas, durations, pauses and voxel sizes.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)
    if hasattr(args, 'voxel_sizes'):
        voxel_sizes = args.voxel_sizes
    else:
        voxel_sizes = [args.vxs, ]
    n_voxel_sizes = len(voxel_sizes)

    # Outputs
    scores = np.zeros((n_sigmas, n_durations, n_pauses, n_voxel_sizes, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'{sim_idx + 1}/{n_sims}: '
                    f'sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}.'
                )
                args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Calculate coverage
                for l, vs in enumerate(voxel_sizes):
                    vc = SS.get_coverage(vs) / vs
                    scores[i, j, k, l] = [vc.mean(), vc.min(), vc.max(), vc.std()]
                if SS.needs_save:
                    SS.save()

                sim_idx += 1

    return scores


def generate_or_load_voxel_scores(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the voxel coverage of trajectories.
    """
    if not hasattr(args, 'npas'):
        npas = get_npas_from_args(args)
    else:
        npas = args.npas
    if not hasattr(args, 'sim_durations'):
        durations = get_durations_from_args(args)
    else:
        durations = args.sim_durations
    if not hasattr(args, 'pauses'):
        pauses = get_pauses_from_args(args)
    else:
        pauses = args.pauses
    if hasattr(args, 'voxel_sizes'):
        vs = [f'{p:.3E}' for p in args.voxel_sizes]
    else:
        vs = [f'{args.vxs:.3E}', ]
    keys = {
        'npas': [f'{s:.3E}' for s in npas],
        'durations': [f'{d:.4f}' for d in durations],
        'pauses': [f'{p:.4f}' for p in pauses],
        'vs': vs,
    }
    cache_path = PE_CACHE_PATH / f'vox_vals_{hash_data(keys)}'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    scores = None
    if not rebuild_cache and cache_fn.exists():
        data = np.load(cache_fn)
        scores = data['scores']
        logger.info(f'Loaded scores from cache: {cache_fn}')

    if scores is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating voxel coverage values.')
        scores = _calculate_voxel_scores(args)
        save_arrs = {
            'scores': scores,
        }
        logger.info(f'Saving voxel scores to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return scores


def _calculate_fractal_dimensions(
        args: Namespace
) -> np.ndarray:
    """
    Calculate the fractal dimensions across a range of sigmas, durations, pauses and voxel sizes.
    """
    npa_sigmas = args.npas
    n_sigmas = len(npa_sigmas)
    sim_durations = args.sim_durations
    n_durations = len(sim_durations)
    pauses = args.pauses
    n_pauses = len(pauses)
    assert args.vxs_num > 20, 'More voxel sizes required!'
    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]

    # Outputs
    fds = np.zeros((n_sigmas, n_durations, n_pauses, 4))
    n_sims = n_sigmas * n_durations * n_pauses
    sim_idx = 0

    # Sweep over the combinations
    for i, npas in enumerate(npa_sigmas):
        for j, duration in enumerate(sim_durations):
            for k, pause in enumerate(pauses):
                logger.info(
                    f'{sim_idx + 1}/{n_sims}: '
                    f'sigma={npas:.2E}, duration={duration:.2f}, pause={pause:.2f}.'
                )
                args.phi_dist_params[1] = npas
                args.sim_duration = duration
                args.nonp_pause_max = pause
                SS = get_sim_state_from_args(args)

                # Calculate dimensions
                dims = SS.get_fractal_dimensions(
                    noise_scale=args.approx_noise,
                    smoothing_window=args.smoothing_window,
                    voxel_sizes=voxel_sizes,
                    plateau_threshold=args.fd_plateau_threshold,
                    sample_size=args.fd_sample_size,
                    sf_min=args.fd_sf_min,
                    sf_max=args.fd_sf_max,
                )
                fds[i, j, k] = [dims.mean(), dims.min(), dims.max(), dims.std()]

                if SS.needs_save:
                    SS.save()

                sim_idx += 1

    return fds


def generate_or_load_fractal_dimensions(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> np.ndarray:
    """
    Generate or load the fractal dimensions of trajectories.
    """
    if not hasattr(args, 'npas'):
        npas = get_npas_from_args(args)
    else:
        npas = args.npas
    if not hasattr(args, 'sim_durations'):
        durations = get_durations_from_args(args)
    else:
        durations = args.sim_durations
    if not hasattr(args, 'pauses'):
        pauses = get_pauses_from_args(args)
    else:
        pauses = args.pauses

    voxel_sizes = get_voxel_sizes_from_args(args)[::-1]

    keys = {
        'npas': [f'{s:.3E}' for s in npas],
        'durations': [f'{d:.4f}' for d in durations],
        'pauses': [f'{p:.4f}' for p in pauses],
        'noise_scale': f'{args.approx_noise:.4f}' if args.approx_noise is not None else '',
        'smoothing_window': args.smoothing_window,
        'vxs': [f'{v:.5E}' for v in voxel_sizes],
        'pt': args.fd_plateau_threshold,
        'ss': args.fd_sample_size,
        'sf_min': args.fd_sf_min,
        'sf_max': args.fd_sf_max,
    }
    cache_path = PE_CACHE_PATH / f'fractal_dims_{hash_data(keys)}'
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    fds = None
    if not rebuild_cache and cache_fn.exists():
        data = np.load(cache_fn)
        fds = data['fds']
        logger.info(f'Loaded fractal dimensions from cache: {cache_fn}')

    if fds is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Calculating fractal dimensions.')
        fds = _calculate_fractal_dimensions(args)
        save_arrs = {
            'fds': fds,
        }
        logger.info(f'Saving fractal dimensions to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return fds
