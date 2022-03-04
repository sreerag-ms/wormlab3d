import json
from typing import Tuple, Any, Dict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from wormlab3d import logger, POSTURE_DISTANCES_CACHE_PATH
from wormlab3d.data.model import Reconstruction, Eigenworms
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.natural_frame import align_complex_vectors
from wormlab3d.trajectories.cache import get_trajectory


def distance_ab(a: np.ndarray, b: np.ndarray) -> float:
    a2 = align_complex_vectors(a, b)
    return np.linalg.norm(a2 - b)


def _calculate_posture_distances(postures: np.ndarray) -> np.ndarray:
    """
    Calculate distances between postures.
    """
    distances = pdist(postures, distance_ab)
    return distances


def get_posture_distances(
        reconstruction_id: str = None,
        use_eigenworms: bool = False,
        eigenworms_id: str = None,
        eigenworms_n_components: int = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
        return_squareform: bool = False,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the full distances matrix and then return required section.
    """
    X, meta = generate_or_load_distances_cache(
        reconstruction_id=reconstruction_id,
        use_eigenworms=use_eigenworms,
        eigenworms_id=eigenworms_id,
        eigenworms_n_components=eigenworms_n_components,
        start_frame=start_frame,
        end_frame=end_frame,
        depth=depth,
        return_squareform=return_squareform,
        rebuild_cache=rebuild_cache
    )
    meta['frame_nums'] = np.arange(meta['start_frame'], meta['end_frame'] + 1)

    # Convert to float64
    X = X.astype(np.float64)

    # Convert frame nums to a list of ints
    meta['frame_nums'] = meta['frame_nums'].tolist()

    return X, meta


def _generate_distances_cache_data(
        reconstruction: Reconstruction,
        depth: int = -1,
        ew: Eigenworms = None,
) -> np.ndarray:
    """
    Load a set of postures and calculate distances between them.
    """
    Z, meta = get_trajectory(
        reconstruction_id=reconstruction.id,
        depth=depth,
        natural_frame=True,
    )

    # Eigenworms embeddings
    if ew is not None:
        Z = ew.transform(np.array(Z))

    # Calculate distances matrix
    logger.info('Calculating pairwise distances.')
    distances = _calculate_posture_distances(Z)

    return distances


def generate_or_load_distances_cache(
        reconstruction_id: str = None,
        use_eigenworms: bool = False,
        eigenworms_id: str = None,
        eigenworms_n_components: int = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
        return_squareform: bool = False,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Try to load an existing distances cache or generate it otherwise.
    """
    reconstruction = Reconstruction.objects.get(id=reconstruction_id)
    trial = reconstruction.trial

    # Fetch eigenworms instance if needed
    ew = None
    if use_eigenworms:
        ew = generate_or_load_eigenworms(
            eigenworms_id=eigenworms_id,
            reconstruction_id=reconstruction_id,
            n_components=eigenworms_n_components,
            regenerate=False
        )

    # Generate or load the pairwise distances between all postures in the reconstruction.
    ews_suffix = f'ew_{ew.id}' if use_eigenworms else 'nf'
    depth_suffix = f'_d={depth}' if depth != -1 else ''
    filename = f'{reconstruction.id}{depth_suffix}_{ews_suffix}'
    filename_meta = f'{filename}.meta'
    filename_X = f'{filename}.npz'
    path_meta = POSTURE_DISTANCES_CACHE_PATH / filename_meta
    path_X = POSTURE_DISTANCES_CACHE_PATH / filename_X
    X_full = None
    if not rebuild_cache and path_meta.exists() and path_X.exists():
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
            X_full = np.memmap(path_X, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
            logger.info(f'Loaded posture distances data from {path_X}.')
        except Exception as e:
            logger.warning(f'Could not load posture distances from {path_X}. {e}')
    elif not rebuild_cache:
        logger.info('Posture distances file cache unavailable, building.')
    else:
        logger.info('Rebuilding posture distances cache.')

    if X_full is None:
        # Generate the distances cache
        X_data = _generate_distances_cache_data(reconstruction, depth, ew)
        X_full = np.memmap(path_X, dtype=np.float32, mode='w+', shape=X_data.shape)
        X_full[:] = X_data

        # Save the cache onto the hard drive
        logger.debug(f'Saving posture distances file cache to {path_X}.')
        X_full.flush()

        # Save the meta data
        meta = {'shape': X_full.shape, }
        with open(path_meta, 'w') as f:
            json.dump(meta, f)

    # Set frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = trial.n_frames_min
    start_frame = max(start_frame, reconstruction.start_frame)
    end_frame = min(end_frame, reconstruction.end_frame)
    meta['start_frame'] = start_frame
    meta['end_frame'] = end_frame

    if return_squareform or (start_frame > 0 or end_frame < reconstruction.end_frame):
        X_sf = squareform(X_full)

        # Take a slice of the full array
        if start_frame > 0 or end_frame < reconstruction.end_frame:
            X_sf = X_sf[start_frame:end_frame + 1, start_frame:end_frame + 1]

        if return_squareform:
            X = X_sf
        else:
            X = squareform(X_sf, checks=False)
    else:
        X = X_full

    return X, meta
