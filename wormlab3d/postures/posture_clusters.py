import json
from typing import Tuple, Any, Dict

import numpy as np
from fastcluster import linkage
from scipy.cluster.hierarchy import optimal_leaf_ordering

from wormlab3d import POSTURE_CLUSTERS_CACHE_PATH
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.posture_distances import get_posture_distances


def _calculate_linkage(distances: np.ndarray, linkage_method: str) -> np.ndarray:
    """
    Calculate linkage matrix.
    """
    L = linkage(distances, linkage_method)
    L = optimal_leaf_ordering(L, distances)
    return L


def get_posture_clusters(
        reconstruction_id: str = None,
        use_eigenworms: bool = False,
        eigenworms_id: str = None,
        eigenworms_n_components: int = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
        linkage_method: str = 'ward',
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the posture cluster linkage matrix.
    """
    L, meta = generate_or_load_clusters_cache(
        reconstruction_id=reconstruction_id,
        use_eigenworms=use_eigenworms,
        eigenworms_id=eigenworms_id,
        eigenworms_n_components=eigenworms_n_components,
        start_frame=start_frame,
        end_frame=end_frame,
        depth=depth,
        linkage_method=linkage_method,
        rebuild_cache=rebuild_cache
    )

    # Convert to float64
    L = L.astype(np.float64)

    return L, meta


def generate_or_load_clusters_cache(
        reconstruction_id: str = None,
        use_eigenworms: bool = False,
        eigenworms_id: str = None,
        eigenworms_n_components: int = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
        linkage_method: str = 'ward',
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Try to load an existing clusters cache or generate it otherwise.
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

    # Set frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = trial.n_frames_min
    start_frame = max(start_frame, reconstruction.start_frame)
    end_frame = min(end_frame, reconstruction.end_frame)
    frame_range = f'({start_frame}-{end_frame})'

    # Add eigenworms suffix if using
    ews_suffix = f'ew_{ew.id}' if use_eigenworms else 'nf'

    # Generate or load the pairwise distances between all postures in the reconstruction.
    filename = f'{reconstruction.id}_{frame_range}_{ews_suffix}_L={linkage_method}'
    filename_meta = f'{filename}.meta'
    filename_L = f'{filename}.npz'
    path_meta = POSTURE_CLUSTERS_CACHE_PATH / filename_meta
    path_L = POSTURE_CLUSTERS_CACHE_PATH / filename_L
    L = None
    if not rebuild_cache and path_meta.exists() and path_L.exists():
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
            L = np.memmap(path_L, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
            logger.info(f'Loaded posture clusters data from {path_L}.')
        except Exception as e:
            logger.warning(f'Could not load posture clusters from {path_L}. {e}')
    elif not rebuild_cache:
        logger.info('Posture clusters file cache unavailable, building.')
    else:
        logger.info('Rebuilding posture clusters cache.')

    if L is None:
        # Get the posture distances
        distances, _ = get_posture_distances(
            reconstruction_id=reconstruction_id,
            use_eigenworms=use_eigenworms,
            eigenworms_id=ew.id if use_eigenworms else None,
            eigenworms_n_components=ew.n_components if use_eigenworms else None,
            start_frame=start_frame,
            end_frame=end_frame,
            depth=depth,
            return_squareform=False,
            rebuild_cache=False
        )

        # Generate the clusters cache
        logger.info('Calculating linkage.')
        L_data = _calculate_linkage(distances, linkage_method)
        L = np.memmap(path_L, dtype=np.float32, mode='w+', shape=L_data.shape)
        L[:] = L_data

        # Save the cache onto the hard drive
        logger.debug(f'Saving posture clusters file cache to {path_L}.')
        L.flush()

        # Save the meta data
        meta = {'shape': L.shape, }
        with open(path_meta, 'w') as f:
            json.dump(meta, f)

    return L, meta
