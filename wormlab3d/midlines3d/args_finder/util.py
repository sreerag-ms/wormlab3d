import os
import pandas as pd

from wormlab3d import logger


def load_head_tail_coordinates(csv_path: str, start_frame: int, end_frame: int):
    """
    Load head and tail coordinates CSV data and an attempt to validate as much as possible.
    """
    if not csv_path:
        raise FileNotFoundError("No head and tail coordinates file path specified")
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Head and tail coordinates file not found: {csv_path}")
    
    try:
        coords_df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded head and tail coordinates from: {csv_path}")
        
        if len(coords_df) == 0:
            logger.warning(f"No head and tail coordinates found in the dataset")
            return None
        
        if end_frame == -1:
            filtered_df = coords_df[coords_df['frame_position'] >= start_frame]
        else:
            filtered_df = coords_df[(coords_df['frame_position'] >= start_frame) & (coords_df['frame_position'] <= end_frame)]
        
        if len(filtered_df) == 0:
            logger.warning(f"No frames found between {start_frame} and {end_frame}")
            return None
        
        frame_positions = filtered_df['frame_position'].unique()
        for frame in frame_positions:
            frame_data = filtered_df[filtered_df['frame_position'] == frame]
            frame_ids = set(frame_data['frame_id'].values)
            if not {0, 1, 2}.issubset(frame_ids):
                missing_ids = {0, 1, 2} - frame_ids
                logger.warning(f"Frame {frame} is missing data for worm(s) with frame_id {missing_ids}")
        
        logger.info(f"Found {len(filtered_df)} coordinate entries for {len(frame_positions)} frames")
        return filtered_df
            
    except Exception as e:
        logger.error(f"Error loading head and tail coordinates: {e}")
        raise FileNotFoundError(f"Could not load head and tail coordinates from {csv_path}: {e}")
