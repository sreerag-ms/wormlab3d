import os
from argparse import ArgumentParser

import pandas as pd

from wormlab3d import logger
from wormlab3d.nn.args.base_args import BaseArgs


class SourceArgs(BaseArgs):
    def __init__(
            self,
            trial_id: int,
            start_frame: int = 0,
            end_frame: int = -1,
            read_head_and_tail_coordinates: bool = True,
            head_and_tail_coordinates: str = 'data/head_and_tail_coords_dataset_2.csv',
            **kwargs
    ):
        self.trial_id = trial_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.read_head_and_tail_coordinates = read_head_and_tail_coordinates
        
        if read_head_and_tail_coordinates:
            self.head_and_tail_coordinates = self._load_head_tail_coordinates(head_and_tail_coordinates, trial_id)
        else:
            self.head_and_tail_coordinates = None
            logger.info("Head and tail coordinates loading disabled")
        
        if end_frame == -1 or start_frame < end_frame:
            self.direction = 1
        else:
            self.direction = -1

    def _load_head_tail_coordinates(self, csv_path: str, trial_id: int):
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
            
            if self.end_frame == -1:
                filtered_df = coords_df[coords_df['frame_position'] >= self.start_frame]
            else:
                filtered_df = coords_df[(coords_df['frame_position'] >= self.start_frame) & 
                                        (coords_df['frame_position'] <= self.end_frame)]
            
            if len(filtered_df) == 0:
                logger.warning(f"No frames found between {self.start_frame} and {self.end_frame}")
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

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group('Source Args')
        group.add_argument('--trial-id',   type=int,   help='Database id for a Trial instance to use as the source.')
        group.add_argument('--start-frame', type=int, default=0, help='Frame number to start from.')
        group.add_argument('--end-frame',   type=int, default=-1, help='Frame number to end at.')
        group.add_argument('--head-and-tail-coordinates', type=str, default='data/head_and_tail_coords_dataset_2.csv', help='Path to head and tail coordinates dataset CSV file.')
        group.add_argument('--read-head-and-tail-coordinates', action='store_true', default=False, help='Whether to read and load head and tail coordinates from CSV file.')
        group.add_argument('--no-read-head-and-tail-coordinates', dest='read_head_and_tail_coordinates', action='store_false', help='Disable reading head and tail coordinates from CSV file.')

