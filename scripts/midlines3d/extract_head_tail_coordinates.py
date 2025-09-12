#!/usr/bin/env python3
"""
Script to extract all midline2d data from the database for trial id = 37.

"""

import csv
import numpy as np
from pathlib import Path

from wormlab3d import logger
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.trial import Trial
from wormlab3d.data.model.frame import Frame


def extract_midlines2d_for_trial(trial_id: int, output_dir: str = None):
    """
    Extract all Midline2D data for a specific trial ID, filtered by specific users.
    Creates a CSV file with head and tail coordinates in the same format as the reference file.
    """

    try:
        trial = Trial.objects.get(id=trial_id)
        logger.info(f"Found trial {trial_id}: {trial}")
    except Exception as e:
        logger.error(f"Failed to find trial {trial_id}: {e}")
        return
    
    frames = Frame.objects(trial=trial)
    logger.info(f"Found {len(frames)} frames for trial {trial_id}")
    
    valid_users = ["Rob", "YO"]
    midlines = Midline2D.objects(frame__in=frames, user__in=valid_users)
    logger.info(f"Found {len(midlines)} midline2d annotations for trial {trial_id} by users: {valid_users}")
    
    if len(midlines) == 0:
        logger.warning(f"No midline2d data found for trial {trial_id} by users: {valid_users}")
        return
    
    if output_dir is None:
        output_dir = Path.cwd() / "extracted_midlines2d"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group midlines by frame for easier processing
    midlines_by_frame = {}
    for midline in midlines:
        frame_num = midline.frame.frame_num
        if frame_num not in midlines_by_frame:
            midlines_by_frame[frame_num] = {}
        midlines_by_frame[frame_num][midline.camera] = midline
    
    # Create head and tail coordinates CSV file
    head_tail_data = []
    row_id = 1
    
    for frame_num in sorted(midlines_by_frame.keys()):
        frame_midlines = midlines_by_frame[frame_num]
        
        for camera in sorted(frame_midlines.keys()):
            midline = frame_midlines[camera]
            
            coordinates = midline.get_prepared_coordinates()
            
            if len(coordinates) == 0:
                logger.warning(f"Empty coordinates for frame {frame_num}, camera {camera}")
                continue
            
            # Head is the first coordinate, tail is the last coordinate
            head_x, head_y = coordinates[0]
            tail_x, tail_y = coordinates[-1]
            
            head_tail_data.append({
                'id': row_id,
                'midline_record_id': str(midline.id),
                'frame_position': frame_num,
                'frame_id': camera,
                'x_head': head_x,
                'y_head': head_y,
                'x_tail': tail_x,
                'y_tail': tail_y
            })
            
            row_id += 1
    
    head_tail_csv_file = output_dir / f"head_and_tail_man_annotated.csv"
    with open(head_tail_csv_file, 'w', newline='') as f:
        fieldnames = ['id', 'midline_record_id', 'frame_position', 'frame_id', 'x_head', 'y_head', 'x_tail', 'y_tail']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(head_tail_data)
    
    logger.info(f"Head and tail coordinates saved to {head_tail_csv_file}")
    
    extracted_data = []
    for midline in midlines:
        frame_num = midline.frame.frame_num
        camera = midline.camera
        coordinates = midline.get_prepared_coordinates()
        user = midline.user if midline.user else "unknown"
        
        midline_data = {
            'trial_id': trial_id,
            'midline_record_id': str(midline.id),
            'frame_num': frame_num,
            'camera': camera,
            'user': user,
            'n_points': len(coordinates),
        }
        extracted_data.append(midline_data)
    
    summary_file = output_dir / f"trial{trial_id:03d}_midlines2d_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_id', 'midline_record_id', 'frame_num', 'camera', 'user', 'n_points'])
        
        for data in extracted_data:
            writer.writerow([
                data['trial_id'],
                data['midline_record_id'],
                data['frame_num'], 
                data['camera'],
                data['user'],
                data['n_points']
            ])
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Print statistics
    cameras = set(data['camera'] for data in extracted_data)
    frames = set(data['frame_num'] for data in extracted_data)
    users = set(data['user'] for data in extracted_data)
    
    logger.info(f"Extraction complete for trial {trial_id}:")
    logger.info(f"  - Total midlines: {len(extracted_data)}")
    logger.info(f"  - Total head/tail pairs: {len(head_tail_data)}")
    logger.info(f"  - Cameras: {sorted(cameras)}")
    logger.info(f"  - Frame range: {min(frames)} to {max(frames)}")
    logger.info(f"  - Users found: {users}")
    logger.info(f"  - Users requested: {valid_users}")
    logger.info(f"  - Output directory: {output_dir}")
    
    return extracted_data


def main():
    """Main function to run the extraction."""
    trial_id = 37
    
    logger.info(f"Starting extraction of Midline2D data for trial {trial_id}")
    
    try:
        data = extract_midlines2d_for_trial(trial_id)
        if data:
            logger.info("Extraction completed successfully!")
        else:
            logger.warning("No data was extracted.")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise


if __name__ == "__main__":
    main()
