#!/usr/bin/env python3
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Frame, Midline3D, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_WT3D, M3D_SOURCE_MF
from wormlab3d.toolkit.util import print_args

show_plots = False
save_plots = True
img_extension = 'png'


def get_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description='Plot WT3D midlines on original frame image.')
    
    parser.add_argument('--frame-num', type=int, required=True, 
                       help='Frame number to plot')
    parser.add_argument('--cam-id', type=int, required=True, choices=[0, 1, 2],
                       help='Camera index (0, 1, or 2)')
    parser.add_argument('--trial-id', type=int, required=True,
                       help='Trial ID')
    
    args = parser.parse_args()
    print_args(args)
    return args


def get_wt3d_reconstruction(trial: Trial) -> Optional[Reconstruction]:
    """Find M3D_SOURCE_WT3D reconstruction for the trial."""
    try:
        rec = Reconstruction.objects(trial=trial, source=M3D_SOURCE_WT3D).first()
        return rec
    except Exception as e:
        logger.warning(f"No WT3D reconstruction found for trial {trial.id}: {e}")
        return None


def get_midline_points_2d(reconstruction: Reconstruction, frame: Frame, frame_num: int) -> np.ndarray:
    """Get 2D projected midline points for a given reconstruction and frame."""
    try:
        midline = Midline3D.objects.get(
            frame=frame.id,
            source=reconstruction.source,
            source_file=reconstruction.source_file,
        )
        points_2d = midline.get_prepared_2d_coordinates()
        points_2d = np.stack(points_2d, axis=1)
        return points_2d
    except Exception as e:
        logger.error(f"Could not get midline for WT3D reconstruction: {e}")
        return None


def draw_midline_on_image(img_array: np.ndarray, points_2d: np.ndarray, cam_id: int, 
                         color: Tuple[int, int, int, int] = (0, 255, 0, 255), 
                         scale_factor: float = 4.0) -> np.ndarray:
    """Draw midline on image using OpenCV with scaling for higher resolution."""
    if img_array.dtype != np.uint8:
        img_8bit = (img_array * 255).astype(np.uint8)
    else:
        img_8bit = img_array
        
    if len(img_8bit.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_8bit.copy()
        if img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    
    original_height, original_width = img_rgb.shape[:2]
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    img_scaled = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    if points_2d is not None and len(points_2d) > 0:
        cam_points = points_2d[:, cam_id, :] * scale_factor
        cam_points = cam_points.astype(np.int32)
        
        valid_mask = ((cam_points[:, 0] >= 0) & (cam_points[:, 0] < new_width) & 
                     (cam_points[:, 1] >= 0) & (cam_points[:, 1] < new_height))
        
        valid_points = cam_points[valid_mask]
        
        if len(valid_points) > 1:
            line_thickness = max(1, int(scale_factor * 0.5))
            for i in range(len(valid_points) - 1):
                pt1 = tuple(valid_points[i])
                pt2 = tuple(valid_points[i + 1])
                cv2.line(img_scaled, pt1, pt2, color[:3], thickness=line_thickness, lineType=cv2.LINE_AA)
            
            circle_radius = max(1, int(scale_factor * 0.3))
            for point in valid_points:
                cv2.circle(img_scaled, tuple(point), circle_radius, color[:3], -1)
        
        logger.debug(f"Drew {len(valid_points)} valid points out of {len(cam_points)} total points for camera {cam_id}")
    
    return img_scaled


def create_wt3d_plot(frame: Frame, cam_id: int, wt3d_points: Optional[np.ndarray]) -> np.ndarray:
    """Create a plot with WT3D midline on original image."""
    
    if len(frame.images) < 3:
        frame.generate_prepared_images()
        frame.save()
    
    base_img = frame.images[cam_id]
    
    forest_green = (34, 139, 34, 255)
    scale_factor = 4.0  # Scale up to 800x800 from 200x200
    
    result_img = draw_midline_on_image(base_img, wt3d_points, cam_id, forest_green, scale_factor)
    
    return result_img


def main():
    """Main function."""
    args = get_args()
    
    try:
        from wormlab3d.data.model import Trial
        trial = Trial.objects.get(id=args.trial_id)
    except Exception as e:
        logger.error(f"Failed to get trial: {e}")
        return
    
    try:
        frame = trial.get_frame(args.frame_num)
    except Exception as e:
        logger.error(f"Failed to get frame {args.frame_num}: {e}")
        return
    
    wt3d_rec = get_wt3d_reconstruction(trial)
    
    if wt3d_rec is None:
        logger.error("No WT3D reconstruction found for this trial")
        return
    
    logger.info("Getting WT3D midline points...")
    
    wt3d_points = None
    try:
        wt3d_points = get_midline_points_2d(wt3d_rec, frame, args.frame_num)
        logger.info(f"Got WT3D points with shape: {wt3d_points.shape if wt3d_points is not None else None}")
    except Exception as e:
        logger.warning(f"Failed to get WT3D points: {e}")
    
    logger.info("Creating WT3D plot...")
    final_img = create_wt3d_plot(frame, args.cam_id, wt3d_points)
    
    # Save the image as PNG
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        filename = (f'{START_TIMESTAMP}_wt3d_midline_'
                   f'trial={trial.id}_frame={args.frame_num}_cam={args.cam_id}.{img_extension}')
        save_path = LOGS_PATH / filename
        
        logger.info(f'Saving WT3D plot to {save_path}')
        img_pil = Image.fromarray(final_img, 'RGB')
        img_pil.save(save_path)
    
    if show_plots:
        img_pil = Image.fromarray(final_img, 'RGB')
        img_pil.show()
    
    logger.info("WT3D plot completed successfully!")


if __name__ == '__main__':
    main()
