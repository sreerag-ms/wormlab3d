import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def print_npz_shapes(file_path):
    """
    Print the shapes of all arrays in an NPZ file.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        data = np.load(file_path)
        
        print(f"Contents of NPZ file: {file_path}")
        print("-" * 50)
        
        # Iterate through all arrays in the NPZ file
        for key in data.files:
            array = data[key]
            print(f"Array name: {key}")
            print(f"Shape: {array.shape}")
            print(f"Data type: {array.dtype}")
            print("-" * 30)
            
        print(f"Total number of arrays: {len(data.files)}")
    
    except Exception as e:
        print(f"Error loading or processing the NPZ file: {e}")

def print_frame(file_path, frame_num):
    """
    Print the contents of the specified frame from the 'X' array in the NPZ file.
    """
    try:
        data = np.load(file_path)
        if 'X' not in data:
            print("Array 'X' not found in the NPZ file.")
            return
        X = data['X']
        if frame_num < 1 or frame_num > X.shape[0]:
            print(f"Frame number {frame_num} is out of bounds (1-{X.shape[0]}).")
            return
        print(f"Contents of frame {frame_num} (index {frame_num-1}) from 'X':")
        print(X[frame_num-1])
    except Exception as e:
        print(f"Error loading or processing the NPZ file: {e}")

def _parse_int_set(spec: str):
    if not spec:
        return set()
    return {int(x) for x in spec.split(',') if x.strip().isdigit()}

def _parse_projections(spec: str):
    axes_map = {'x':0,'y':1,'z':2}
    proj = []
    for pair in spec.split(','):
        pair = pair.strip().lower()
        if len(pair) == 2 and all(c in axes_map for c in pair):
            proj.append((axes_map[pair[0]], axes_map[pair[1]], pair))
    return proj

def _robust_normalize(vals, size, robust=True):
    if robust:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = vals.min(), vals.max()
    if vmax <= vmin:
        return np.full_like(vals, size/2.0)
    vals = np.clip(vals, vmin, vmax)
    return (vals - vmin) / (vmax - vmin) * (size - 1)

def plot_reconstruction_on_images(reconstruction_file, frame_num, images_path=None,
                                  projections=None, flip_x=None, flip_y=None, robust=True):
    """
    Plot 3D reconstruction points overlaid on original images.
    """
    try:
        # Load reconstruction data
        recon_data = np.load(reconstruction_file)
        if 'X' not in recon_data:
            print("Array 'X' not found in the reconstruction NPZ file.")
            return
        
        X = recon_data['X']
        if frame_num < 1 or frame_num > X.shape[0]:
            print(f"Frame number {frame_num} is out of bounds (1-{X.shape[0]}).")
            return
        
        # Get the 3D points for the specified frame
        frame_points = X[frame_num-1]  # Shape: (128, 3)
        print(f"3D points shape for frame {frame_num}: {frame_points.shape}")
        
        # Load original images if path is provided
        if images_path is None:
            # Bad! Hardcoded path for demo purposes
            images_path = f"/Users/sreeragms/Desktop/prepared_images/037/037/{frame_num:06d}.npz"
        
        if not os.path.exists(images_path):
            print(f"Images file not found: {images_path}")
            return
        
        images_data = np.load(images_path)
        if 'images' not in images_data:
            print("Array 'images' not found in the images NPZ file.")
            return
        
        original_images = images_data['images']
        print(f"Original images shape: {original_images.shape}")
        
        H, W = original_images.shape[1], original_images.shape[2]
        
        if projections is None:
            projections = [(0,1,'xy'), (0,2,'xz'), (1,2,'yz')]
        num_cameras = min(original_images.shape[0], len(projections))

        fig, axes = plt.subplots(1, num_cameras, figsize=(2*num_cameras, 2), dpi=100)
        if num_cameras == 1:
            axes = [axes]
        
        for cam_idx in range(num_cameras):
            ax = axes[cam_idx]
            ax.imshow(original_images[cam_idx], cmap='gray', vmin=0, vmax=1)

            ax.set_xlim(0, W-1)
            ax.set_ylim(H-1, 0)
            ax.set_aspect('equal')
            
            ax_i, ay_i, label = projections[cam_idx]
            x_raw = frame_points[:, ax_i]
            y_raw = frame_points[:, ay_i]
            
            x_img = _robust_normalize(x_raw, W, robust)
            y_img = _robust_normalize(y_raw, H, robust)
            
            x_img = (x_img - (W - 1)/2.0) * 0.5 + (W - 1)/2.0
            y_img = (y_img - (H - 1)/2.0) * 0.5 + (H - 1)/2.0
            
            if flip_x and cam_idx in flip_x:
                x_img = (W - 1) - x_img
            if flip_y and cam_idx in flip_y:
                y_img = (H - 1) - y_img
            
            ax.scatter(x_img, y_img, c='red', s=18, alpha=0.75, label=f'3D ({label})')
            ax.scatter(x_img[0], y_img[0], c='blue', s=70, marker='o', label='Head')
            ax.scatter(x_img[-1], y_img[-1], c='green', s=70, marker='o', label='Tail')
            
            ax.set_title(f"Camera {cam_idx} - Frame {frame_num}")
            ax.axis('off')
            ax.legend(loc='upper right', framealpha=0.4)
        
        plt.suptitle(f"3D Reconstruction Overlay - Frame {frame_num}", fontsize=16)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting reconstruction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print shapes / plot reconstruction.')
    parser.add_argument('--file', type=str, 
                        default='data/reconstruction_xyz/trial=037_reconstruction=6269c8e489239b727e0634fc.npz')
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--images-path', type=str)
    parser.add_argument('--camera-projections', type=str, default='xy,xz,yz')
    parser.add_argument('--flip-x', type=str, default='')
    parser.add_argument('--flip-y', type=str, default='')
    parser.add_argument('--no-robust-scale', action='store_true')
    args = parser.parse_args()
    
    print_npz_shapes(args.file)
    if args.frame is not None:
        if args.plot:
            proj = _parse_projections(args.camera_projections)
            fx = _parse_int_set(args.flip_x)
            fy = _parse_int_set(args.flip_y)
            plot_reconstruction_on_images(args.file, args.frame, args.images_path,
                                          projections=proj, flip_x=fx, flip_y=fy,
                                          robust=not args.no_robust_scale)
        else:
            print_frame(args.file, args.frame)
