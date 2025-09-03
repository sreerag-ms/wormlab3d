import numpy as np
from PIL import Image
import os
import glob
import cv2

output_base_dir = 'data/prepared_images_jpg'
os.makedirs(output_base_dir, exist_ok=True)

dataset_dir = os.path.join(output_base_dir, 'dataset_2_0815_0830')
os.makedirs(dataset_dir, exist_ok=True)

npz_files_pattern = 'data/prepared_images/037/0008*.npz'
npz_files = sorted(glob.glob(npz_files_pattern))

npz_files = [f for f in npz_files if 815 <= int(os.path.basename(f).split('.')[0]) < 830]

if not npz_files:
    print(f"No NPZ files found matching pattern: {npz_files_pattern}")
    exit(1)

print(f"Found {len(npz_files)} files to process: {[os.path.basename(f) for f in npz_files]}")

def create_videos_from_frames(npz_files, output_dir, fps=30):
    """
    Create 3 separate videos from the frames extracted from NPZ files.
    
    Args:
        npz_files: List of NPZ file paths
        output_dir: Directory to save the videos
        fps: Frames per second for the output videos
    """
    video_writers = [None, None, None]
    video_paths = [
        os.path.join(output_dir, 'camera_0.mp4'),
        os.path.join(output_dir, 'camera_1.mp4'),
        os.path.join(output_dir, 'camera_2.mp4')
    ]
    
    print(f"Creating videos at {fps} FPS...")
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            
            if 'images' in data.files:
                images = data['images']
                
                for i in range(images.shape[0]):
                    scaled_image = (255 - (images[i] * 255)).astype(np.uint8)
                    
                    if video_writers[i] is None:
                        height, width = scaled_image.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writers[i] = cv2.VideoWriter(
                            video_paths[i], fourcc, fps, (width, height), isColor=False
                        )
                        print(f"Initialized video writer for camera {i}: {video_paths[i]}")
                    
                    video_writers[i].write(scaled_image)
                    
        except Exception as e:
            print(f"Error processing {npz_file} for video creation: {e}")
    
    for i, writer in enumerate(video_writers):
        if writer is not None:
            writer.release()
            print(f"Video saved: {video_paths[i]}")
    
    print("Video creation complete!")

# Process each NPZ file
for npz_file in npz_files:
    frame_number = os.path.basename(npz_file).split('.')[0]
    
    try:
        data = np.load(npz_file)
        print(f"Processing {npz_file} - Keys: {data.files}")
        
        if 'images' in data.files:
            images = data['images']
            
            for i in range(images.shape[0]):
                scaled_image = (255 - (images[i] * 255)).astype(np.uint8)
                
                img = Image.fromarray(scaled_image, mode='L')
                
                output_filename = os.path.join(dataset_dir, f'{frame_number}_image_{i}.jpg')
                img.save(output_filename)
                print(f"Saved {output_filename}")
        else:
            print(f"Warning: 'images' key not found in {npz_file}")
    except Exception as e:
        print(f"Error processing {npz_file}: {e}")

print("Conversion complete!")

# Create videos from the processed frames
create_videos_from_frames(npz_files, dataset_dir, fps=30)
