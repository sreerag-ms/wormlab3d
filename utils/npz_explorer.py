import numpy as np
from PIL import Image
import cv2
import os

start_frame = 1000
end_frame = 1600
base_path = '/Users/sreeragms/Desktop/prepared_images/037/037'
output_dir = 'output/images/frames_001000_001600'
os.makedirs(output_dir, exist_ok=True)

for frame_num in range(start_frame, end_frame + 1):
    npz_filename = f'{base_path}/{frame_num:06d}.npz'
    
    try:
        data = np.load(npz_filename)
        images = data['images']
        
        for i in range(images.shape[0]):
            scaled_image = ((images[i] * 255)).astype(np.uint8)
            output_filename = f'{output_dir}/{frame_num:06d}_{i}.png'
            
            img = Image.fromarray(scaled_image)
            img.save(output_filename)
            print(f"Saved: {output_filename}")
            
    except FileNotFoundError:
        print(f"File not found: {npz_filename}")
        continue
    except Exception as e:
        print(f"Error processing file {npz_filename}: {e}")
        continue

print("Image extraction complete!")
