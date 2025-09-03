import numpy as np
from PIL import Image
import cv2
import os

start_frame = 0
end_frame = 1000

channel_frames = [[] for _ in range(3)]

for frame_num in range(start_frame, end_frame + 1):
    filename = f'data/prepared_images/037/{frame_num:06d}.npz'
    
    try:
        data = np.load(filename)
        
        images = data['images']
        
        for i in range(images.shape[0]):
            scaled_image = ((images[i] * 255)).astype(np.uint8)
            channel_frames[i].append(scaled_image)
            
        print(f"Processed frame {frame_num}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
        continue
    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")
        continue

output_dir = 'output/videos'
os.makedirs(output_dir, exist_ok=True)

for channel_idx, frames in enumerate(channel_frames):
    if not frames:
        print(f"No frames for channel {channel_idx}")
        continue
        
    output_filename = f'{output_dir}/channel_{channel_idx}_frames_{start_frame}_to_{end_frame}.mp4'
    
    height, width = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, 15, (width, height), True)
    
    for idx, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        actual_frame_num = start_frame + idx
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        text = str(actual_frame_num)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x = width - text_width - 10
        text_y = text_height + 10
        
        cv2.putText(frame_bgr, text, (text_x, text_y), font, font_scale, color, thickness)
        
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"Saved video: {output_filename}")

print("Video creation complete!")
