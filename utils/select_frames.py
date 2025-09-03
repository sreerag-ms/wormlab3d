import cv2
import os

def select_specific_frame(video_path, output_dir, frame_number=2000):

    """
    Selects a specific frame from a video and saves it as an image.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save the output image
        frame_number (int): Specific frame number to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames in video: {total_frames}")
    print(f"Video FPS: {fps}")
    
    if frame_number >= total_frames:
        print(f"Error: Requested frame ({frame_number}) exceeds total frames in video ({total_frames})")
        cap.release()
        return
    
    timestamp_seconds = frame_number / fps
    minutes = int(timestamp_seconds // 60)
    seconds = timestamp_seconds % 60
    print(f"Frame {frame_number} timestamp: {minutes}m {seconds:.2f}s")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    
    if ret:
        output_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_number} to {output_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")
    
    cap.release()

if __name__ == "__main__":
    video_path = "data/trial=037/trial=037_tracking_camera=2.mp4"
    output_dir = "data/images"
    
    select_specific_frame(video_path, output_dir)
    print("Done!")
