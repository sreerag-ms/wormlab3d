import pandas as pd
import json
import os
import re

def clean_dataset(input_path=None, output_path=None):
    """
    Cleans the worm dataset by:
    """    
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    df['frame_position'] = df['img'].str.extract(r'^(\d+)_image_')[0]
    df['frame_id'] = df['img'].str.extract(r'_image_(\d+)\.jpg$')[0]
    
    df['frame_position'] = df['frame_position'].astype(int)
    df['frame_id'] = df['frame_id'].astype(int)
    
    def extract_coordinates(kp_str):
        try:
            json_str = kp_str.replace('""', '"')
            
            keypoints = json.loads(json_str)
            
            head_x = None
            head_y = None
            tail_x = None
            tail_y = None
            
            for kp in keypoints:
                if "keypointlabels" in kp and "head" in kp["keypointlabels"]:
                    head_x = kp.get("x")
                    head_y = kp.get("y")
                elif "keypointlabels" in kp and "tail" in kp["keypointlabels"]:
                    tail_x = kp.get("x")
                    tail_y = kp.get("y")
            
            return pd.Series([head_x, head_y, tail_x, tail_y])
        except Exception as e:
            print(f"Error parsing keypoints in row: {e}")
            return pd.Series([None, None, None, None])
    
    df[['x_head', 'y_head', 'x_tail', 'y_tail']] = df['kp-1'].apply(extract_coordinates)
    
    result_df = df[['id', 'frame_position', 'frame_id', 'x_head', 'y_head', 'x_tail', 'y_tail']]
    
    result_df = result_df.sort_values(by=['frame_position', 'frame_id'])
    
    result_df.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset saved to {output_path}")
    return result_df

if __name__ == "__main__":
    input_path = "data/dataset_4.csv"
    output_path = "data/cleaned_dataset_4.csv"
    cleaned_df = clean_dataset(input_path=input_path, output_path=output_path)

