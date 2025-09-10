import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

#  data files
pred_file = "1_convolution_method/comparisons/datasets/ht_detection_results_600.csv"
gt_file = "1_convolution_method/comparisons/datasets/wt_3d_endpoints.csv"

pred_data = pd.read_csv(pred_file)
gt_data = pd.read_csv(gt_file)

results = []

for frame_id in [2]:
    for frame_position in pred_data[pred_data['frame_id'] == frame_id]['frame_position'].unique():
        # Get predicted endpoints
        pred_row = pred_data[(pred_data['frame_id'] == frame_id) & (pred_data['frame_position'] == frame_position)].iloc[0]
        end1 = np.array([pred_row['x_end1'], pred_row['y_end1']])
        end2 = np.array([pred_row['x_end2'], pred_row['y_end2']])
        
        gt_row = gt_data[(gt_data['frame_id'] == frame_id) & (gt_data['frame_position'] == frame_position)].iloc[0]
        gt_head = np.array([gt_row['x_head'], gt_row['y_head']])
        gt_tail = np.array([gt_row['x_tail'], gt_row['y_tail']])
        
        # Calculate distances
        dist_head_to_end1 = np.linalg.norm(end1 - gt_head)
        dist_head_to_end2 = np.linalg.norm(end2 - gt_head)
        dist_tail_to_end1 = np.linalg.norm(end1 - gt_tail)
        dist_tail_to_end2 = np.linalg.norm(end2 - gt_tail)
        
        # Determine endopoint by shortest distance
        total_dist_1 = dist_head_to_end1 + dist_tail_to_end2
        total_dist_2 = dist_head_to_end2 + dist_tail_to_end1
        
        if total_dist_1 <= total_dist_2:
            head_distance = dist_head_to_end1
            tail_distance = dist_tail_to_end2
            head_endpoint = 'end1'
            tail_endpoint = 'end2'
        else:
            head_distance = dist_head_to_end2
            tail_distance = dist_tail_to_end1
            head_endpoint = 'end2'
            tail_endpoint = 'end1'
        
        avg_distance = (head_distance + tail_distance) / 2
        
        # Append to results
        results.append({
            'frame_id': frame_id,
            'frame_position': frame_position,
            'head_distance': head_distance,
            'tail_distance': tail_distance,
            'avg_distance': avg_distance,
            'head_endpoint': head_endpoint,
            'tail_endpoint': tail_endpoint,
            'euclidean_error': math.sqrt(head_distance**2 + tail_distance**2)
        })

results_df = pd.DataFrame(results)

overall_stats = {
    'mean_head_distance': results_df['head_distance'].mean(),
    'mean_tail_distance': results_df['tail_distance'].mean(),
    'mean_avg_distance': results_df['avg_distance'].mean(),
    'std_head_distance': results_df['head_distance'].std(),
    'std_tail_distance': results_df['tail_distance'].std(),
    'std_avg_distance': results_df['avg_distance'].std(),
    'median_head_distance': results_df['head_distance'].median(),
    'median_tail_distance': results_df['tail_distance'].median(),
    'median_avg_distance': results_df['avg_distance'].median(),
    'max_head_distance': results_df['head_distance'].max(),
    'max_tail_distance': results_df['tail_distance'].max(),
    'max_avg_distance': results_df['avg_distance'].max(),
    'min_head_distance': results_df['head_distance'].min(),
    'min_tail_distance': results_df['tail_distance'].min(),
    'min_avg_distance': results_df['avg_distance'].min(),
}

script_path = Path(__file__).resolve()
outputs_dir = script_path.parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(outputs_dir / 'distance_analysis_results.csv', index=False)

# Save overall statistics
stats_df = pd.DataFrame([overall_stats])
stats_df.to_csv(outputs_dir / 'distance_analysis_stats.csv', index=False)

mean_by_position = results_df.groupby('frame_position')['avg_distance'].mean().sort_index()

plt.figure(figsize=(8, 6))
plt.plot(mean_by_position.index, mean_by_position.values, marker='o')
plt.xlabel('Frame Position')
plt.ylabel('Mean Average Distance (pixels)')
plt.title('Mean Detection Error vs Frame Position')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_dir / 'mean_distance_vs_frame_position.png')
plt.close()

print(f"Analysis complete. Results saved to:")
print(f"  - {outputs_dir / 'distance_analysis_results.csv'}")
print(f"  - {outputs_dir / 'distance_analysis_stats.csv'}")
print(f"  - {outputs_dir / 'mean_distance_vs_frame_position.png'}")
print(f"Overall statistics:")
for key, value in overall_stats.items():
    print(f"  {key}: {value:.4f}")