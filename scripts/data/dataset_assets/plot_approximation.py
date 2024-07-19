import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection

show_plots = True
save_plots = True


def equal_aspect_ratio(ax: Axes):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))


# Load the trajectory
data = np.load('../tracking/trial=%%TRIAL_ID%%_tracking.npz')
X = data['X']
x, y, z = X.T

# Load the approximation details
with open('../tracking/trial=%%TRIAL_ID%%_approximation.json', 'r') as f:
    approx = json.load(f)

# Construct colours
colours = np.linspace(0, 1, len(X))
cmap = plt.get_cmap('viridis_r')

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# Scatter the vertices
ax.scatter(x, y, z, c=colours, cmap=cmap, s=5, alpha=0.2, zorder=-1)

# Draw lines connecting points
points = X[:, None, :]
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = Line3DCollection(segments, array=colours, cmap=cmap, alpha=0.2, zorder=-2)
ax.add_collection(lc)

# Add tumble points
tumble_idxs = np.array(approx['tumble_idxs'])
tumbles = X[tumble_idxs]
ax.scatter(*tumbles.T, color='blue', marker='x', s=500, linewidth=4, alpha=0.9, zorder=1)

# Add runs lines
tumbles = np.concatenate([X[0][None], tumbles, X[-1][None]])
points = tumbles[:, None, :]
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = Line3DCollection(segments, color='red', zorder=5, linewidth=4, linestyle='--', alpha=0.8)
ax.add_collection(lc)

# Setup axis
ax.set_title('Run and tumble approximation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
equal_aspect_ratio(ax)
fig.tight_layout()

if save_plots:
    plt.savefig('run_tumble_approx.png')
if show_plots:
    plt.show()
