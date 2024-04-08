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

# Construct colours
colours = np.linspace(0, 1, len(X))
cmap = plt.get_cmap('viridis_r')

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# Scatter the vertices
ax.scatter(x, y, z, c=colours, cmap=cmap, s=10, alpha=0.4, zorder=-1)

# Draw lines connecting points
points = X[:, None, :]
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
ax.add_collection(lc)

# Setup axis
ax.set_title('Tracked 3D trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
equal_aspect_ratio(ax)
fig.tight_layout()

if save_plots:
    plt.savefig('tracked_3d.png')
if show_plots:
    plt.show()
