import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes

show_plots = True
save_plots = True


def equal_aspect_ratio(ax: Axes):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))


# Load the reconstruction
data = np.load('../reconstruction_xyz/trial=%%TRIAL_ID%%_reconstruction=%%RECONSTRUCTION_ID%%.npz')
X = data['X']  # shape: (n_frames, n_body_points, 3)
N = X.shape[1]

# Construct colours
colours = np.linspace(0, 1, len(X))
cmap = plt.get_cmap('viridis_r')

# Plot the 3D trajectory using the centre-of-mass of the body points
X_com = X.mean(axis=1)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(*X_com.T, c=colours, cmap=cmap, s=10, alpha=0.4, zorder=-1)
ax.set_title('3D centre-of-mass trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
equal_aspect_ratio(ax)
fig.tight_layout()
if save_plots:
    plt.savefig('com_trajectory_3d.png')
if show_plots:
    plt.show()

# Plot a section of the head, midpoint and tail trajectories
start_idx = 0
end_idx = 100
X_head = X[start_idx:end_idx, 0]
X_mid = X[start_idx:end_idx, int(N / 2)]
X_tail = X[start_idx:end_idx, -1]
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, (X_part, title) in enumerate(zip([X_head, X_mid, X_tail], ['Head', 'Midpoint', 'Tail'])):
    for j, label in enumerate(['x', 'y', 'z']):
        ax = axes[j, i]
        ax.plot(X_part[:, j])
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel(label)
fig.tight_layout()
if save_plots:
    plt.savefig('coordinate_traces.png')
if show_plots:
    plt.show()

# Show some postures
frames = np.arange(9) * 100
cmap = cm.get_cmap('plasma')
fc = cmap((np.arange(N) + 0.5) / N)
fig = plt.figure(figsize=(10, 10))
k = 0
for i in range(3):
    for j in range(3):
        frame_idx = int(frames[k])
        Xk = X[frame_idx]
        ax = fig.add_subplot(3, 3, k + 1, projection='3d')
        ax.scatter(*Xk.T, c=fc, s=20, alpha=0.8)
        equal_aspect_ratio(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f'Frame {frame_idx}')
        k += 1
fig.tight_layout()
if save_plots:
    plt.savefig('reconstruction_postures.png')
if show_plots:
    plt.show()
