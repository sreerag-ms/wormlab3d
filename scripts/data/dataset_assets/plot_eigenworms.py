import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes

from cpca import load_cpca_from_components

show_plots = True
save_plots = True


def equal_aspect_ratio(ax: Axes):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))


def normalise(v: np.ndarray) -> np.ndarray:
    """Normalise an array along its final dimension."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def an_orthonormal(x: np.ndarray) -> np.ndarray:
    """Generate an arbitrary normalised vector orthogonal to x."""
    if abs(x[0]) < 1e-20:
        return np.array([1., 0., 0.])
    if abs(x[1]) < 1e-20:
        return np.array([0., 1., 0.])
    X = np.array([x[1], -x[0], 0.])
    return normalise(X)


def convert_bishop_to_xyz(
        X_bishop: np.ndarray,
        X0: np.ndarray = None,
        T0: np.ndarray = None,
        M0: np.ndarray = None,
):
    """
    Convert a Bishop frame representation to xyz components.
    Implements a simple euler solver to integrate the Bishop equations.
    """
    N = X_bishop.shape[0]
    m1 = np.real(X_bishop)
    m2 = np.imag(X_bishop)

    # Position offset
    if X0 is None:
        X0 = np.array([0, 0, 0])
    else:
        assert X0.shape == (3,)

    # Orientation - initial tangent direction
    if T0 is None:
        T0 = np.array([1, 0, 0])
    else:
        assert T0.shape == (3,)
        T0 = normalise(T0)

    # Orientation - initial M1 direction
    if M0 is None:
        M0 = an_orthonormal(T0)
    else:
        # Orthogonalise M0 against T0 and normalise
        assert M0.shape == (3,)
        M0 = M0 - np.dot(T0, M0) * T0
        M0 = M0 / np.linalg.norm(M0, keepdims=True)

    # Initialise the components
    shape = (N, 3)
    X = np.zeros(shape)
    T = np.zeros(shape)
    M1 = np.zeros(shape)
    M2 = np.zeros(shape)
    X[0] = X0
    T[0] = T0
    M1[0] = M0
    M2[0] = np.cross(T[0], M1[0])
    h = 1 / (N - 1)

    # Calculate the components (X/T/M1/M2)
    for i in range(1, N):
        k1 = m1[i]
        k2 = m2[i]

        dTds = k1 * M1[i - 1] + k2 * M2[i - 1]
        dM1ds = -k1 * T[i - 1]
        dM2ds = -k2 * T[i - 1]

        T_tilde = T[i - 1] + h * dTds
        M1_tilde = M1[i - 1] + h * dM1ds
        M2_tilde = M2[i - 1] + h * dM2ds

        X[i] = X[i - 1] + h * T[i - 1]
        T[i] = normalise(T_tilde)
        M1[i] = normalise(M1_tilde)
        M2[i] = normalise(M2_tilde)

    return X


# Load the eigenworms
ew_data = np.load('../reconstruction_eigenworms/eigenworms_%%EIGENWORMS_ID%%.npz')
ew_mean = ew_data['mean']
ew_components = ew_data['components']

# Instantiate the Complex PCA object
cpca = load_cpca_from_components(ew_components, ew_mean)
print(type(cpca))

# Load the reconstruction encoded in the eigenworms space
data = np.load(
    '../reconstruction_eigenworms/trial=%%TRIAL_ID%%_reconstruction=%%RECONSTRUCTION_ID%%_eigenworms=%%EIGENWORMS_ID%%.npz')
Z = data['X']
print(Z.shape)  # shape: (n_frames, n_components)
print(Z.dtype)  # NOTE: dtype=complex!

# Inverse transform to get Bishop-frame curves, defined along the body
X_bishop = cpca.inverse_transform(Z)
N = X_bishop.shape[1]
print(X_bishop.shape)  # shape: (n_frames, n_body_points)
print(X_bishop.dtype)  # NOTE: dtype=complex!

# Plot the eigenworm components over time
start_idx = 0
end_idx = 500
plot_n_components = 5
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax = axes[0]
for i in range(plot_n_components):
    axes[0].plot(np.abs(Z[start_idx:end_idx, i]), label=f'Component {i}')
    axes[1].plot(np.angle(Z[start_idx:end_idx, i]), label=f'Component {i}')
axes[0].set_title('$|Z|$')
axes[1].set_title('$\\arg{Z}$')
axes[1].set_xlabel('Frame #')
axes[0].set_ylabel('Magnitude')
axes[1].set_ylabel('Complex argument')
axes[0].legend()
fig.tight_layout()
if save_plots:
    plt.savefig('eigenworm_traces.png')
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
        Xk = convert_bishop_to_xyz(X_bishop[frame_idx])
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
    plt.savefig('eigenworm_postures.png')
if show_plots:
    plt.show()
