import numpy as np
from scipy.spatial.transform import Rotation

from wormlab3d.toolkit.util import normalise
from wormlab3d.trajectories.util import calculate_rotation_matrix


def get_rotation_matrix(u, angle):
    # Convert rotation axis and angle to 3x3 rotation matrix
    # (See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
    R = np.cos(angle) * np.eye(3) \
        + np.sin(angle) * np.cross(u, -np.eye(3)) \
        + (1 - np.cos(angle)) * (np.outer(u, u))
    return R


def test_rotation_matrix():
    """
    Check that the explicit construction matches the scipy library.
    """
    angle = np.random.rand() * np.pi
    u = normalise(np.random.random(3))
    R1 = Rotation.from_rotvec(angle * u).as_matrix()
    R2 = get_rotation_matrix(u, angle)
    assert np.allclose(R1, R2)


def test_rotation_angle_decomposition():
    """
    Check that rotation angles can be recovered in an arbitrary basis.
    Adapted from: https://stackoverflow.com/questions/58504937/faster-approach-for-decomposing-a-rotation-to-rotations-around-arbitrary-orthogo
    """

    # Generate random basis
    A = Rotation.from_rotvec(normalise(np.random.random(3)) * np.random.rand() * np.pi).as_matrix()

    # Generate random rotation matrix
    t0 = np.random.rand() * np.pi
    t1 = np.random.rand() * np.pi
    t2 = np.random.rand() * np.pi
    R = Rotation.from_rotvec(A[:, 0] * t0) * Rotation.from_rotvec(A[:, 1] * t1) * Rotation.from_rotvec(A[:, 2] * t2)
    R = R.as_matrix()

    # Decompose rotation matrix R into the axes of A
    rp = Rotation.from_matrix(A.T @ R @ A)
    a2, a1, a0 = rp.as_euler('zyx')
    xp = A @ Rotation.from_rotvec(a0 * np.array([1, 0, 0])).as_matrix() @ A.T
    yp = A @ Rotation.from_rotvec(a1 * np.array([0, 1, 0])).as_matrix() @ A.T
    zp = A @ Rotation.from_rotvec(a2 * np.array([0, 0, 1])).as_matrix() @ A.T

    # Test that the generated matrix is equal to 'r' (should give 0)
    assert np.allclose(xp @ yp @ zp, R)

    # Test that the individual rotations preserve the axes (should give 0)
    assert np.allclose(xp @ A[:, 0], A[:, 0])
    assert np.allclose(yp @ A[:, 1], A[:, 1])
    assert np.allclose(zp @ A[:, 2], A[:, 2])

    return


def test_rotating_vector_to_axis():
    """
    Check that the rotation matrix will align two vectors.
    """
    a = np.random.randn(3)
    a = a / np.linalg.norm(a)
    b = np.random.randn(3)
    b = b / np.linalg.norm(b)
    R = calculate_rotation_matrix(a, b)
    assert np.allclose(R @ a, b)


if __name__ == '__main__':
    test_rotation_matrix()
    test_rotation_angle_decomposition()
    test_rotating_vector_to_axis()
