import numpy as np


def euler_angles_to_rotation_matrix(alpha, beta, gamma):
    cos = np.cos
    sin = np.sin

    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_g = cos(gamma)

    sin_a = sin(alpha)
    sin_b = sin(beta)
    sin_g = sin(gamma)

    # https://en.wikipedia.org/wiki/Rotation_matrix
    R = np.stack([
        np.stack([
            cos_a * cos_b,
            cos_a * sin_b * sin_g - sin_a * cos_g,
            cos_a * sin_b * cos_g + sin_a * sin_g
        ], axis=-1),
        np.stack([
            sin_a * cos_b,
            sin_a * sin_b * sin_g + cos_a * cos_g,
            sin_a * sin_b * cos_g - cos_a * sin_g
        ], axis=-1),
        np.stack([
            -sin_b,
            cos_b * sin_g,
            cos_b * cos_g
        ], axis=-1)
    ], axis=-2)

    return R


def rotate(points, alpha, beta, gamma):
    R = euler_angles_to_rotation_matrix(alpha, beta, gamma)
    points_rotated = np.einsum('ij,pj->pi', R, points)
    return points_rotated


def test():
    points = np.random.randn(30,3)
    abg = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=3)
    R = euler_angles_to_rotation_matrix(alpha=abg[0], beta=abg[1], gamma=abg[2])
    points_rotated = np.einsum('ij,pj->pi', R, points)
    points_unrotated = np.einsum('ij,pj->pi', R.T, points_rotated)

    assert np.allclose(points, points_unrotated)

if __name__=='__main__':
    test()
