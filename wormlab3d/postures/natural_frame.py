import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

PSI_ESTIMATION_KAPPA_THRESHOLD_DEFAULT = 0.5


def normalise(v):
    return v / norm(v, axis=-1, keepdims=True)


def an_orthonormal(x):
    if abs(x[0]) < 1e-20:
        return np.array([1., 0., 0.])
    if abs(x[1]) < 1e-20:
        return np.array([0., 1., 0.])

    X = np.array([x[1], -x[0], 0.])
    return X / norm(X)


def rotate(v: np.ndarray, theta: float):
    """
    Rotate a complex vector by angle theta.
    """
    return np.exp(1j * theta) * v


def align_complex_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Align vector a to vector b.
    """
    opt_angle = -np.angle(np.dot(a, b.conj()))
    a_aligned = rotate(a, opt_angle)
    return a_aligned


class NaturalFrame:
    def __init__(
            self,
            X: np.ndarray,
            length: float = 1.,
            X0: np.ndarray = None,
            T0: np.ndarray = None,
            M0: np.ndarray = None,
            threshold: float = PSI_ESTIMATION_KAPPA_THRESHOLD_DEFAULT
    ):
        """
        Takes either a midline and converts it into the Bishop frame or a Bishop frame representation
        and converts it into a midline.

        For the conversion from Bishop frame, length, position (mid-point, X0), orientation
        (initial tangent direction, T0, and initial M1 direction, M0) may be provided as none
        of this is available from the complex curvature alone.

        Changes to psi are ignored when the curvature is below given threshold.
        (This gives a nicer result at no cost since m1/m2 are not affected.)

        Propagates instance variables:
            X_pos     - The midline coordinates.
            T         - Tangent to the midline.
            K         - The vector curvature.
            M1, M2    - Normalised frame vectors at each point along the worm.
            m1, m2    - Magnitudes of the curvature in the M1 and M2 directions respectively.
            mc        - Complex frame representation = m1 + j.m2
            kappa     - Absolute scalar curvature.
            psi       - Internal rotation angle.
            pca       - Fitted PCA instance used to estimate psi.
        """
        self.X = X
        self.N = len(X)
        self.threshold = threshold

        if X.shape[-1] == 3:
            self.X = X.astype(np.float64)
            self.X_pos = X - self.X.min(axis=0)
            self._calculate_tangent_and_curvature()
            self._calculate_pca()
            self._calculate_frame_components()

        elif X.ndim == 1 and np.iscomplexobj(X):
            self.X = X.astype(np.complex128)
            self._convert_from_bishop(length, X0, T0, M0)
            self._calculate_pca()

        else:
            raise RuntimeError('Unrecognised input!')

        self._smooth_psi()

    @property
    def length(self) -> float:
        return np.linalg.norm(self.X_pos[:-1] - self.X_pos[1:], axis=1).sum()

    @property
    def e0(self) -> np.ndarray:
        return self.T

    @property
    def e1(self) -> np.ndarray:
        return self.M1

    @property
    def e2(self) -> np.ndarray:
        return self.M2

    def _calculate_tangent_and_curvature(self):
        """
        Calculate the tangent and the curvature from the midline coordinates.
        """

        # Distance between vertices
        q = norm(self.X_pos[1:] - self.X_pos[:-1], axis=1)
        q = np.r_[q[0], q, q[-1]]

        # Average distances over both neighbours
        spacing = (q[:-1] + q[1:]) / 2
        locs = np.cumsum(spacing)

        # Tangent is normalised gradient of curve
        T = np.gradient(self.X_pos, locs, axis=0, edge_order=1)
        T_norm = norm(T, axis=-1, keepdims=True)
        self.T = T / T_norm

        # Curvature is gradient of tangent
        K = np.gradient(self.T, 1 / (self.N - 1), axis=0, edge_order=1)
        self.K = K / T_norm

    def _calculate_pca(self):
        """
        Calculate some reference directions using the whole body.
        """
        pca = PCA()
        pca.fit(self.X_pos)
        self.pca = pca

    def _calculate_frame_components(self):
        """
        Calculate the frame components; M1, M2, their scalar magnitudes, m1, m2, the
        complex representations mc = m1 + j.m2 and from that the scalar curvature and twist.
        """

        # Use PCA components as reference
        ref = self.pca.components_

        # M1/M2 should change smoothly along the body, so start in place of maximum curvature and work outwards
        start_idx = np.argmax(norm(self.K, axis=1)[1:-1])
        M1 = np.zeros_like(self.T)
        M1_tilde = ref[1] - np.dot(self.T[start_idx], ref[1]) * self.T[start_idx]
        M1[start_idx] = normalise(M1_tilde)
        for i in range(start_idx - 1, -1, -1):
            M1_tilde = M1[i + 1] - np.dot(self.T[i], M1[i + 1]) * self.T[i]
            M1[i] = normalise(M1_tilde)
        for i in range(start_idx + 1, self.N):
            M1_tilde = M1[i - 1] - np.dot(self.T[i], M1[i - 1]) * self.T[i]
            M1[i] = normalise(M1_tilde)
        self.M1 = M1

        # M2 is the remaining orthogonal vector
        M2 = np.cross(self.T, self.M1)
        self.M2 = normalise(M2)

        # Project curvature onto frame
        self.m1 = np.einsum('ni,ni->n', self.M1, self.K)
        self.m2 = np.einsum('ni,ni->n', self.M2, self.K)

        # Convert m1/m2 into a complex number then the scalar curvature and twist fall out
        self.mc = self.m1 + 1.j * self.m2
        self.kappa = np.abs(self.mc)
        self.psi = np.angle(self.mc)

    def _convert_from_bishop(
            self,
            scale: float = 1.,
            X0: np.ndarray = None,
            T0: np.ndarray = None,
            M0: np.ndarray = None,
    ):
        """
        Convert a Bishop frame representation to recover the full position and components.
        """
        self.mc = self.X
        self.m1 = np.real(self.mc)
        self.m2 = np.imag(self.mc)
        self.kappa = np.abs(self.mc)
        self.psi = np.angle(self.mc)

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
            M0 = M0 / norm(M0, keepdims=True)

        # Initialise the components
        shape = (self.N, 3)
        X = np.zeros(shape)
        T = np.zeros(shape)
        M1 = np.zeros(shape)
        M2 = np.zeros(shape)
        X[0] = X0
        T[0] = T0
        M1[0] = M0
        M2[0] = np.cross(T[0], M1[0])
        h = scale / (self.N - 1)

        # Calculate the frame components (X/T/M1/M2)
        for i in range(1, self.N):
            k1 = self.m1[i]
            k2 = self.m2[i]

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

        self.X_pos = X
        self.K = self.m1[:, None] * M1 + self.m2[:, None] * M2
        self.T = T
        self.M1 = M1
        self.M2 = M2

    def _smooth_psi(self):
        """
        Linearly interpolate angles in regions where the curvature is below threshold.
        """
        below_threshold = False
        for i in range(len(self.kappa)):
            if not below_threshold and self.kappa[i] < self.threshold / self.length:
                below_threshold = True
                start_idx = i
                start_psi = self.psi[i - 1]
            elif below_threshold and self.kappa[i] >= self.threshold / self.length:
                below_threshold = False
                if start_idx == 0:
                    self.psi[0:i] = self.psi[i]
                else:
                    self.psi[start_idx:i] = np.linspace(start_psi, self.psi[i], i - start_idx + 2)[1:-1]
        if below_threshold:
            self.psi[start_idx:] = start_psi

    def non_planarity(self) -> float:
        """
        Compute the non-planarity of the curve.
        """
        r = self.pca.explained_variance_ratio_
        return r[2] / np.sqrt(r[1] * r[0])

    def chirality(self) -> float:
        """
        Compute the chirality of the curve.
        """

        # Try to align v1 with the direction of the curve
        u = normalise(self.X_pos[-1] - self.X_pos[0])
        v1, v2, v3 = self.pca.components_
        if np.dot(u, v1) < 0:
            v1 *= -1

        # Rotate points to align with the principal components.
        R = np.stack([v1, v2, v3], axis=1)
        Xt = np.einsum('ij,bj->bi', R.T, self.X_pos)

        # Use the difference vectors between adjacent points and ignore first coordinate.
        diff = (Xt[1:] - Xt[:-1])[:, 1:]

        # Convert into polar coordinates
        r = np.linalg.norm(diff, axis=-1)
        theta = np.unwrap(np.arctan2(*diff.T))

        # Weight the angular changes by the radii and sum to give chirality measure.
        r = (r[1:] + r[:-1]) / 2
        c = np.sum(r * (theta[1:] - theta[:-1]))

        # Correct for reflections
        if np.allclose(np.linalg.det(R), -1):
            c *= -1

        return c

    def surface(
            self,
            N_theta: int = 32,
            radius: float = 0.02,
            taper: float = 0.25,
            shape_k1: float = 1.5,
            shape_k2: float = 1,
            use_centred: bool = False
    ) -> np.ndarray:
        """
        Generate a tapered cylindrical mesh.
        """
        from scipy.interpolate import interp1d
        assert 0 <= taper <= 0.5, f'taper out of bounds! {taper}'
        N_taper = int(self.N * taper)
        theta = np.linspace(0, 2 * np.pi, N_theta)

        # Define the radius as a function of midline
        if N_taper > 0:
            s = np.linspace(0, np.pi / 2, N_taper)
            x = N_taper * np.cos(s)**shape_k1
            y = radius * np.sin(s)**shape_k2
            t_idxs = np.linspace(0, N_taper, N_taper)

            # Resample to get equally spaced points
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
            slopes = f(t_idxs)
            r = np.concatenate([
                slopes[::-1],
                np.ones(self.N - 2 * N_taper) * radius,
                slopes
            ])
        else:
            r = np.ones(self.N) * radius

        # Generate cylinder
        cylinder = np.einsum('i,jk->ijk', np.cos(theta), self.M1) \
                   + np.einsum('i,jk->ijk', np.sin(theta), self.M2)

        # Generate surface
        if use_centred or np.iscomplexobj(self.X):
            surface = self.X_pos + r[None, :, None] * cylinder
        else:
            surface = self.X + r[None, :, None] * cylinder

        # Map the curvature to the surface
        K_surf = np.einsum('ij,kij->ki', self.K, cylinder)

        return surface, K_surf
