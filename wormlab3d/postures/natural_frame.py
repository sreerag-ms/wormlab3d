import numpy as np
from sklearn.decomposition import PCA

PSI_ESTIMATION_KAPPA_THRESHOLD_DEFAULT = 0.5


class NaturalFrame:
    def __init__(self, X: np.ndarray, threshold: float = PSI_ESTIMATION_KAPPA_THRESHOLD_DEFAULT):
        """
        Takes a midline X and converts it into the Bishop frame.
        Changes to psi are ignored when the curvature is below given threshold.
        (This gives a nicer result at no cost since m1/m2 are not affected.)

        Propagates instance variables:
            X
            T, M1, M2 - the normalised frame at each point along the worm.
            m1, m2    - the magnitudes of the curvature in the M1 and M2 directions respectively.
            kappa     - absolute scalar curvature
            psi       - internal rotation angle (based on PCA estimate)
            pca       - fitted PCA instance used to estimate psi
        """
        assert X.shape[-1] == 3
        self.X = X
        self.X_pos = X - self.X.min(axis=0)
        self.N = len(X)
        self.threshold = threshold
        self._calculate_tangent_and_curvature()
        self._calculate_pca()
        self._calculate_frame_components()

    def _calculate_tangent_and_curvature(self):
        # Distance between vertices
        q = np.linalg.norm(self.X_pos[1:] - self.X_pos[:-1], axis=1)
        q = np.r_[q[0], q, q[-1]]

        # Average distances over both neighbours
        spacing = (q[:-1] + q[1:]) / 2
        locs = np.cumsum(spacing)

        # Tangent is gradient of curve
        self.T = np.gradient(self.X_pos, locs, axis=0, edge_order=1)

        # Curvature is gradient of tangent
        self.K = np.gradient(self.T, locs, axis=0, edge_order=1)

    def _calculate_pca(self):
        # Calculate some reference directions using the whole body
        pca = PCA()
        pca.fit(self.X_pos)
        self.pca = pca

    def _calculate_frame_components(self):
        # Use PCA components as reference
        ref = self.pca.components_

        # M1/M2 should change smoothly along the body, so start in place of maximum curvature and work outwards
        start_idx = np.argmax(np.linalg.norm(self.K, axis=1)[1:-1])
        M1 = np.zeros_like(self.T)
        M1[start_idx] = ref[1] - np.dot(self.T[start_idx], ref[1]) * self.T[start_idx]
        for i in range(start_idx - 1, -1, -1):
            M1[i] = M1[i+1] - np.dot(self.T[i], M1[i+1]) * self.T[i]
        for i in range(start_idx, self.N):
            M1[i] = M1[i-1] - np.dot(self.T[i], M1[i-1]) * self.T[i]
        self.M1 = M1

        # M2 is the remaining orthogonal vector
        M2 = np.cross(self.T, self.M1)
        self.M2 = M2 / np.linalg.norm(M2, axis=1)[:, None]

        # Project curvature onto frame
        self.m1 = np.einsum('ni,ni->n', self.M1, self.K)
        self.m2 = np.einsum('ni,ni->n', self.M2, self.K)

        # Find point of maximum curvature as it should give us most reliable estimate for psi
        self.mc = self.m1 + 1.j * self.m2
        self.kappa = np.abs(self.mc)
        self.psi = np.angle(self.mc)

        # Linearly interpolate angles in regions where the curvature is below threshold
        below_threshold = False
        for i in range(len(self.kappa)):
            if not below_threshold and self.kappa[i] < self.threshold:
                below_threshold = True
                start_idx = i
                start_psi = self.psi[i-1]
            elif below_threshold and self.kappa[i] >= self.threshold:
                below_threshold = False
                if start_idx == 0:
                    self.psi[0:i] = self.psi[i]
                else:
                    self.psi[start_idx:i] = np.linspace(start_psi, self.psi[i], i-start_idx+2)[1:-1]
        if below_threshold:
            self.psi[start_idx:] = start_psi
