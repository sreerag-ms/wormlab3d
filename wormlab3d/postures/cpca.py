import numbers

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


def complex_covariance_matrix(X):
    """
    Complex covariance matrix. Same as np.cov on real arrays.
    X.shape: (n_samples, n_features)
    """
    return np.dot(X.T, np.conj(X)) / (X.shape[0] - 1.)


class CPCA(PCA):
    """
    Complex-PCA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.svd_solver == 'auto', 'Alternative SVD solvers not implemented!'

    def fit_transform(self, X, y=None):
        raise NotImplementedError()

    def _fit(self, X: np.ndarray):
        """
        Override to ensure complex data types and only a single solver option.
        """

        # Raise an error for sparse input.
        if issparse(X):
            raise TypeError('CPCA does not support sparse input.')
        X = self._validate_data(X)

        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            assert type(self.n_components) == int
            n_components = self.n_components

        # Handle svd_solver
        if self.svd_solver != 'auto':
            raise ValueError('Only the auto solver is supported!')
        self._fit_svd_solver = 'full'

        return self._fit_full(X, n_components)

    def _fit_full(self, X: np.ndarray, n_components: int):
        """
        Fit the model by computing eigenvalue decomposition on the complex covariance matrix of X.
        """
        n_samples, n_features = X.shape

        if not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                f'n_components={n_components} must be between 0 and min(n_samples, n_features)={min(n_samples, n_features)}')
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(f'n_components={n_components} must be of type int. Found type={type(n_components)}')

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # Do decomposition
        cc = complex_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eigh(cc)

        # Sort the eigenvalues/vectors in descending order of variance contribution
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        components_ = eigenvectors.T

        # Get variance explained by singular values
        total_var = eigenvalues.sum()
        explained_variance_ = eigenvalues
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = eigenvalues.copy()

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return eigenvalues, eigenvectors

    def _fit_truncated(self, X, n_components, svd_solver):
        raise NotImplementedError()

    def score_samples(self, X):
        raise NotImplementedError()

    def score(self, X, y=None):
        raise NotImplementedError()

    def _more_tags(self):
        return {'preserves_dtype': [np.complex128, np.complex64]}

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = self._validate_data(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.conj().T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def _validate_data(self, X, y='no_validation', reset=True,
                       validate_separately=False, **check_params):
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional.')
        if not np.iscomplexobj(X):
            raise ValueError('Input data must be complex.')
        if self.copy:
            X = X.copy()
        return X


def testit(phase):
    s = phase

    a = np.array([1j, 0., 1.])
    b = s * np.array([0., 1., 0])
    c = np.array([-1., 0, 0.])
    d = np.array([0, -1, 0.0000])
    e = s * np.array([1., 0., -1.])
    f = np.array([-1., 0., 0.0000]) / s

    X = np.array([a, b, c, d, e, f])

    p = CPCA()
    p.fit_full(X)
    print(s)
    print(p.explained_variance_)
    print(p.components_[0])


if __name__ == '__main__':
    from math import cos, sin

    for i in range(10):
        s = sin(i) * 1j + cos(i)
        testit(s)
