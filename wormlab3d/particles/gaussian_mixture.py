import numpy as np
from numbers import Number

import torch
from scipy.stats import rv_continuous, norm
from torch.distributions import Distribution, Normal
from torch.distributions.utils import broadcast_all


class GaussianMixtureTorch(Distribution):
    arg_constraints = {}

    def __init__(self, weights, loc, scale, validate_args=None):
        assert weights.ndim == 1
        assert loc.ndim == 1
        assert scale.ndim == 1
        assert weights.shape == loc.shape == scale.shape
        self.n_components = loc.shape[0]
        self.weights = weights / weights.sum()
        self.loc = loc
        self.scale = scale
        batch_shape = torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)

        # Create the individual Gaussians
        self.components = []
        for i in range(self.n_components):
            self.components.append(Normal(loc=self.loc[i], scale=self.scale[i], validate_args=validate_args))


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        N = shape.numel()

        with torch.no_grad():
            # Draw samples from all the components
            component_samples = torch.zeros((*sample_shape, self.n_components))
            for i in range(self.n_components):
                component_samples[:, i] = self.components[i].sample(sample_shape)

            # Select the components with weighted probabilities
            idxs = torch.stack([
                torch.arange(N),
                torch.multinomial(self.weights, N, replacement=True).reshape(shape)
            ], dim=0)

            samples = component_samples[idxs.chunk(chunks=N, dim=0)]
            return samples




class GaussianMixtureScipy(rv_continuous):
    def __init__(self, weights, loc, scale):
        assert weights.ndim == 1
        assert loc.ndim == 1
        assert scale.ndim == 1
        assert weights.shape == loc.shape == scale.shape
        self.n_components = loc.shape[0]
        self.weights = weights / weights.sum()
        self.loc = loc
        self.scale = scale

        # Create the individual Gaussians
        self.components = []
        for i in range(self.n_components):
            self.components.append(norm(self.loc[i], self.scale[i]))

    def pdf(self, x):
        ys = np.zeros_like(x)
        for i in range(self.n_components):
            ys += self.weights[i] * self.components[i].pdf(x)
        return ys
