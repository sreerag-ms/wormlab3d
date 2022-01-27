import datetime
import os
from pathlib import Path

import numpy as np
from mongoengine import *

from wormlab3d import EIGENWORMS_PATH
from wormlab3d.postures.cpca import CPCA


class Eigenworms(Document):
    created = DateTimeField(required=True, default=datetime.datetime.now)
    updated = DateTimeField(required=True, default=datetime.datetime.now)
    dataset = ReferenceField('Dataset')
    reconstruction = ReferenceField('Reconstruction')

    n_components = IntField(required=True)
    n_samples = IntField(required=True)
    n_features = IntField(required=True)
    explained_variance = ListField(FloatField())
    explained_variance_ratio = ListField(FloatField())
    singular_values = ListField(FloatField())

    meta = {
        'indexes': [
            'dataset',
            'reconstruction',
            {
                'fields': ['dataset', 'reconstruction', 'n_components', 'n_samples', 'n_features'],
                'unique': True
            },
        ]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = None
        self.components = None
        self.cpca = None

    @property
    def components_path(self) -> Path:
        if self.dataset is not None:
            dest = EIGENWORMS_PATH / 'datasets' / f'{self.dataset.id}_{self.id}.npz'
        elif self.reconstruction is not None:
            dest = EIGENWORMS_PATH / 'reconstructions' / f'{self.reconstruction.id}_{self.id}.npz'
        return dest

    def __getattribute__(self, k):
        if k not in ['mean', 'components', 'cpca']:
            return super().__getattribute__(k)

        # Check if the variable has been defined or loaded already
        v = super().__getattribute__(k)
        if v is not None:
            return v

        if k == 'mean':
            # If not then try to load it from the filesystem
            try:
                mean = np.load(self.components_path)['mean']
                self.mean = mean
                return mean
            except Exception:
                return None

        if k == 'components':
            # If not then try to load them from the filesystem
            try:
                components = np.load(self.components_path)['components']
                self.components = components
                return components
            except Exception:
                return None

        if k == 'cpca':
            if self.id is None:
                raise RuntimeError(
                    'Instantiating a CPCA instance from an unsaved Eigenworms instance is not supported.'
                )

            # Instantiate a new CPCA instance
            cpca = CPCA(n_components=self.n_components, whiten=False)
            cpca.n_components_ = self.n_components
            cpca.n_samples_ = self.n_samples
            cpca.n_features_ = self.n_features
            cpca.explained_variance_ = np.array(self.explained_variance)
            cpca.explained_variance_ratio_ = np.array(self.explained_variance_ratio)
            cpca.singular_values_ = np.array(self.singular_values)
            cpca.mean_ = self.mean
            cpca.components_ = self.components

            if cpca.mean_ is None or cpca.components_ is None:
                raise RuntimeError('Failed to restore the mean or components - could not create CPCA instance.')

            self.cpca = cpca
            return cpca

    def validate(self, clean=True):
        super().validate(clean=clean)

        # Check one and only one of dataset or reconstruction is set
        if self.dataset is None and self.reconstruction is None:
            raise ValidationError('Dataset and reconstruction both undefined.')
        if self.dataset is not None and self.reconstruction is not None:
            raise ValidationError('Dataset and reconstruction both defined.')

        # Validate the mean
        if self.mean is None:
            raise ValidationError('Mean not set.')
        if type(self.mean) != np.ndarray:
            raise ValidationError('Mean is not a numpy array.')
        if self.mean.shape != (self.n_features,):
            raise ValidationError('"mean" shape incorrect.')

        # Validate the components
        if self.components is None:
            raise ValidationError('Components not set.')
        if type(self.components) != np.ndarray:
            raise ValidationError('Components are not a numpy array.')
        if self.components.shape != (self.n_components, self.n_features):
            raise ValidationError('Components shape incorrect.')

    def save(self, *args, **kwargs):
        self.updated = datetime.datetime.now()
        res = super().save(*args, **kwargs)

        # Store the mean and components on the hard drive
        os.makedirs(self.components_path.parent, exist_ok=True)
        np.savez_compressed(
            self.components_path,
            components=self.components,
            mean=self.mean
        )

        return res

    def transform(self, X: np.ndarray):
        """
        Apply dimensionality reduction to X.
        X is projected on the principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return self.cpca.transform(X)

    def inverse_transform(self, X: np.ndarray):
        """
        Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        return self.cpca.inverse_transform(X)
