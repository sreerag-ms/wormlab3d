import numpy as np
import pickle
from bson import Binary
from mongoengine.base import BaseField


class NumpyField(BaseField):
    def to_python(self, value):
        return pickle.loads(value)

    def to_mongo(self, value):
        return Binary(pickle.dumps(value))

    def validate(self, value):
        if not isinstance(value, np.ndarray):
            self.error('NumpyField only accepts numpy arrays.')
