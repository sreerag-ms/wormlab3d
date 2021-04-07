import pickle

import blosc
import numpy as np
from bson import Binary
from mongoengine.base import BaseField

from wormlab3d import logger

COMPRESS_NONE = 0
COMPRESS_BLOSC_PACK = 1
COMPRESS_BLOSC_POINTER = 2
COMPRESSION_OPTIONS = [COMPRESS_NONE, COMPRESS_BLOSC_PACK, COMPRESS_BLOSC_POINTER]


class NumpyField(BaseField):
    def __init__(
            self,
            *args,
            shape: tuple = None,
            dtype: type = None,
            compression: int = COMPRESS_NONE,
            **kwargs
    ):
        self.shape = shape
        self.dtype = dtype
        assert compression in COMPRESSION_OPTIONS
        if compression == COMPRESS_BLOSC_POINTER:
            assert shape is not None, 'blosc pointer compression requires the shape to be set!'
            assert dtype is not None, 'blosc pointer compression requires the dtype to be set!'
        self.compression = compression
        super().__init__(*args, **kwargs)

    def to_mongo(self, array: np.ndarray):
        """
        Convert array to binary and compress.
        """
        if self.dtype is not None and array.dtype != self.dtype:
            array = array.astype(self.dtype)

        if self.compression == COMPRESS_NONE:
            return Binary(pickle.dumps(array))

        if self.compression == COMPRESS_BLOSC_PACK:
            return blosc.pack_array(
                array=array,
                clevel=9,
                shuffle=blosc.SHUFFLE
            )

        if self.compression == COMPRESS_BLOSC_POINTER:
            return blosc.compress_ptr(
                address=array.__array_interface__['data'][0],
                items=array.size,
                typesize=array.dtype.itemsize,
                clevel=9,
                shuffle=blosc.SHUFFLE
            )

    def to_python(self, bytes_array):
        """
        Unpack array.
        If it fails it tries the other methods so we can change the method without losing the data.
        """

        # Put the expected method at the top of the list of options to try
        options = COMPRESSION_OPTIONS.copy()
        options.remove(self.compression)
        options = [self.compression, ] + options

        array = None
        for compression in options:
            try:
                array = self._decompress(bytes_array, compression)
                break
            except Exception:
                logger.warn(f'Failed to unpack array using method={compression}. Trying next.')
        if array is None:
            raise ValueError('Could not unpack array.')

        return array

    def validate(self, value: np.ndarray, clean: bool = True):
        """
        Validate the array type and shape.
        """
        # Validate type
        if not isinstance(value, np.ndarray):
            self.error('NumpyField only accepts numpy arrays.')

        # Validate shape
        if self.shape is not None and value.shape != self.shape:
            self.error(f'Value shape {value.shape} does not match required shape {self.shape}')

    def _decompress(self, bytes_array, compression: int) -> np.ndarray:
        """
        Decompress the data.
        """
        if compression == COMPRESS_NONE:
            return pickle.loads(bytes_array)

        if compression == COMPRESS_BLOSC_PACK:
            return blosc.unpack_array(bytes_array)

        if compression == COMPRESS_BLOSC_POINTER:
            array = np.empty(self.shape, dtype=self.dtype)
            blosc.decompress_ptr(bytes_array, array.__array_interface__['data'][0])
            return array
