import numpy as np
import pytest
from mongoengine import *

from tests.data.utils import MongoDBTestCase
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK, COMPRESS_BLOSC_POINTER

TESTDOC_COLLECTION = 'test_numpy'


class TestNumpyField(MongoDBTestCase):
    def _save_test_doc(self, numpy_field_options: dict = None, test_array: np.ndarray = None):
        if numpy_field_options is None:
            numpy_field_options = {}

        class TestDoc(Document):
            array = NumpyField(**numpy_field_options)
            meta = {'collection': TESTDOC_COLLECTION}

        doc = TestDoc()
        if test_array is not None:
            doc.array = test_array
        doc.save()
        return doc.id

    def test_no_data(self):
        class TestDoc(Document):
            array = NumpyField()

        doc = TestDoc()
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert id(doc) != id(doc2)
        assert 'array' not in doc

    def test_defaults(self):
        class TestDoc(Document):
            array = NumpyField()

        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3))
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert np.allclose(doc2.array, test_array)

    def test_type_validation(self):
        class TestDoc(Document):
            array = NumpyField()

        bad_arrays = [1, 125.125, 'string', {'k': 'v'}, [1, 2]]
        for bad_array in bad_arrays:
            doc = TestDoc()
            doc.array = bad_array
            with pytest.raises(ValidationError):
                doc.save()

    def test_shape_validation(self):
        class TestDoc(Document):
            array = NumpyField(shape=(3, 3))

        # Check that the field can still be omitted
        doc = TestDoc()
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert doc2.array is None

        # Check that a valid shape can be saved
        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3))
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert doc2.array.shape == (3, 3)

        # Check that an invalid shape raises a validation error
        doc = TestDoc()
        test_array = np.random.normal(size=(4, 3))
        doc.array = test_array
        with pytest.raises(ValidationError):
            doc.save()
        assert doc.id is None

    def test_dtype_conversion(self):
        class TestDoc(Document):
            array = NumpyField(dtype=np.float32)

        # Check that the data is converted to the target dtype
        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3)).astype(np.float64)
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert test_array.dtype == np.float64
        assert doc2.array.dtype == np.float32

        # Check again with integers
        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3)).astype(np.uint8)
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert test_array.dtype == np.uint8
        assert doc2.array.dtype == np.float32

    def test_compression_blosc_pack(self):
        class TestDoc(Document):
            array = NumpyField(compression=COMPRESS_BLOSC_PACK)

        # Check the packing/unpacking with blosc's pack_array method
        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3))
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert doc2.array.dtype == test_array.dtype
        assert np.allclose(doc2.array, test_array)

    def test_compression_blosc_pointer_validation(self):
        with pytest.raises(AssertionError):
            class TestDoc1(Document):
                array = NumpyField(compression=COMPRESS_BLOSC_POINTER)

        with pytest.raises(AssertionError):
            class TestDoc2(Document):
                array = NumpyField(compression=COMPRESS_BLOSC_POINTER, shape=(3, 3))

        with pytest.raises(AssertionError):
            class TestDoc3(Document):
                array = NumpyField(compression=COMPRESS_BLOSC_POINTER, dtype=np.float32)

        class TestDoc4(Document):
            array = NumpyField(compression=COMPRESS_BLOSC_POINTER, shape=(3, 3), dtype=np.float32)

        doc = TestDoc4()
        doc.save()
        assert doc.id is not None

    def test_compression_blosc_pointer(self):
        class TestDoc(Document):
            array = NumpyField(compression=COMPRESS_BLOSC_POINTER, shape=(3, 3), dtype=np.float32)

        # Check the packing/unpacking with blosc's pack_array method
        doc = TestDoc()
        test_array = np.random.normal(size=(3, 3)).astype(np.float64)
        doc.array = test_array
        doc.save()
        doc2 = TestDoc.objects.get(id=doc.id)
        assert doc2.array.dtype == np.float32
        assert test_array.dtype == np.float64
        assert np.allclose(doc2.array, test_array)
