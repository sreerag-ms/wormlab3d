from mongoengine import get_connection
from pymongo import MongoClient

from tests.data.utils import MongoDBTestCase


class TestConnection(MongoDBTestCase):

    def test_connection(self):
        conn = get_connection()
        assert isinstance(conn, MongoClient)

