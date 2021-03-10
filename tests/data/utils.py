import unittest

from mongoengine import disconnect_all, connect, get_db

MONGO_TEST_DB = 'wl3d_test'  # standard name for the test database



class MongoDBTestCase(unittest.TestCase):
    """Base class for tests that need a mongodb connection
    It ensures that the db is clean at the beginning and dropped at the end automatically
    """

    @classmethod
    def setUpClass(cls):
        disconnect_all()
        cls._connection = connect(db=MONGO_TEST_DB)
        cls._connection.drop_database(MONGO_TEST_DB)
        cls.db = get_db()

    @classmethod
    def tearDownClass(cls):
        cls._connection.drop_database(MONGO_TEST_DB)
        disconnect_all()
