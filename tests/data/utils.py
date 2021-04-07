import unittest

from mongoengine import disconnect_all, connect, get_db

from wormlab3d import DB_NAME, DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD

TEST_DB_NAME = DB_NAME + '_test'  # standard name for the test database


class MongoDBTestCase(unittest.TestCase):
    """
    Base class for tests that need a mongodb connection.
    It ensures that the db is clean at the beginning and dropped at the end automatically.
    """

    @classmethod
    def setUpClass(cls):
        disconnect_all()
        cls._connection = connect(
            TEST_DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            username=DB_USERNAME,
            password=DB_PASSWORD,
            authentication_source='admin'
        )
        cls.db = get_db()
        cls._connection.drop_database(TEST_DB_NAME)

    @classmethod
    def tearDownClass(cls):
        cls._connection.drop_database(TEST_DB_NAME)
        disconnect_all()
