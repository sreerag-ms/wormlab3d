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
        # Need to connect with uri as authentication details are dropped otherwise!
        # see https://github.com/MongoEngine/mongoengine/issues/851
        uri = f'mongodb://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TEST_DB_NAME}?authSource=admin'
        cls._connection = connect(host=uri)
        # cls._connection = connect(
        #     TEST_DB_NAME,
        #     host=DB_HOST,
        #     port=DB_PORT,
        #     username=DB_USERNAME,
        #     password=DB_PASSWORD,
        #     authentication_source='admin'
        # )
        cls._connection.drop_database(TEST_DB_NAME)
        cls.db = get_db()

    @classmethod
    def tearDownClass(cls):
        cls._connection.drop_database(TEST_DB_NAME)
        disconnect_all()
