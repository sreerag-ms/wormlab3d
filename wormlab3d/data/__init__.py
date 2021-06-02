from mongoengine import connect

from wormlab3d import DB_NAME, DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD

# Connect to the database
connect(
    DB_NAME,
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USERNAME,
    password=DB_PASSWORD,
    authentication_source='admin'
)
