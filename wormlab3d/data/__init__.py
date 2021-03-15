# Connect to the database
from mongoengine import connect

# todo: get env vars
connect(
    'wormlab3d',
    host='127.0.0.1',
    port=27017,
    username='root',
    password='example',
    authentication_source='admin'
)
