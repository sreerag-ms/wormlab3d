import hashlib
import json


def hash_data(data):
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
