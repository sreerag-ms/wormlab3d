import datetime
import json

from bson import ObjectId


class JSONEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, datetime.datetime):
            return str(z)
        elif isinstance(z, ObjectId):
            return str(z)
        else:
            return super().default(z)
