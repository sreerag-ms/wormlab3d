from mongoengine import *


class Tag(Document):
    id = IntField(primary_key=True)
    name = StringField(required=True, unique=True)
    short_name = StringField(required=True, unique=True)
    symbol = StringField(unique=True)
    description = StringField()
