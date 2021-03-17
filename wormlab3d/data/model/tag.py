from mongoengine import *


class Tag(Document):
    id = IntField(primary_key=True)
    name = StringField(required=True)
    short_name = StringField(required=True)
    symbol = StringField()
    description = StringField()
