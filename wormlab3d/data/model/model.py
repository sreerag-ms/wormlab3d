from mongoengine import *


class Model(Document):
    id = SequenceField(primary_key=True)
    file = StringField()
    git_sha = StringField()
    parameters = DictField()
    date = DateTimeField()
