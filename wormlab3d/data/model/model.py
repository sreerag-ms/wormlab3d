from mongoengine import *

from wormlab3d.data.model.frame import Frame


class Model(Document):
    file = StringField()
    git_sha = StringField()
    parameters = DictField()
    date = DateTimeField()



