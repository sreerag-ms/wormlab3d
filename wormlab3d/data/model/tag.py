from typing import List

from mongoengine import *

from wormlab3d.data.model.frame import Frame


class Tag(Document):
    name = StringField(required=True)
    description = StringField()


