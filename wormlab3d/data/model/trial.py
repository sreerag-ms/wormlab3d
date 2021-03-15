from mongoengine import *

from wormlab3d.data.model.experiment import Experiment


class File(EmbeddedDocument):
    location = StringField(required=True)


class Trial(Document):
    experiment = ReferenceField(Experiment)
    date = DateTimeField(required=True)
    num_frames = IntField(required=True, default=0)
    fps = FloatField()
    quality = FloatField()
    temperature = FloatField(min_value=0)
    comments = StringField()
    files = ListField(EmbeddedDocumentField(File))
    legacy_id = StringField(unique=True)
    legacy_data = DictField()
