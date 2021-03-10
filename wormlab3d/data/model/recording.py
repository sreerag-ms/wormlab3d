from mongoengine import *

from wormlab3d.data.model.experiment import Experiment


class File(EmbeddedDocument):
    location = StringField(required=True)


class Recording(Document):
    experiment = ReferenceField(Experiment)
    start = DateTimeField(required=True)
    end = DateTimeField()
    duration = DateTimeField()
    files = ListField(EmbeddedDocumentField(File))

