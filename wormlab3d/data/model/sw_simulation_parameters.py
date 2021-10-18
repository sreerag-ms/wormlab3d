import datetime

from mongoengine import *


class SwControlGate(EmbeddedDocument):
    block = BooleanField()
    grad_up = FloatField()
    offset_up = FloatField()
    grad_down = FloatField()
    offset_down = FloatField()


class SwControlGates(EmbeddedDocument):
    alpha = EmbeddedDocumentField(SwControlGate)
    beta = EmbeddedDocumentField(SwControlGate)
    gamma = EmbeddedDocumentField(SwControlGate)


class SwSimulationParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    worm_length = IntField(required=True)
    duration = FloatField(required=True)
    dt = FloatField(required=True)
    gates = EmbeddedDocumentField(SwControlGates)

    # Now moved to fields in SwRun
    K = FloatField()
    K_rot = FloatField()
    A = FloatField()
    B = FloatField()
    C = FloatField()
    D = FloatField()

    meta = {
        'ordering': ['-created'],
    }
