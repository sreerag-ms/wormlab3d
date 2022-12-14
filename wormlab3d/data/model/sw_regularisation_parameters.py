import datetime

from mongoengine import *


class L2(EmbeddedDocument):
    alpha = FloatField(default=0)
    beta = FloatField(default=0)
    gamma = FloatField(default=0)
    psi0 = FloatField(default=0)


class grad_t(EmbeddedDocument):
    alpha = FloatField(default=0)
    beta = FloatField(default=0)
    gamma = FloatField(default=0)


class grad_x(EmbeddedDocument):
    alpha = FloatField(default=0)
    beta = FloatField(default=0)
    psi0 = FloatField(default=0)


class SwRegularisationParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    L2 = EmbeddedDocumentField(L2, required=True)
    grad_t = EmbeddedDocumentField(grad_t, required=True)
    grad_x = EmbeddedDocumentField(grad_x, required=True)

    meta = {
        'ordering': ['-created']
    }
