from mongoengine import *


class Experiment(Document):
    date = DateTimeField(required=True)
    user = StringField()
    strain = StringField()
    concentration = FloatField(min_value=0)
    temperature = FloatField(min_value=0)
    sex = StringField()
    age = StringField()

