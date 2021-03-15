from mongoengine import *

SEX_CHOICES = ['H', 'M']
AGE_CHOICES = ['YA']
STRAIN_CHOICES = [None, 'N2', 'UL4207', 'CB540', 'COP2029']


class Experiment(Document):
    user = StringField()
    strain = StringField(choices=STRAIN_CHOICES, null=True, default=None)
    sex = StringField(choices=SEX_CHOICES)
    age = StringField(choices=AGE_CHOICES)
    concentration = FloatField(min_value=0)
    worm_length = FloatField(min_value=0)
    legacy_id = IntField(null=True, default=None)
