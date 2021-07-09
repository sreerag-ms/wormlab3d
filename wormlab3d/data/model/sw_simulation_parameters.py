import datetime

from mongoengine import *

from simple_worm.material_parameters import MaterialParameters


class SwSimulationParameters(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    worm_length = IntField(required=True)
    duration = FloatField(required=True)
    dt = FloatField(required=True)
    K = FloatField(required=True)
    K_rot = FloatField(required=True)
    A = FloatField(required=True)
    B = FloatField(required=True)
    C = FloatField(required=True)
    D = FloatField(required=True)

    meta = {
        'ordering': ['-created'],
    }

    def get_material_parameters(self) -> MaterialParameters:
        return MaterialParameters(
            K=self.K,
            K_rot=self.K_rot,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D
        )
