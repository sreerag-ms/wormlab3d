from typing import List, Union

from mongoengine import *

from wormlab3d.data.model.cameras import Cameras

SEX_CHOICES = ['H', 'M']
AGE_CHOICES = ['YA']
STRAIN_CHOICES = [None, 'N2', 'UL4207', 'CB540', 'COP2029']


class Experiment(Document):
    id = SequenceField(primary_key=True)
    user = StringField()
    strain = StringField(choices=STRAIN_CHOICES, null=True, default=None)
    sex = StringField(choices=SEX_CHOICES)
    age = StringField(choices=AGE_CHOICES)
    concentration = FloatField(min_value=0)
    worm_length = FloatField(min_value=0)
    legacy_id = IntField(null=True, default=None)

    def get_cameras(self, best: bool = True, source: str = None) -> Union[Cameras, List[Cameras]]:
        """
        Fetch the camera models for this experiment.
        If best=False then returns a list of all associated, otherwise picks the best according to reprojection_error.
        """
        filters = {'experiment': self}
        if source is not None:
            filters['source'] = source
        cameras = Cameras.objects(**filters)
        if best:
            return cameras.first()
        return cameras

    @property
    def num_trials(self):
        from wormlab3d.data.model import Trial
        return Trial.objects(experiment=self).count()

    @property
    def num_frames(self):
        from wormlab3d.data.model import Trial
        return sum([t.n_frames_min for t in Trial.objects(experiment=self).only('n_frames_min')])
