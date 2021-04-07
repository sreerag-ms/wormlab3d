import datetime

from mongoengine import *

from wormlab3d.data.model import Dataset
from wormlab3d.data.model.network_parameters import NetworkParameters


class Checkpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('Checkpoint')
    dataset = ReferenceField(Dataset, required=True)
    network_params = ReferenceField(NetworkParameters, required=True)
    epoch = IntField(required=True, default=1)
    step = IntField(required=True, default=0)
    loss_train = FloatField(required=True, default=1e10)
    loss_test = FloatField(required=True, default=1e10)
    stats_train = DictField()
    stats_test = DictField()

    optimiser_args = DictField()
    dataset_args = DictField()
    runtime_args = DictField()

    meta = {
        'ordering': '-created'
    }

    def clone(self):
        """
        Return a clone of the current checkpoint with a reference back to where it was cloned from
        """
        return Checkpoint(
            cloned_from=self,
            dataset=self.dataset,
            network_params=self.network_params,
            epoch=self.epoch,
            step=self.step,
            loss_train=self.loss_train,
            loss_test=self.loss_test,
            stats_train=self.stats_train,
            stats_test=self.stats_test,
            optimiser_args=self.optimiser_args,
            dataset_args=self.dataset_args,
            runtime_args=self.runtime_args,
        )
