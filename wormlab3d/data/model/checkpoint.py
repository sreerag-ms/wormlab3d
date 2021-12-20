import datetime

from mongoengine import *

from wormlab3d.nn.args.optimiser_args import LOSS_MSE, LOSSES


class Checkpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('Checkpoint')
    dataset = ReferenceField('Dataset', required=True)
    network_params = ReferenceField('NetworkParameters', required=True)
    epoch = IntField(required=True, default=0)
    step = IntField(required=True, default=0)
    examples_count = IntField(required=True, default=0)
    loss_type = StringField(required=True, default=LOSS_MSE, choices=LOSSES)
    loss_train = FloatField(required=True, default=1e10)
    loss_test = FloatField(required=True, default=1e10)
    metrics_train = DictField()
    metrics_test = DictField()

    dataset_args = DictField()
    optimiser_args = DictField()
    runtime_args = DictField()

    parameters_file = StringField()

    meta = {
        'ordering': ['-created'],
        'indexes': ['dataset', 'network_params']
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
            examples_count=self.examples_count,
            loss_type=self.loss_type,
            loss_train=self.loss_train,
            loss_test=self.loss_test,
            metrics_train=self.metrics_train,
            metrics_test=self.metrics_test,
            dataset_args=self.dataset_args,
            optimiser_args=self.optimiser_args,
            runtime_args=self.runtime_args,
        )

    def clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        super().clean()
        for k, v in self.metrics_train.items():
            self.metrics_train[k] = float(v)
        for k, v in self.metrics_test.items():
            self.metrics_test[k] = float(v)
