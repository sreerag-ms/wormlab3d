import datetime

from mongoengine import *


class MFCheckpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('MFCheckpoint')
    trial = ReferenceField('Trial')
    parameters = ReferenceField('MFParameters', required=True)

    frame_num = IntField(required=True, default=0)
    step = IntField(required=True, default=0)
    step_frame = IntField(required=True, default=0)
    loss = FloatField(required=True, default=1e10)
    metrics = DictField()

    runtime_args = DictField()
    source_args = DictField()

    meta = {
        'collection': 'mf_checkpoint',
        'ordering': ['-created'],
        'indexes': ['parameters']
    }

    def clone(self):
        """
        Return a clone of the current checkpoint with a reference back to where it was cloned from.
        """
        return MFCheckpoint(
            cloned_from=self,
            trial=self.trial,
            parameters=self.parameters,
            frame_num=self.frame_num,
            step=self.step,
            loss=self.loss,
            metrics=self.metrics,
            runtime_args=self.runtime_args,
            source_args=self.source_args,
        )

    def clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        super().clean()
        for k, v in self.metrics.items():
            self.metrics[k] = float(v)
