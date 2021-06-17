import datetime

from mongoengine import *


class SwCheckpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('SwCheckpoint')
    frame_sequence = ReferenceField('FrameSequence', required=True)
    sim_params = ReferenceField('SwSimulationParameters', required=True)
    reg_params = ReferenceField('SwRegularisationParameters', required=True)
    step = IntField(required=True, default=0)
    loss = FloatField(required=True, default=1e10)
    metrics = DictField()

    frame_sequence_args = DictField()
    optimiser_args = DictField()
    runtime_args = DictField()
    sim_args = DictField()
    reg_args = DictField()

    meta = {
        'ordering': ['-created'],
        'indexes': ['frame_sequence', 'sim_params', 'reg_params']
    }

    def clone(self):
        """
        Return a clone of the current checkpoint with a reference back to where it was cloned from.
        """
        return SwCheckpoint(
            cloned_from=self,
            frame_sequence=self.frame_sequence,
            sim_params=self.sim_params,
            reg_params=self.reg_params,
            step=self.step,
            loss=self.loss,
            metrics=self.metrics,
            frame_sequence_args=self.frame_sequence_args,
            optimiser_args=self.optimiser_args,
            runtime_args=self.runtime_args,
            sim_args=self.sim_args,
            reg_args=self.reg_args,
        )

    def clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        super().clean()
        for k, v in self.metrics.items():
            self.metrics[k] = float(v)
