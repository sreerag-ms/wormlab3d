import datetime

from mongoengine import *


class MFCheckpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('MFCheckpoint')
    trial = ReferenceField('Trial')
    masks = ReferenceField('SegmentationMasks')
    model_params = ReferenceField('MFModelParameters', required=True)

    frame_num = IntField(required=True, default=0)
    step_cc = IntField(required=True, default=0)
    step_curve = IntField(required=True, default=0)
    loss_cc = FloatField(required=True, default=1e10)
    loss_curve = FloatField(required=True, default=1e10)
    metrics_cc = DictField()
    metrics_curve = DictField()

    runtime_args = DictField()
    source_args = DictField()
    optimiser_args = DictField()

    meta = {
        'collection': 'mf_checkpoint',
        'ordering': ['-created'],
        'indexes': ['masks', 'model_params']
    }

    def clone(self):
        """
        Return a clone of the current checkpoint with a reference back to where it was cloned from.
        """
        return MFCheckpoint(
            cloned_from=self,
            masks=self.masks,
            model_params=self.model_params,
            step_cc=self.step_cc,
            step_curve=self.step_curve,
            loss_cc=self.loss_cc,
            loss_curve=self.loss_curve,
            metrics_cc=self.metrics_cc,
            metrics_curve=self.metrics_curve,
            runtime_args=self.runtime_args,
            source_args=self.source_args,
            optimiser_args=self.optimiser_args,
        )

    def clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        super().clean()
        for k, v in self.metrics_cc.items():
            self.metrics_cc[k] = float(v)
        for k, v in self.metrics_curve.items():
            self.metrics_curve[k] = float(v)
