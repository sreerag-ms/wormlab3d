import datetime
from typing import Union

from mongoengine import *

from wormlab3d.data.model.frame_sequence import FrameSequence
from wormlab3d.data.model.sw_run import SwRun


class SwCheckpoint(Document):
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    cloned_from = ReferenceField('SwCheckpoint')
    frame_sequence = ReferenceField('FrameSequence')
    sim_run_target = ReferenceField('SwRun')
    sim_params = ReferenceField('SwSimulationParameters', required=True)
    reg_params = ReferenceField('SwRegularisationParameters', required=True)
    step = IntField(required=True, default=0)
    loss = FloatField(required=True, default=1e10)
    loss_data = FloatField(required=True, default=1e10)
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
            sim_run_target=self.sim_run_target,
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

    def set_target(self, target: Union[FrameSequence, SwRun]):
        """
        Target can either be a FrameSequence or a SwRun instance.
        """
        if isinstance(target, SwRun):
            self.sim_run_target = target
        elif isinstance(target, FrameSequence):
            self.frame_sequence = target
        else:
            raise RuntimeError(f'Unrecognised target type: {type(target).__name__}')

    def clean(self):
        """
        Fix the metric values to be standard floats rather than torch tensors.
        """
        super().clean()
        for k, v in self.metrics.items():
            self.metrics[k] = float(v)

    def get_runs(self):
        """
        Get the simulation runs associated with this checkpoint.
        """
        return SwRun.objects(checkpoint=self)

    @queryset_manager
    def find_checkpoint(
            doc_cls,
            queryset,
            target: Union['FrameSequence', 'SwRun'],
            sim_params: 'SwSimulationParameters',
            reg_params: 'SwRegularisationParameters',
            order: str = 'best'
    ):
        """
        Find a checkpoint matching the target (frame sequence or simulation run) and other parameters.
        """
        filters = {
            'sim_params': sim_params,
            'reg_params': reg_params
        }

        if isinstance(target, SwRun):
            filters['sim_run_target'] = target.id
        elif isinstance(target, FrameSequence):
            filters['frame_sequence'] = target.id
        else:
            raise RuntimeError(f'Unrecognised target type: {type(target)}')

        order_by = '-created' if order == 'latest' else '+loss'

        return queryset.filter(**filters).order_by(order_by)
