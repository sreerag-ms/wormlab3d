import datetime
from typing import Dict, Union

from mongoengine import *

from wormlab3d.data.model.mf_parameters import MFParameters
from wormlab3d.data.model.ht_parameters import HTParameters
from wormlab3d.data.model.trial import Trial


class MFRuntimeLog(Document):
    """
    Runtime log document to store losses and metrics for each training step.
    The combination of mf_parameters and ht_parameters identifies a unique run.
    """
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    
    trial = ReferenceField(Trial, required=True)
    mf_parameters = ReferenceField(MFParameters, required=True)
    ht_parameters = ReferenceField(HTParameters, required=False)  # Optional since HT params might not always be used
    
    global_step = IntField(required=True)
    frame_step = IntField(required=True)
    frame_num = IntField(required=True)
    
    total_loss = FloatField(required=True)
    
    loss_masks = FloatField()
    loss_scores = FloatField()
    loss_parents = FloatField()
    loss_smoothness = FloatField()
    loss_intersections = FloatField()
    loss_alignment = FloatField()
    loss_consistency = FloatField()
    loss_curvature = FloatField()
    loss_temporal = FloatField()
    loss_temporal_points = FloatField()
    loss_global = FloatField()
    loss_head_and_tail = FloatField()
    
    # Depth-specific losses (stored as lists for each depth)
    loss_depth = ListField(FloatField())
    
    # Training metrics
    learning_rate = FloatField()
    shifts = IntField()  # Number of shifted points
    
    # Additional metrics (flexible storage for other stats)
    additional_metrics = DictField()
    
    meta = {
        'collection': 'mf_runtime_logs',
        'ordering': ['-created'],
        'indexes': [
            # Index for efficient querying by run identification
            ('trial', 'mf_parameters', 'ht_parameters'),
            # Index for time-series queries
            ('trial', 'mf_parameters', 'ht_parameters', 'global_step'),
            ('trial', 'mf_parameters', 'ht_parameters', 'frame_num'),
            # Performance indexes
            'global_step',
            'frame_num',
            'created'
        ]
    }
    
    def __repr__(self):
        return f"MFRuntimeLog(trial={self.trial.id}, step={self.global_step}, frame={self.frame_num}, loss={self.total_loss:.6f})"
    
    @classmethod
    def get_logs_for_run(cls, trial, mf_parameters, ht_parameters=None):
        """
        Get all logs for a specific run identified by the parameter combination.
        """
        query = {
            'trial': trial,
            'mf_parameters': mf_parameters,
        }
        if ht_parameters is not None:
            query['ht_parameters'] = ht_parameters
        else:
            query['ht_parameters__exists'] = False
            
        return cls.objects(**query).order_by('global_step')
    
    @classmethod
    def get_latest_log_for_run(cls, trial, mf_parameters, ht_parameters=None):
        """
        Get the most recent log for a specific run.
        """
        logs = cls.get_logs_for_run(trial, mf_parameters, ht_parameters)
        return logs.order_by('-global_step').first()
    
    @classmethod
    def get_logs_for_frame(cls, trial, mf_parameters, frame_num, ht_parameters=None):
        """
        Get all logs for a specific frame.
        """
        query = {
            'trial': trial,
            'mf_parameters': mf_parameters,
            'frame_num': frame_num,
        }
        if ht_parameters is not None:
            query['ht_parameters'] = ht_parameters
        else:
            query['ht_parameters__exists'] = False
            
        return cls.objects(**query).order_by('frame_step')