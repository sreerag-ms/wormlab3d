from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from app.model.experiment import ExperimentView
from wormlab3d.data.model import Trial
from wormlab3d.data.model.trial import TRIAL_QUALITY_CHOICES


class TrialView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Trial

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        experiment_view = ExperimentView(
            hide_fields=['_id', 'legacy_id', 'num_trials', 'num_frames'],
            prefix=self.prefix + 'experiment'
        )

        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'relation',
                    'type_rel': 'int',
                    'collection_name': 'trial'
                },
            ),
            (
                self.prefix + 'experiment', {
                    'title': 'Experiment',
                    'type': 'relation',
                    'type_rel': 'int',
                    'filter_type': 'integer',
                    'collection_name': 'experiment',
                    'view_class': experiment_view,
                },
            ),
            *experiment_view.fields.items(),
            (
                self.prefix + 'date', {
                    'title': 'Date',
                    'type': 'date',
                },
            ),
            (
                self.prefix + 'trial_num', {
                    'title': 'Trial num.',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'n_frames_min', {
                    'title': 'Num. frames',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'duration', {
                    'title': 'Duration',
                    'type': 'time',
                    'query': {
                        'operation': 'divide',
                        'fields': [self.prefix + 'n_frames_min', self.prefix + 'fps']
                    },
                },
            ),
            (
                self.prefix + 'fps', {
                    'title': 'FPS',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'temperature', {
                    'title': 'Temperature',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'legacy_id', {
                    'title': 'Legacy ID',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'comments', {
                    'title': 'Comments',
                    'type': 'string',
                    'filter_type': 'none'
                },
            ),
            (
                self.prefix + 'num_reconstructions', {
                    'title': 'Num. Reconstructions',
                    'type': 'integer',
                    'query': {
                        'lookup': 'reconstruction',
                        'aggregation': 'count'
                    }
                },
            ),
            (
                self.prefix + 'quality', {
                    'title': 'Quality',
                    'type': 'enum',
                    'choices': TRIAL_QUALITY_CHOICES
                },
            ),
        ])
