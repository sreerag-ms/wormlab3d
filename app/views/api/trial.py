from collections import OrderedDict
from typing import Any, Dict

from flask import Blueprint

from app.views.api import ExperimentView
from app.views.document_view import DocumentView, NESTED_DOCUMENT_SEPARATOR
from wormlab3d.data.model import Trial



class TrialView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Trial

    # # @classmethod
    # # @property
    # def fields(self, prefix: str = '') -> OrderedDict[str, Any]:
    #     if prefix != '':
    #         prefix = prefix + NESTED_DOCUMENT_SEPARATOR

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        experiment_view = ExperimentView(
            hide_fields=['_id', 'legacy_id', 'num_trials', 'num_frames'],
            prefix='experiment'
        )

        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'legacy_id', {
                    'title': 'Legacy ID',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'experiment', {
                    'title': 'Experiment',
                    'type': 'relation',
                    'filter_type': 'integer',
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
                self.prefix + 'num_frames', {
                    'title': 'Num. frames',
                    'type': 'integer',
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
                self.prefix + 'comments', {
                    'title': 'Comments',
                    'type': 'string',
                    'filter_type': 'none'
                },
            ),
        ])
