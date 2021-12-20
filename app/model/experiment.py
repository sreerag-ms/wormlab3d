from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from wormlab3d.data.model import Experiment


class ExperimentView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Experiment

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'integer',
                }
            ),
            (
                self.prefix + 'user', {
                    'title': 'User',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'strain', {
                    'title': 'Strain',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'sex', {
                    'title': 'Sex',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'age', {
                    'title': 'Age',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'concentration', {
                    'title': 'Conc.',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'worm_length', {
                    'title': 'Worm length',
                    'type': 'float',
                    'precision': 2
                },
            ),
            (
                self.prefix + 'legacy_id', {
                    'title': 'Legacy id',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'num_trials', {
                    'title': 'Num. Trials',
                    'type': 'integer',
                    'query': {
                        'lookup': 'trial',
                        'aggregation': 'count'
                    }
                },
            ),
            (
                self.prefix + 'num_frames', {
                    'title': 'Num. Frames',
                    'type': 'integer',
                    'query': {
                        'lookup': 'trial',
                        'aggregation': 'sum',
                        'field': 'n_frames_min'
                    }
                }
            )
        ])
