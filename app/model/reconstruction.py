from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from app.model.experiment import ExperimentView
from app.model.trial import TrialView
from wormlab3d.data.model import Reconstruction


class ReconstructionView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Reconstruction

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        # experiment_view = ExperimentView(
        #     hide_fields=['_id', 'legacy_id', 'num_trials', 'num_frames'],
        #     prefix='experiment'
        # )
        trial_view = TrialView(
            # hide_fields=['_id', 'legacy_id', 'comments', 'experiment2'],
            prefix=self.prefix + 'trial'
        )

        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'objectid',
                },
            ),
            # (
            #     self.prefix + 'experiment', {
            #         'title': 'Experiment',
            #         'type': 'relation',
            #         'filter_type': 'integer',
            #         'view_class': experiment_view,
            #     },
            # ),
            # *experiment_view.fields.items(),
            (
                self.prefix + 'trial', {
                    'title': 'Trial',
                    'type': 'relation',
                    'filter_type': 'integer',
                    'view_class': trial_view,
                },
            ),
            *trial_view.fields.items(),
            (
                self.prefix + 'start_frame', {
                    'title': 'Start frame',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'end_frame', {
                    'title': 'End frame',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'source', {
                    'title': 'Source',
                    'type': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'source_file', {
                    'title': 'Source file',
                    'type': 'string',
                },
            ),
            (
                self.prefix + 'model', {
                    'title': 'Model',
                    'type': 'relation',
                    'filter_type': 'integer',
                    # 'view_class': trial_view,
                },
            ),
        ])
