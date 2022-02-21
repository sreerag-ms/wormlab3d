from collections import OrderedDict
from typing import Dict

from app.model import MFParametersView
from app.model.document_view import DocumentView
from app.model.trial import TrialView
from wormlab3d.data.model import Reconstruction


class ReconstructionView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Reconstruction

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        trial_view = TrialView(
            prefix=self.prefix + 'trial'
        )

        parameters_view = MFParametersView(
            prefix=self.prefix + 'mf_parameters'
        )

        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'relation',
                    'collection_name': 'reconstruction'
                },
            ),
            (
                self.prefix + 'created', {
                    'title': 'Created',
                    'type': 'datetime',
                },
            ),
            (
                self.prefix + 'updated', {
                    'title': 'Updated',
                    'type': 'datetime',
                },
            ),
            (
                self.prefix + 'trial', {
                    'title': 'Trial',
                    'type': 'relation',
                    'filter_type': 'integer',
                    'collection_name': 'trial',
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
                self.prefix + 'mf_parameters', {
                    'title': 'Parameters',
                    'type': 'relation',
                    'filter_type': 'string',
                    'view_class': parameters_view,
                },
            ),
            *parameters_view.fields.items(),
        ])
