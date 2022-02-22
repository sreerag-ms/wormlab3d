from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from wormlab3d.data.model import Dataset
from wormlab3d.data.model.dataset import DATASET_TYPES
from wormlab3d.data.model.trial import TRIAL_QUALITY_CHOICES


class DatasetView(DocumentView):
    has_item_view = True

    @classmethod
    @property
    def document_class(cls):
        return Dataset

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        return OrderedDict([
            (
                self.prefix + '_id', {
                    'title': 'ID',
                    'type': 'relation',
                    'type_rel': 'objectid',
                    'collection_name': 'dataset'
                },
            ),
            (
                self.prefix + 'dataset_type', {
                    'title': 'Type',
                    'type': 'enum',
                    'choices': DATASET_TYPES
                },
            ),
            (
                self.prefix + 'created', {
                    'title': 'Created',
                    'type': 'date',
                },
            ),
            (
                self.prefix + 'train_test_split_target', {
                    'title': 'Split Target',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'train_test_split_actual', {
                    'title': 'Split Actual',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'none',
                },
            ),
            (
                self.prefix + 'size_all', {
                    'title': 'Size All',
                    'type': 'integer',
                    'filter_type': 'none',
                },
            ),
            (
                self.prefix + 'size_train', {
                    'title': 'Size Train',
                    'type': 'integer',
                    'filter_type': 'none',
                },
            ),
            (
                self.prefix + 'size_test', {
                    'title': 'Size Test',
                    'type': 'integer',
                    'filter_type': 'none',
                },
            ),
            (
                self.prefix + 'restrict_users', {
                    'title': 'Users',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_strains', {
                    'title': 'Strains',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_sexes', {
                    'title': 'Sexes',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_ages', {
                    'title': 'Ages',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_tags', {
                    'title': 'Tags',
                    'type': 'array',
                    'type_array': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_concs', {
                    'title': 'Concs.',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'restrict_sources', {
                    'title': 'Source',
                    'type': 'array',
                    'type_array': 'string',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'min_reconstruction_frames', {
                    'title': 'Min. Frames',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'mf_depth', {
                    'title': 'MF Depth',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'n_worm_points', {
                    'title': 'Num. Worm Points',
                    'type': 'integer',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'centre_3d_max_error', {
                    'title': 'Max Centre Error',
                    'type': 'float',
                    'precision': 2,
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'exclude_experiments', {
                    'title': 'Exclude Experiments',
                    'type': 'array',
                    'type_array': 'relation',
                    'collection_name': 'experiment',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'include_experiments', {
                    'title': 'Include Experiments',
                    'type': 'array',
                    'type_array': 'relation',
                    'collection_name': 'experiment',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'exclude_trials', {
                    'title': 'Exclude Trials',
                    'type': 'array',
                    'type_array': 'relation',
                    'collection_name': 'trial',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'include_trials', {
                    'title': 'Include Trials',
                    'type': 'array',
                    'type_array': 'relation',
                    'collection_name': 'trial',
                    'filter_type': 'choice_query',
                },
            ),
            (
                self.prefix + 'min_trial_quality', {
                    'title': 'Min. Trial Quality',
                    'type': 'enum',
                    'choices': TRIAL_QUALITY_CHOICES
                },
            ),
            (
                self.prefix + 'n_reconstructions', {
                    'title': 'Num. Reconstructions',
                    'type': 'integer',
                    'query': {
                        'operation': 'size',
                        'field': 'reconstructions',
                    }
                },
            ),
        ])
