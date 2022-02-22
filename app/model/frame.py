from collections import OrderedDict
from typing import Dict

from app.model.document_view import DocumentView
from wormlab3d.data.model import Frame


class FrameView(DocumentView):
    has_item_view = False

    @classmethod
    @property
    def document_class(cls):
        return Frame

    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        return OrderedDict([
            (
                self.prefix + 'trial', {
                    'title': 'Trial',
                    'type': 'relation',
                    'type_rel': 'integer',
                    'collection_name': 'trial',
                    'filter_type': 'integer',
                    'early_match': True
                },
            ),
            (
                self.prefix + 'frame_num', {
                    'title': 'Frame num.',
                    'type': 'integer',
                },
            ),
            (
                self.prefix + 'max_brightnesses', {
                    'title': 'Max brightnesses',
                    'type': 'array',
                    'type_array': 'integer',
                },
            ),
            (
                self.prefix + 'centre_3d.point_3d', {
                    'title': 'Centre 3D',
                    'type': 'array',
                    'type_array': 'float',
                    'precision': 2,
                },
            ),
            (
                self.prefix + 'centre_3d.error', {
                    'title': 'Triangulation error',
                    'type': 'float',
                    'precision': 3,
                },
            ),
            (
                self.prefix + 'centre_3d_fixed.point_3d', {
                    'title': 'Centre 3D Fixed',
                    'type': 'array',
                    'type_array': 'float',
                    'precision': 2,
                },
            ),
        ])
