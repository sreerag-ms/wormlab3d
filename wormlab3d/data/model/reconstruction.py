from datetime import datetime
from pathlib import Path
from typing import List

from mongoengine import *

from wormlab3d import RECONSTRUCTION_VIDEOS_PATH, MF_DATA_PATH
from wormlab3d.data.model.eigenworms import Eigenworms
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline3d import M3D_SOURCES, M3D_SOURCE_MF


class Reconstruction(Document):
    created = DateTimeField(required=True, default=datetime.now)
    updated = DateTimeField(required=True, default=datetime.now)
    trial = ReferenceField('Trial', required=True)
    start_frame = IntField(required=True)
    end_frame = IntField(required=True)
    midlines = ListField(LazyReferenceField('Midline3D'))
    source = StringField(choices=M3D_SOURCES, required=True)
    source_file = StringField()
    mf_parameters = ReferenceField('MFParameters')
    copied_from = ReferenceField('Reconstruction')

    meta = {
        'indexes': [
            'trial',
            'source',
            'source_file',
            'mf_parameters',
            {
                'fields': ['trial', 'source', 'source_file', 'mf_parameters'],
                'unique': True
            },
        ]
    }

    def save(self, *args, **kwargs):
        self.updated = datetime.now()
        return super().save(*args, **kwargs)

    def get_frame(self, frame_num: int) -> Frame:
        return self.trial.get_frame(frame_num)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def eigenworms(self) -> List[Eigenworms]:
        return Eigenworms.objects(reconstruction=self).order_by('-updated')

    @property
    def video_filename(self) -> Path:
        return RECONSTRUCTION_VIDEOS_PATH / f'{self.id}.mp4'

    @property
    def has_video(self) -> bool:
        return self.video_filename.exists()

    @property
    def has_data(self) -> bool:
        if self.source != M3D_SOURCE_MF:
            return True
        path_meta = MF_DATA_PATH / f'trial_{self.trial.id}' / str(self.id) / 'metadata.json'
        if path_meta.exists():
            return True
        return False

    def get_time(self, frame_num: int) -> datetime:
        return datetime.fromtimestamp(frame_num / self.trial.fps)
