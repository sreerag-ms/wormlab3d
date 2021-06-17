from typing import List

import numpy as np
from mongoengine import *

from wormlab3d.data.model.dataset import Dataset
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline3d import Midline3D
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trial import Trial
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.data.triplet_field import TripletField


class FrameSequence(Document):
    dataset = ReferenceField(Dataset)
    trial = ReferenceField(Trial)
    frames = ListField(ReferenceField(Frame), required=True)
    midlines = ListField(ReferenceField(Midline3D), required=True)
    tags = ListField(ReferenceField(Tag))
    centre = TripletField(FloatField(), required=True)
    X = NumpyField(dtype=np.float32, compression=COMPRESS_BLOSC_PACK, required=True)

    meta = {
        'indexes': ['trial', 'dataset']
    }

    def set_from_sequence(self, sequence: List[Midline3D]):
        """
        Set the frame references.
        Stack the midlines to form a trajectory.
        Optionally re-centre the trajectory to (0,0,0).
        Collate the tags.
        """
        self.trial = sequence[0].frame.trial
        self.frames = [m.frame for m in sequence]
        self.midlines = sequence
        assert all(m.frame.trial == self.trial for m in sequence)

        # Centre the worm position using the average over the sequence of frames
        X = np.stack([f.X for f in sequence])
        centre = X.mean(axis=(0, 1))
        self.X = X - centre
        self.centre = [float(c) for c in centre]

        # Collate tags
        tags = []
        for f in self.frames:
            tags.extend(f.tags)
        self.tags = list(set(tags))

    @staticmethod
    def find_from_args(args: 'FrameSequenceArgs', n_frames: int) -> QuerySet:
        """
        Search the database for a frame sequence matching the arguments.
        """
        from wormlab3d.simple_worm.args import FrameSequenceArgs
        args: FrameSequenceArgs

        pipeline = [
            {'$match': {'trial': args.trial_id, 'frames': {'$size': n_frames}}},
            {'$project': {'_id': 1, 'frame0': {'$first': '$frames'}, 'midline0': {'$first': '$midlines'}}},
            {'$lookup': {'from': 'frame', 'localField': 'frame0', 'foreignField': '_id', 'as': 'frame0'}},
            {'$unwind': {'path': '$frame0'}},
            {'$lookup': {'from': 'midline3d', 'localField': 'midline0', 'foreignField': '_id', 'as': 'midline0'}},
            {'$unwind': {'path': '$midline0'}},
            {'$match': {'frame0.frame_num': args.start_frame, 'midline0.source': args.midline_source}},
            {'$sort': {'_id': -1}},
            {'$project': {'_id': 1}}
        ]
        cursor = FrameSequence.objects().aggregate(pipeline)

        # Build query set to return
        ids = []
        for res in cursor:
            ids.append(res['_id'])
        qs = FrameSequence.objects(id__in=ids)

        return qs
