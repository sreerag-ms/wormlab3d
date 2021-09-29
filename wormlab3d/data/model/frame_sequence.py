from typing import List

import numpy as np
from mongoengine import *
from wormlab3d.data.model.dataset import Dataset
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline3d import Midline3D, M3D_SOURCES
from wormlab3d.data.model.model import Model
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trial import Trial
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK
from wormlab3d.data.triplet_field import TripletField


class FrameSequence(Document):
    dataset = ReferenceField(Dataset)
    trial = ReferenceField(Trial)
    frames = ListField(ReferenceField(Frame), required=True)
    midlines = ListField(ReferenceField(Midline3D), required=True)

    # Model/source used to generate this midline
    source = StringField(choices=M3D_SOURCES)
    source_file = StringField()
    model = ReferenceField(Model)

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

        # Check that all the midlines relate to the same trial and came from the same place
        midline_ids = [m.id for m in sequence]
        pipeline = [
            {'$match': {'_id': {'$in': midline_ids}}},
            {'$lookup': {'from': 'frame', 'localField': 'frame', 'foreignField': '_id', 'as': 'frame'}},
            {'$unwind': {'path': '$frame'}},
            {'$group': {
                '_id': None,
                'trial_ids': {'$addToSet': '$frame.trial'},
                'sources': {'$addToSet': '$source'},
                'source_files': {'$addToSet': '$source_file'}
            }},
        ]
        res = list(Midline3D.objects().aggregate(pipeline))
        assert len(res) == 1
        res = res[0]
        assert len(res['trial_ids']) == 1
        self.trial = Trial.objects.get(id=res['trial_ids'][0])
        assert len(res['sources']) == 1
        self.source = res['sources'][0]
        assert len(res['source_files']) <= 1
        if len(res['source_files']) == 1:
            self.source_file = res['source_files'][0]
        self.frames = [m.frame for m in sequence]
        self.midlines = sequence

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

        # matches = {'frame0.frame_num': args.start_frame, 'midline0.source': args.midline_source}
        matches = {'trial': args.trial_id, 'frames': {'$size': n_frames}, 'source': args.midline_source}
        if args.midline_source_file is not None:
            matches['source_file'] = args.midline_source_file

        pipeline = [
            {'$match': matches},
            {'$project': {'_id': 1, 'frame0': {'$first': '$frames'}, 'midline0': {'$first': '$midlines'}}},
            {'$lookup': {'from': 'frame', 'localField': 'frame0', 'foreignField': '_id', 'as': 'frame0'}},
            {'$unwind': {'path': '$frame0'}},
            {'$lookup': {'from': 'midline3d', 'localField': 'midline0', 'foreignField': '_id', 'as': 'midline0'}},
            {'$unwind': {'path': '$midline0'}},
            {'$match': {'frame0.frame_num': args.start_frame}},
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
