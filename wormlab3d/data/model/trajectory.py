from typing import List

import numpy as np
from mongoengine import *

from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trajectory_dataset import TrajectoryDataset
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_PACK


class Trajectory(Document):
    dataset = ReferenceField(TrajectoryDataset)
    frames = ListField(ReferenceField(Frame))
    tags = ListField(ReferenceField(Tag))
    X = NumpyField(dtype=np.float32, compression=COMPRESS_BLOSC_PACK)

    def set_from_sequence(self, sequence: List[Frame], centre: bool = True):
        """
        Set the frame references.
        Stack the midlines to form a trajectory.
        Optionally re-centre the trajectory to (0,0,0).
        Collate the tags.
        """
        self.frames = sequence
        self.X = np.stack([f.X for f in sequence])

        # Centre the worm position using the average over the sequence of frames
        if centre:
            centre = self.X.mean(axis=(0, 2), keepdims=True)
            self.X = self.X - centre

        # Collate tags
        tags = []
        for f in self.frames:
            tags.extend(f.tags)
        self.tags = list(set(tags))
