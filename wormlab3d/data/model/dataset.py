import datetime
from typing import List

import numpy as np
from mongoengine import *

from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.segmentation_masks import SegmentationMasks
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_POINTER
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.nn.args import DatasetArgs

DATA_TYPES = ['xyz', 'xyz_inv', 'bishop', 'cpca']

DATASET_TYPE_2D_MIDLINE = '2d_midline'
DATASET_TYPE_SEGMENTATION_MASKS = 'segmentation_masks'
DATASET_TYPES = [
    DATASET_TYPE_2D_MIDLINE,
    DATASET_TYPE_SEGMENTATION_MASKS
]


class TagInfo(EmbeddedDocument):
    tag = ReferenceField(Tag)
    name = StringField(required=True)
    n = IntField(default=0, required=True)
    n_train = IntField(default=0, required=True)
    n_test = IntField(default=0, required=True)
    n_target_train = IntField(default=0, required=True)
    n_target_test = IntField(default=0, required=True)
    split = FloatField(default=0)


class Dataset(Document):
    dataset_type = StringField(required=True, choices=DATASET_TYPES)
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    train_test_split_target = FloatField(required=True, default=None, min_value=0, max_value=1)
    train_test_split_actual = FloatField(required=False, default=None, min_value=0, max_value=1)
    size_all = IntField(default=0)
    size_train = IntField(default=0)
    size_test = IntField(default=0)
    restrict_tags = ListField(ReferenceField(Tag))
    restrict_concs = ListField(FloatField())
    centre_3d_max_error = FloatField(required=True)
    exclude_experiments = ListField(ReferenceField('Experiment'))
    include_experiments = ListField(ReferenceField('Experiment'))
    exclude_trials = ListField(ReferenceField('Trial'))
    include_trials = ListField(ReferenceField('Trial'))
    tag_info = EmbeddedDocumentListField(TagInfo)

    meta = {
        'allow_inheritance': True,
        'ordering': ['-created']
    }

    def get_size(self, train_or_test: str = None):
        if train_or_test == 'train':
            return self.size_train
        elif train_or_test == 'test':
            return self.size_test
        else:
            return self.size

    @queryset_manager
    def find_from_args(doc_cls, queryset, args: DatasetArgs):
        return queryset.filter(
            dataset_type=args.dataset_type,
            train_test_split_target=args.train_test_split,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
        )

    @staticmethod
    def from_args(args: DatasetArgs) -> 'Dataset':
        common_args = dict(
            train_test_split_target=args.train_test_split,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
        )

        if args.dataset_type == DATASET_TYPE_2D_MIDLINE:
            DS = DatasetMidline2D(**common_args)
        elif args.dataset_type == DATASET_TYPE_SEGMENTATION_MASKS:
            DS = DatasetSegmentationMasks(**common_args)
        else:
            raise RuntimeError(f'Unrecognised dataset_type={args.dataset_type}.')

        return DS


class DatasetMidline2D(Dataset):
    X_train = ListField(ReferenceField(Midline2D))
    X_test = ListField(ReferenceField(Midline2D))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = DATASET_TYPE_2D_MIDLINE

    def set_data(self, train: List[Midline2D], test: List[Midline2D]):
        """
        Convenience method for setting the train and test data and automatically generating some stats.
        """
        self.X_train = train
        self.X_test = test
        self.size_all = len(train) + len(test)
        self.size_train = len(train)
        self.size_test = len(test)
        if self.size_all > 0:
            self.train_test_split_actual = len(train) / self.size_all


class DatasetSegmentationMasks(Dataset):
    X_train = ListField(LazyReferenceField(SegmentationMasks))
    X_test = ListField(LazyReferenceField(SegmentationMasks))
    cams_train = ListField(LazyReferenceField(Cameras))
    cams_test = ListField(LazyReferenceField(Cameras))
    cam_coeffs_train = ListField(NumpyField(shape=(3, 19), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))
    cam_coeffs_test = ListField(NumpyField(shape=(3, 19), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))
    points_3d_train = ListField(TripletField(FloatField()))
    points_3d_test = ListField(TripletField(FloatField()))
    points_2d_train = ListField(TripletField(ListField(FloatField())))
    points_2d_test = ListField(TripletField(ListField(FloatField())))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = DATASET_TYPE_SEGMENTATION_MASKS

    def set_data(self, train: List[SegmentationMasks], test: List[SegmentationMasks]):
        """
        Convenience method for setting the train and test data and automatically generating some stats.
        """
        self.X_train = train
        self.X_test = test
        self.size_all = len(train) + len(test)
        self.size_train = len(train)
        self.size_test = len(test)
        if self.size_all > 0:
            self.train_test_split_actual = len(train) / self.size_all

    def get_cameras(self, tt: str) -> List[Cameras]:
        assert tt in ['train', 'test']
        cams = getattr(self, f'cams_{tt}')
        if len(cams) == 0:
            logger.info('No cameras linked, linking now.')
            cams = []
            X = getattr(self, f'X_{tt}')
            for mask in X:
                mask = mask.fetch()
                cams.append(mask.frame.centre_3d.cameras)
            setattr(self, f'cams_{tt}', cams)
            self.save()
        return cams

    def get_camera_coefficients(self, tt: str) -> List[List[float]]:
        assert tt in ['train', 'test']
        coeffs = getattr(self, f'cam_coeffs_{tt}')
        if len(coeffs) == 0:
            logger.info('No camera coefficients generated, generating now.')
            coeffs = []
            cams = getattr(self, f'cams_{tt}')
            for cameras in cams:
                cameras = cameras.fetch()
                # Extract camera coefficients
                fx = np.array([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
                fy = np.array([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
                R = np.array([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
                t = np.array([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
                d = np.array([cameras.distortion[c] for c in CAMERA_IDXS])
                coeffs_i = np.concatenate([
                    fx.reshape(3, 1), fy.reshape(3, 1), R.reshape(3, 9), t, d
                ], axis=1).astype(np.float32)
                coeffs.append(coeffs_i)
            setattr(self, f'cam_coeffs_{tt}', coeffs)
            self.save()

            exit()
        return coeffs

    def get_points_3d(self, tt: str) -> List[float]:
        assert tt in ['train', 'test']
        p3d = getattr(self, f'points_3d_{tt}')
        if len(p3d) == 0:
            logger.info('No 3d points linked, linking now.')
            p3d = []
            X = getattr(self, f'X_{tt}')
            for mask in X:
                mask = mask.fetch()
                p3d.append(mask.frame.centre_3d.point_3d)
            setattr(self, f'points_3d_{tt}', p3d)
            self.save()
        return p3d

    def get_points_2d(self, tt: str) -> List[float]:
        assert tt in ['train', 'test']
        p2d = getattr(self, f'points_2d_{tt}')
        if len(p2d) == 0:
            logger.info('No 2d points linked, linking now.')
            p2d = []
            X = getattr(self, f'X_{tt}')
            for mask in X:
                mask = mask.fetch()
                p2d.append(mask.frame.centre_3d.reprojected_points_2d)
            setattr(self, f'points_2d_{tt}', p2d)
            self.save()
            exit()
        return p2d
