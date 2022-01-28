import datetime
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from mongoengine import *

from wormlab3d import logger, CAMERA_IDXS, DATASETS_MIDLINES3D_PATH
from wormlab3d.data.model import Cameras
from wormlab3d.data.model.experiment import STRAIN_CHOICES, SEX_CHOICES, AGE_CHOICES
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.midline3d import M3D_SOURCES
from wormlab3d.data.model.segmentation_masks import SegmentationMasks
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.numpy_field import NumpyField, COMPRESS_BLOSC_POINTER
from wormlab3d.data.triplet_field import TripletField
from wormlab3d.nn.args import DatasetArgs

DATA_TYPES = ['xyz', 'xyz_inv', 'bishop', 'cpca']

DATASET_TYPE_2D_MIDLINE = '2d_midline'
DATASET_TYPE_3D_MIDLINE = '3d_midline'
DATASET_TYPE_SEGMENTATION_MASKS = 'segmentation_masks'
DATASET_TYPES = {
    DATASET_TYPE_2D_MIDLINE: '2D Midline',
    DATASET_TYPE_3D_MIDLINE: '3D Midline',
    DATASET_TYPE_SEGMENTATION_MASKS: 'Segmentation Masks'
}


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
    dataset_type = StringField(required=True, choices=list(DATASET_TYPES.keys()))
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)
    train_test_split_target = FloatField(required=True, default=None, min_value=0, max_value=1)
    train_test_split_actual = FloatField(required=False, default=None, min_value=0, max_value=1)
    size_all = IntField(default=0)
    size_train = IntField(default=0)
    size_test = IntField(default=0)
    restrict_users = ListField(StringField())
    restrict_strains = ListField(StringField(choices=STRAIN_CHOICES))
    restrict_sexes = ListField(StringField(choices=SEX_CHOICES))
    restrict_ages = ListField(StringField(choices=AGE_CHOICES))
    restrict_tags = ListField(ReferenceField(Tag))
    restrict_concs = ListField(FloatField())
    centre_3d_max_error = FloatField()
    exclude_experiments = ListField(ReferenceField('Experiment'))
    include_experiments = ListField(ReferenceField('Experiment'))
    exclude_trials = ListField(ReferenceField('Trial'))
    include_trials = ListField(ReferenceField('Trial'))
    min_trial_quality = IntField()
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
            restrict_users=args.restrict_users,
            restrict_strains=args.restrict_strains,
            restrict_sexes=args.restrict_sexes,
            restrict_ages=args.restrict_ages,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
            min_trial_quality=args.min_trial_quality,
            n_worm_points=args.n_worm_points,
            restrict_sources=args.restrict_sources,
            mf_depth=args.mf_depth,
        )

    @staticmethod
    def from_args(args: DatasetArgs) -> 'Dataset':
        common_args = dict(
            train_test_split_target=args.train_test_split,
            restrict_users=args.restrict_users,
            restrict_strains=args.restrict_strains,
            restrict_sexes=args.restrict_sexes,
            restrict_ages=args.restrict_ages,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
            min_trial_quality=args.min_trial_quality,
        )

        if args.dataset_type == DATASET_TYPE_2D_MIDLINE:
            DS = DatasetMidline2D(**common_args)
        elif args.dataset_type == DATASET_TYPE_SEGMENTATION_MASKS:
            DS = DatasetSegmentationMasks(**common_args)
        elif args.dataset_type == DATASET_TYPE_3D_MIDLINE:
            DS = DatasetMidline3D(
                n_worm_points=args.n_worm_points,
                restrict_sources=args.restrict_sources,
                mf_depth=args.mf_depth,
                **common_args
            )
        else:
            raise RuntimeError(f'Unrecognised dataset_type={args.dataset_type}.')

        return DS

    def get_cameras(self, tt: str) -> List[Cameras]:
        assert tt in ['train', 'test']
        if not hasattr(self, f'cams_{tt}'):
            raise RuntimeError(f'Dataset does not have cams_{tt} property.')
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

    def get_camera_coefficients(self, tt: str, rebuild: bool = False) -> List[List[float]]:
        assert tt in ['train', 'test']
        if not hasattr(self, f'cam_coeffs_{tt}'):
            raise RuntimeError(f'Dataset does not have cam_coeffs_{tt} property.')
        coeffs = getattr(self, f'cam_coeffs_{tt}')
        if len(coeffs) == 0 or rebuild:
            if rebuild:
                logger.info('Rebuilding camera coefficients.')
            else:
                logger.info('No camera coefficients generated, generating now.')
            coeffs = []
            cams = getattr(self, f'cams_{tt}')
            for cameras in cams:
                cameras = cameras.fetch()
                # Extract camera coefficients
                fx = np.array([cameras.matrix[c][0, 0] for c in CAMERA_IDXS])
                fy = np.array([cameras.matrix[c][1, 1] for c in CAMERA_IDXS])
                cx = np.array([cameras.matrix[c][0, 2] for c in CAMERA_IDXS])
                cy = np.array([cameras.matrix[c][1, 2] for c in CAMERA_IDXS])
                R = np.array([cameras.pose[c][:3, :3] for c in CAMERA_IDXS])
                t = np.array([cameras.pose[c][:3, 3] for c in CAMERA_IDXS])
                d = np.array([cameras.distortion[c] for c in CAMERA_IDXS])
                if cameras.shifts is not None:
                    s = np.array([cameras.shifts.dx, cameras.shifts.dy, cameras.shifts.dz])
                else:
                    s = np.zeros(3)
                coeffs_i = np.concatenate([
                    fx.reshape(3, 1),
                    fy.reshape(3, 1),
                    cx.reshape(3, 1),
                    cy.reshape(3, 1),
                    R.reshape(3, 9),
                    t,
                    d,
                    s.reshape(3, 1)
                ], axis=1).astype(np.float32)
                coeffs.append(coeffs_i)

            setattr(self, f'cam_coeffs_{tt}', coeffs)
            self.save()

            exit()
        return coeffs

    def get_camera_coeffs_range(self) -> Tuple[np.ndarray, np.ndarray]:
        cc_train = np.array(self.get_camera_coefficients('train'))
        cc_test = np.array(self.get_camera_coefficients('test'))
        cc_all = np.concatenate((cc_train, cc_test), axis=0)
        mean = cc_all.mean(axis=0)
        amin = cc_all.min(axis=0)
        amax = cc_all.max(axis=0)
        arange = (amax - amin)
        return mean, arange

    def get_points_3d_range(self) -> Tuple[np.ndarray, np.ndarray]:
        p3d_all = np.concatenate((np.array(self.points_3d_train), np.array(self.points_3d_test)), axis=0)
        mean = p3d_all.mean(axis=0)
        amin = p3d_all.min(axis=0)
        amax = p3d_all.max(axis=0)
        arange = (amax - amin)
        return mean, arange

    def get_points_2d_range(self) -> Tuple[np.ndarray, np.ndarray]:
        p2d_all = np.concatenate((np.array(self.points_2d_train), np.array(self.points_2d_test)), axis=0)
        mean = p2d_all.mean(axis=0)
        amin = p2d_all.min(axis=0)
        amax = p2d_all.max(axis=0)
        arange = (amax - amin)
        return mean, arange


class DatasetMidline2D(Dataset):
    X_train = ListField(ReferenceField(Midline2D))
    X_test = ListField(ReferenceField(Midline2D))
    cams_train = ListField(LazyReferenceField(Cameras))
    cams_test = ListField(LazyReferenceField(Cameras))
    cam_coeffs_train = ListField(NumpyField(shape=(3, 22), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))
    cam_coeffs_test = ListField(NumpyField(shape=(3, 22), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))

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
    cam_coeffs_train = ListField(NumpyField(shape=(3, 22), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))
    cam_coeffs_test = ListField(NumpyField(shape=(3, 22), dtype=np.float32, compression=COMPRESS_BLOSC_POINTER))
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


class DatasetMidline3D(Dataset):
    n_worm_points = IntField(required=True)
    restrict_sources = ListField(StringField(choices=M3D_SOURCES))
    mf_depth = IntField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = DATASET_TYPE_3D_MIDLINE
        self.X_train = None
        self.X_test = None
        self.metas = None

    @property
    def data_path(self) -> Path:
        dest = DATASETS_MIDLINES3D_PATH / f'{self.id}.npz'
        return dest

    @property
    def metas_path(self) -> Path:
        dest = DATASETS_MIDLINES3D_PATH / f'{self.id}.meta.json'
        return dest

    def set_data(self, train: np.ndarray, test: np.ndarray = None, metas: Dict[str, List[int]] = None):
        """
        Convenience method for setting the train and test data and automatically generating some stats.
        """
        if test is None:
            test = np.zeros((0, *train.shape[1:]))
        self.X_train = train
        self.X_test = test
        self.size_all = len(train) + len(test)
        self.size_train = len(train)
        self.size_test = len(test)
        if self.size_all > 0:
            self.train_test_split_actual = len(train) / self.size_all
        self.metas = metas

    def __getattribute__(self, k):
        if k not in ['X_train', 'X_test', 'metas']:
            return super().__getattribute__(k)

        # Check if the variable has been defined or loaded already
        v = super().__getattribute__(k)
        if v is not None:
            return v

        if k == 'metas':
            # Check for metadata first
            try:
                with open(self.metas_path, 'r') as f:
                    metas = json.load(f)
                    setattr(self, k, metas)
                    return metas
            except Exception:
                return None

        # If not then try to load it from the filesystem
        try:
            X = np.load(self.data_path)[k]
            setattr(self, k, X)
            return X
        except Exception:
            return None

    def validate(self, clean=True):
        super().validate(clean=clean)

        # Validate the data
        if self.X_train is None:
            raise ValidationError('X_train not set.')
        if type(self.X_train) != np.ndarray:
            raise ValidationError('X_train is not a numpy array.')
        if self.X_test is None:
            raise ValidationError('X_test not set.')
        if type(self.X_test) != np.ndarray:
            raise ValidationError('X_test is not a numpy array.')

    def save(self, *args, **kwargs):
        res = super().save(*args, **kwargs)

        # Store the metas and data on the hard drive
        os.makedirs(self.data_path.parent, exist_ok=True)

        # Metas
        with open(self.metas_path, 'w') as f:
            json.dump(self.metas, f, indent=2, separators=(',', ': '))

        # Data
        np.savez_compressed(
            self.data_path,
            X_train=self.X_train,
            X_test=self.X_test,
        )

        return res
