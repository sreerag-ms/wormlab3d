import datetime
from typing import List

from mongoengine import *

from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.tag import Tag
from wormlab3d.midlines2d.args import DatasetArgs

DATA_TYPES = ['xyz', 'xyz_inv', 'bishop', 'cpca']

DATASET_TYPE_2D_MIDLINE = '2d_midline'
DATASET_TYPES = [DATASET_TYPE_2D_MIDLINE]


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

    @queryset_manager
    def find_from_args(doc_cls, queryset, args: DatasetArgs):
        return queryset.filter(
            train_test_split_target=args.train_test_split,
            restrict_tags=args.restrict_tags,
            restrict_concs=args.restrict_concs,
            centre_3d_max_error=args.centre_3d_max_error,
            exclude_experiments=args.exclude_experiments,
            include_experiments=args.include_experiments,
            exclude_trials=args.exclude_trials,
            include_trials=args.include_trials,
        )
