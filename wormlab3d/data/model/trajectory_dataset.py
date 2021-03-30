from mongoengine import *

from wormlab3d.data.model.tag import Tag

DATA_TYPES = ['xyz', 'xyz_inv', 'bishop', 'cpca']


class TrajectoryDataset(Document):
    data_type = StringField(required=True, choices=DATA_TYPES)
    train_test_split_target = FloatField(required=True, default=None, min_value=0, max_value=1)
    train_test_split_actual = FloatField(required=True, default=None, min_value=0, max_value=1)
    n_frames = IntField(required=True)
    frame_shift = IntField(required=True)
    n_cpca_components = IntField()
    restrict_tags = ListField(ReferenceField(Tag))
    restrict_concs = ListField(FloatField())
    restrict_length = ListField(IntField())
    include_mirrors = BooleanField(required=True, default=False)
    inv_opt_params = DictField()
    trajectories_train = ListField(ReferenceField('Trajectory'))
    trajectories_test = ListField(ReferenceField('Trajectory'))
    size_all = IntField(default=0)
    size_train = IntField(default=0)
    size_test = IntField(default=0)

    def set_trajectories(self, train, test):
        self.trajectories_train = train
        self.trajectories_test = test
        self.size_all = len(train) + len(test)
        self.size_train = len(train)
        self.size_test = len(test)
        self.train_test_split_actual = len(train) / (len(train) + len(test))
