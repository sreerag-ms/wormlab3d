from tests.data.utils import MongoDBTestCase
from wormlab3d.data.model.experiment import Experiment


class TestExperiment(MongoDBTestCase):
    def test_create_experiment_no_date(self):
        exp = Experiment()
        exp.save()
        # todo: assert raises exception
