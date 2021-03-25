# Connect to the database
from mongoengine import connect

# todo: get env vars
connect(
    'wormlab3d',
    host='127.0.0.1',
    port=27017,
    username='root',
    password='example',
    authentication_source='admin'
)


from wormlab3d.data.model.cameras import Cameras
from wormlab3d.data.model.experiment import Experiment
from wormlab3d.data.model.frame import Frame
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.midline3d import Midline3D
from wormlab3d.data.model.model import Model
from wormlab3d.data.model.tag import Tag
from wormlab3d.data.model.trajectory import Trajectory
from wormlab3d.data.model.trajectory_dataset import TrajectoryDataset
from wormlab3d.data.model.trial import Trial
