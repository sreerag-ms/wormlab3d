from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool
from .util import load_head_tail_coordinates


class HeadAndTailOptimizationArgs(BaseArgs):
    def __init__(
            self,
            read_head_and_tail_coordinates: bool = True,
            head_and_tail_coordinates: str = 'data/head_and_tail_coords_dataset_2.csv',
            
            initial_head_and_tail_loss_weight: float = 0.,
            n_steps_head_tail_refine: int = 100,
            ht_freeze_length: bool = True,
            
            load_ht_params: bool = False,
            ht_parameters_id: str = None,
            
            start_frame: int = 0,
            end_frame: int = -1,
            **kwargs
    ):
        self.read_head_and_tail_coordinates = read_head_and_tail_coordinates
        
        if read_head_and_tail_coordinates:
            self.head_and_tail_coordinates = load_head_tail_coordinates(
                head_and_tail_coordinates, start_frame, end_frame
            )
        else:
            self.head_and_tail_coordinates = None
            
        self.initial_head_and_tail_loss_weight = initial_head_and_tail_loss_weight
        self.n_steps_head_tail_refine = n_steps_head_tail_refine
        self.ht_freeze_length = ht_freeze_length
        self.load_ht_params = load_ht_params
        self.ht_parameters_id = ht_parameters_id

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group('Head and Tail Optimization Args')
        
        group.add_argument('--head-and-tail-coordinates', type=str, 
                          default='data/head_and_tail_coords_dataset_2.csv', 
                          help='Path to head and tail coordinates dataset CSV file.')
        group.add_argument('--read-head-and-tail-coordinates', action='store_true', default=False, 
                          help='Whether to read and load head and tail coordinates from CSV file.')
        group.add_argument('--no-read-head-and-tail-coordinates', dest='read_head_and_tail_coordinates', 
                          action='store_false', 
                          help='Disable reading head and tail coordinates from CSV file.')
        
        group.add_argument('--initial-head-and-tail-loss-weight', type=float, default=0.,
                          help='Weight for head and tail coordinate loss from ground truth data.')
        group.add_argument('--n-steps-head-tail-refine', type=int, default=100,
                          help='Number of refinement steps for head and tail.')
        group.add_argument('--ht-freeze-length', type=str2bool, default=True,
                          help='Freeze length for a few steps so endpoints can slide along the curve. Default=True.')
        
        group.add_argument('--load-ht-params', action='store_true', default=False,
                          help='Load existing HT parameters from database.')
        group.add_argument('--ht-parameters-id', type=str, default=None,
                          help='ID of the HT parameters record to load from database.')

    def get_db_params(self) -> dict:
        from wormlab3d.data.model.ht_parameters import HTParameters
        p = {}
        for k in HTParameters._fields.keys():
            if hasattr(self, k):
                p[k] = getattr(self, k)
        return p
