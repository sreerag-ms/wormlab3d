from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool
from .util import load_head_tail_coordinates


class HeadAndTailOptimizationArgs(BaseArgs):
    def __init__(
            self,
            use_head_and_tail_optimisations: bool = False,
            head_and_tail_coordinates: str = 'data/head_and_tail_coords_dataset_2.csv',
            
            initial_head_and_tail_loss_weight: float = 0.,
            n_steps_head_tail_refine: int = 100,
            ht_freeze_length: bool = True,
            
            load_ht_params: bool = False,
            ht_parameters_id: str = None,
            
            loss_ht_norm: str = "l2",
            loss_ht_delta: float = 3.0,
            loss_ht_eps: float = 1.0,
            
            central_freeze_applied: bool = True,
            
            start_frame: int = 0,
            end_frame: int = -1,
            **kwargs
    ):
        self.use_head_and_tail_optimisations = use_head_and_tail_optimisations
        
        # Load head and tail coordinates if the optimization is enabled
        if use_head_and_tail_optimisations:
            self.head_and_tail_coordinates = load_head_tail_coordinates(
                head_and_tail_coordinates, start_frame, end_frame
            )
        else:
            self.head_and_tail_coordinates = None
            
        self.initial_head_and_tail_loss_weight = initial_head_and_tail_loss_weight if use_head_and_tail_optimisations else 0.
        self.n_steps_head_tail_refine = n_steps_head_tail_refine if use_head_and_tail_optimisations else 0
        self.ht_freeze_length = ht_freeze_length and use_head_and_tail_optimisations
        self.load_ht_params = load_ht_params and use_head_and_tail_optimisations
        self.ht_parameters_id = ht_parameters_id if use_head_and_tail_optimisations else None
        
        # Loss penalty configuration
        self.loss_ht_norm = loss_ht_norm
        self.loss_ht_delta = loss_ht_delta
        self.loss_ht_eps = loss_ht_eps
        
        # Central freeze configuration
        self.central_freeze_applied = central_freeze_applied and use_head_and_tail_optimisations

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        group = parser.add_argument_group('Head and Tail Optimization Args')
        
        group.add_argument('--use-head-and-tail-optimisations', action='store_true', default=False,
                          help='Enable head and tail optimization features. This is a master switch that controls all head and tail related functionality.')
        group.add_argument('--no-use-head-and-tail-optimisations', dest='use_head_and_tail_optimisations',
                          action='store_false',
                          help='Disable all head and tail optimization features.')
        
        group.add_argument('--head-and-tail-coordinates', type=str, 
                          default='data/head_and_tail_coords_dataset_2.csv', 
                          help='Path to head and tail coordinates dataset CSV file.')
        
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
                          
        group.add_argument('--loss-ht-norm', type=str, default='l2', choices=['l2', 'huber', 'charbonnier', 'normal'],
                          help='Type of penalty function for head/tail distance loss. Options: l2, huber, charbonnier.')
        group.add_argument('--loss-ht-delta', type=float, default=3.0,
                          help='Delta parameter for Huber loss (threshold between quadratic and linear regions).')
        group.add_argument('--loss-ht-eps', type=float, default=1.0,
                          help='Epsilon parameter for Charbonnier loss (smoothing parameter).')
        
        group.add_argument('--central-freeze-applied', type=str2bool, default=True,
                          help='Whether to apply central freeze after convergence to stabilize midline sections. Default=True.')
        group.add_argument('--no-central-freeze-applied', dest='central_freeze_applied', 
                          action='store_false', 
                          help='Disable central freeze after convergence.')

    def get_db_params(self) -> dict:
        from wormlab3d.data.model.ht_parameters import HTParameters
        p = {}
        for k in HTParameters._fields.keys():
            if hasattr(self, k):
                p[k] = getattr(self, k)
        return p
