from argparse import ArgumentParser
from typing import List

from wormlab3d.nn.args import RuntimeArgs
from wormlab3d.toolkit.util import str2bool

LINKAGE_METHOD_OPTIONS = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']


class DynamicsRuntimeArgs(RuntimeArgs):
    def __init__(
            self,
            save_plots: bool = True,
            cluster_every_n_epochs: int = 1,
            min_clusters: int = 2,
            max_clusters: int = 10,
            linkage_methods: List[str] = LINKAGE_METHOD_OPTIONS,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.save_plots = save_plots
        self.cluster_every_n_epochs = cluster_every_n_epochs
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        if cluster_every_n_epochs > -1:
            assert linkage_methods != [], 'At least one linkage method must be provided for clustering.'
        self.linkage_methods = linkage_methods

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = RuntimeArgs.add_args(parser)
        group.add_argument('--save-plots', type=str2bool, default=False,
                           help='Save plot images to disk. Default = False.')
        group.add_argument('--cluster-every-n-epochs', type=int, default=1,
                           help='Cluster the latent representations every n epochs, -1 turns this off.')
        group.add_argument('--min-clusters', type=int, default=2,
                           help='Minimum number of clusters to attempt.')
        group.add_argument('--max-clusters', type=int, default=10,
                           help='Maximum number of clusters to attempt.')
        group.add_argument('--linkage-methods', type=lambda s: [str(item) for item in s.split(',')],
                           default=LINKAGE_METHOD_OPTIONS,
                           help='Linkage methods to use when clustering.')
