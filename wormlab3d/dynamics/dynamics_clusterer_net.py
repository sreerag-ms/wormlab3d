from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from wormlab3d.nn.models.basenet import BaseNet


class DynamicsClustererNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: Tuple[int],
            classifier_net: BaseNet,
            dynamics_net: BaseNet,
            X0_duration: int,
            build_model: bool = True,
    ):
        super().__init__(input_shape, output_shape)
        self.classifier_net = classifier_net
        self.dynamics_net = dynamics_net
        self.X0_duration = X0_duration
        if build_model:
            self._build_model()
            self._init_params()

    def _build_model(self):
        self.classifier_net._build_model()
        self.dynamics_net._build_model()

    def forward(
            self,
            X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Put the full X sample through the classifier network and the first bit (X0) through each of the dynamics simulators.
        Return both the dynamics outputs and the classifications.
        """

        # Classify/cluster
        Z = self.classifier_net.forward(X)
        # Z = F.softmax(Z, dim=-1)

        # Simulate
        X0 = X[..., :self.X0_duration]
        Y = self.dynamics_net.forward(X0, Z)

        return Y, Z
