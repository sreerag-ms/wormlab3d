from typing import Tuple

import torch
from torch import nn
from wormlab3d import PREPARED_IMAGE_SIZE
from wormlab3d.nn.models.basenet import BaseNet


class CoordsNet(nn.Module):
    def __init__(
            self,
            net: BaseNet,
            n_worm_points: int
    ):
        super().__init__()
        self.net = net
        self.n_worm_points = n_worm_points

    def forward(
            self,
            X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Use the neural network to generate flattened coordinates from individual images.
        Reshape these into 2D coordinates and centre on the image.
        """
        bs = X.shape[0]
        device = X.device
        z = self.net(X)
        z = z.reshape(bs, self.n_worm_points, 2)

        # Add centre-point offset
        cp = torch.tensor((PREPARED_IMAGE_SIZE[0] // 2, PREPARED_IMAGE_SIZE[0] // 2), device=device)
        z = z + cp

        return z

    def get_n_params(self) -> int:
        """Return from the encoder network."""
        return self.net.get_n_params()

    def calc_norms(self, p: int = 2) -> float:
        """Return from the encoder network."""
        return self.net.calc_norms(p=p)
