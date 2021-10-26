import torch
from torch import nn


class ParameterHolder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise RuntimeError('The forward method of this model is not enabled!')

    def get(self, key: str) -> torch.Tensor:
        return self._parameters[key]

    def set(self, key: str, value: torch.Tensor, requires_grad: bool = False):
        if key in self._parameters:
            self._parameters[key].data = value
        else:
            self.register_parameter(key, nn.Parameter(value, requires_grad=requires_grad))
