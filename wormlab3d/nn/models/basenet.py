from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn


class BaseNet(nn.Module, ABC):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model: nn.Module = None

    @abstractmethod
    def _build_model(self):
        pass

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_n_params(self) -> int:
        return sum([p.data.nelement() for p in self.parameters()])

    def multi_gpu_mode(self):
        self.model = nn.DataParallel(self.model)

    def calc_norms(self, p: int = 2) -> float:
        p_norms = {}
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                p_norms[name] = m.weight.norm(p)
        return p_norms

    def forward(self, x):
        return self.model(x)


class InputLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, n_channels_in, output_shape):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_channels_in)
        self.relu = nn.ReLU(inplace=True)

        # flat output, eg, classifier
        if len(output_shape) == 1:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(n_channels_in, output_shape[0])

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.linear(x)
        return x
