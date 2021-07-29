from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from wormlab3d.nn.models.basenet import InputLayer, OutputLayer
from wormlab3d.nn.models.resnet import ResNet, _ResBlock, _ResLayer, _ResBottleneckLayer, _Shortcut, _ZeroPadChannels


class InputLayer1d(InputLayer):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=1):
        super().__init__(n_channels_in, n_channels_out, kernel_size, stride, padding)
        self.conv = nn.Conv1d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class OutputLayer1d(OutputLayer):
    def __init__(self, n_channels_in, output_shape):
        super().__init__(n_channels_in, output_shape)
        self.bn = nn.BatchNorm1d(n_channels_in)
        self.pool = nn.AdaptiveAvgPool1d(1)


class ResNet1d(ResNet):
    @property
    def id(self):
        bottleneck_variety = 'A' if not self.use_bottlenecks else 'B'
        return 'ResNet1D/{}i_{}_sc-{}_{}_d={}'.format(self.n_init_channels, ','.join(map(str, self.block_config)),
                                                      self.shortcut_type, bottleneck_variety, self.dropout_prob)

    def _build_model(self):
        # First convolution layer
        input_layer = InputLayer1d(self.input_shape[0], self.n_init_channels)

        # Residual components
        components, n_channels = self._build_model_components()

        # Add OutputLayer
        output_layer = OutputLayer1d(n_channels_in=n_channels, output_shape=self.output_shape)

        # Construct model
        self.model = nn.Sequential(OrderedDict([
            ('input_layer', input_layer),
            ('components', components),
            ('output_layer', output_layer),
        ]))

        return self.model

    def _get_res_block(self, n_layers, n_channels_in, n_channels, downsample):
        return _ResBlock1d(
            n_layers=n_layers,
            n_channels_in=n_channels_in,
            n_channels=n_channels,
            downsample=downsample,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            dropout_prob=self.dropout_prob)


class _ResBlock1d(_ResBlock):
    def _get_res_layer(self, n_channels_in, n_channels, downsample):
        Layer = _ResLayer1d if not self.use_bottlenecks else _ResBottleneckLayer1d
        return Layer(
            n_channels_in=n_channels_in,
            n_channels=n_channels,
            downsample=downsample,
            shortcut_type=self.shortcut_type,
            dropout_prob=self.dropout_prob)


class _ResLayer1d(_ResLayer):
    def __init__(self, n_channels_in, n_channels, downsample=False, shortcut_type='id', dropout_prob=0.):
        super().__init__(n_channels_in, n_channels, downsample, shortcut_type, dropout_prob)

        self.shortcut = _Shortcut1d(n_channels_in, n_channels, downsample, shortcut_type)

        conv1_stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm1d(n_channels_in)
        self.conv1 = nn.Conv1d(n_channels_in, n_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_channels)


class _ResBottleneckLayer1d(_ResBottleneckLayer):
    def __init__(self, n_channels_in, n_channels, downsample=False, shortcut_type='id', dropout_prob=0.):
        super().__init__(n_channels_in, n_channels, downsample, shortcut_type, dropout_prob)

        self.shortcut = _Shortcut1d(n_channels_in, n_channels, downsample, shortcut_type)

        # todo: this differs from other implementations, but makes much more sense to me this way
        n_bottleneck_channels = n_channels // 4
        conv1_stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm1d(n_channels_in)
        self.conv1 = nn.Conv1d(n_channels_in, n_bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(n_bottleneck_channels)
        self.conv2 = nn.Conv1d(n_bottleneck_channels, n_bottleneck_channels, kernel_size=3, stride=conv1_stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(n_bottleneck_channels, n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm1d(n_channels)


class _Shortcut1d(_Shortcut):
    def __init__(self, n_channels_in, n_channels_out, downsample, shortcut_type):
        super().__init__(n_channels_in, n_channels_out, downsample, shortcut_type)

        # when number of channels changes we need to project input to new size
        if n_channels_in != n_channels_out:
            self.shortcut = nn.Sequential()
            if shortcut_type == 'conv':
                stride = 2 if downsample else 1
                self.shortcut.add_module('bn', nn.BatchNorm1d(n_channels_in))
                self.shortcut.add_module('relu', nn.ReLU(inplace=True))
                self.shortcut.add_module('conv', nn.Conv1d(n_channels_in, n_channels_out, kernel_size=1, stride=stride,
                                                           bias=False))
            elif shortcut_type == 'id':
                if downsample:
                    self.shortcut.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True))
                self.shortcut.add_module('zero_pad', _ZeroPadChannels1d(n_channels_out - n_channels_in))


class _ZeroPadChannels1d(_ZeroPadChannels):
    def __init__(self, pad_size):
        super().__init__(pad_size)

    def forward(self, x):
        batch_size, _, spatial_size_1 = x.shape
        zeros = Variable(torch.zeros([batch_size, self.pad_size, spatial_size_1], device=x.device))
        return torch.cat([x, zeros], 1)
