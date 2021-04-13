import math
from collections import OrderedDict

import torch
import torch.nn as nn

from wormlab3d.nn.models.basenet import BaseNet, InputLayer


class AENet(BaseNet):
    def __init__(
            self,
            data_shape: tuple,
            latent_size,
            n_init_channels,
            growth_rate,
            block_config_enc,
            block_config_dec,
            compression_factor,
            dropout_prob=0.,
            build_model=True
    ):
        super().__init__(input_shape=data_shape, output_shape=data_shape)
        self.n_init_channels = n_init_channels
        self.latent_size = latent_size
        self.growth_rate = growth_rate
        self.block_config_enc = block_config_enc
        self.block_config_dec = block_config_dec
        self.compression_factor = compression_factor
        self.dropout_prob = dropout_prob

        # print(self.input_shape)
        # self.latent_shape = (self.input_shape[0], 2, self.input_shape[3], self.input_shape[3])
        # self.register_buffer('z', torch.zeros(self.latent_shape))
        # self.register_buffer('X_S', torch.zeros(self.input_shape))
        self.z = None
        self.X_S = None

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'AENet/{self.n_init_channels}i' \
               f'_enc{",".join(map(str, self.block_config_enc))}' \
               f'_dec{",".join(map(str, self.block_config_dec))}' \
               f'_k={self.growth_rate}' \
               f'_c={self.compression_factor}' \
               f'_d={self.dropout_prob}'

    def _build_model(self):
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        # First convolution layer
        input_layer = InputLayer(self.input_shape[0], self.n_init_channels)
        n_channels = self.n_init_channels

        # Add DenseBlocks
        components = nn.Sequential()
        for i, n_layers in enumerate(self.block_config_enc):
            block = self._get_dense_block(n_layers, n_channels)
            components.add_module('Block_%d' % (i + 1), block)
            n_channels = n_channels + n_layers * self.growth_rate

            # Add Transition layers between DenseBlocks
            if i != len(self.block_config_enc) - 1:
                n_compressed_channels = math.floor(n_channels * self.compression_factor)
                trans = _Transition(n_channels_in=n_channels,
                                    n_channels_out=n_compressed_channels,
                                    dropout_prob=self.dropout_prob)
                components.add_module('Transition_%d' % (i + 1), trans)
                n_channels = n_compressed_channels

        # Add Transition layer to latent representation
        latent_transition = _Transition(n_channels_in=n_channels,
                                        n_channels_out=2,
                                        dropout_prob=self.dropout_prob)

        # Construct encoder
        enc = nn.Sequential(OrderedDict([
            ('input_layer', input_layer),
            ('components', components),
            ('latent_transition', latent_transition),
        ]))

        return enc

    def _build_decoder(self):
        # Add Transition layer from latent representation
        latent_transition = _Transition(n_channels_in=2,
                                        n_channels_out=self.n_init_channels,
                                        dropout_prob=self.dropout_prob)
        n_channels = self.n_init_channels

        # Add DenseBlocks
        components = nn.Sequential()
        for i, n_layers in enumerate(self.block_config_dec):
            block = self._get_dense_block(n_layers, n_channels)
            components.add_module('Block_%d' % (i + 1), block)
            n_channels = n_channels + n_layers * self.growth_rate

            # Add Transition layers between DenseBlocks
            if i != len(self.block_config_dec) - 1:
                n_compressed_channels = math.floor(n_channels * self.compression_factor)
                trans = _Transition(n_channels_in=n_channels,
                                    n_channels_out=n_compressed_channels,
                                    dropout_prob=self.dropout_prob)
                components.add_module('Transition_%d' % (i + 1), trans)
                n_channels = n_compressed_channels

        # Add Transition layer back to data representation
        data_transition = _Transition(n_channels_in=n_channels,
                                      n_channels_out=self.input_shape[0])

        # Construct decoder
        dec = nn.Sequential(OrderedDict([
            ('latent_transition', latent_transition),
            ('components', components),
            ('output_layer', data_transition),
        ]))

        return dec

    def _get_dense_block(self, n_layers, n_channels_in):
        return _DenseBlock(
            n_layers=n_layers,
            n_channels_in=n_channels_in,
            growth_rate=self.growth_rate,
            dropout_prob=self.dropout_prob)

    def forward(self, X):
        self.z = self.encoder(X)
        self.X_S = self.decoder(self.z)
        return self.z, self.X_S


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, n_channels_in, growth_rate, dropout_prob=0.):
        super().__init__()
        self.dropout_prob = dropout_prob
        n_channels = n_channels_in
        for i in range(n_layers):
            layer = self._get_dense_layer(n_channels, growth_rate)
            self.add_module('Layer_%d' % (i + 1), layer)
            n_channels += growth_rate

    def _get_dense_layer(self, n_channels_in, n_channels_out):
        return _DenseLayer(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            dropout_prob=self.dropout_prob)


class _CompositeLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=1, dropout_prob=0.):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dropout = None
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class _DenseLayer(_CompositeLayer):
    def __init__(self, n_channels_in, n_channels_out, dropout_prob=0.):
        super().__init__(n_channels_in, n_channels_out,
                         kernel_size=3, stride=1, padding=1,
                         dropout_prob=dropout_prob)

    def _stack(self, x_in, x):
        return torch.cat([x_in, x], 1)

    def forward(self, x):
        x_in = x
        x = super(_DenseLayer, self).forward(x)
        x = self._stack(x_in, x)
        return x


class _Transition(_CompositeLayer):
    def __init__(self, n_channels_in, n_channels_out, dropout_prob=0.):
        super().__init__(n_channels_in, n_channels_out,
                         kernel_size=1, stride=1, padding=0,
                         dropout_prob=dropout_prob)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = super().forward(x)
        # x = self.pool(x)
        return x
