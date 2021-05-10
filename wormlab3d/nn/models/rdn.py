from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wormlab3d.nn.models.basenet import BaseNet


class RDN(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: Tuple[int],
            K: int,
            M: int,
            N: int,
            G: int,
            kernel_size: int = 3,
            activation: str = 'relu',
            batch_norm: bool = False,
            act_out: str = False,
            dropout_prob: float = 0.,
            build_model: bool = True
    ):
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self.K = K  # number of channels
        self.M = M  # number of RDBs
        self.N = N  # number of convs in each RDB
        self.G = G  # growth rate in each RDB
        self.C_out = self.output_shape[0]  # Number of channels required at output
        self.batch_norm = batch_norm  # Apply batch normalisation layers
        self.kernel_size = kernel_size  # Spatial convolution kernel size
        self.activation = activation  # Activation function to use 'relu', 'elu', 'gelu'
        self.act_out = act_out  # Apply activation to output
        self.dropout_prob = dropout_prob  # Dropout probability (not used everywhere)

        # Model output
        self.X_S = None
        self.Z = None

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        id_ = f'RDN/' \
              f'K={self.K},' \
              f'M={self.M},' \
              f'N={self.N},' \
              f'G={self.G},' \
              f'ks={self.kernel_size},' \
              f'a={self.activation}'
        if self.act_out:
            id_ += f',ao={self.act_out}'
        if self.dropout_prob > 0:
            id_ += f',do={self.dropout_prob}'
        if self.batch_norm:
            id_ += f',bn'
        return id_

    def _build_model(self):
        C_in = self.input_shape[0]

        # Shallow Feature Extraction
        self.SFE = _SFENet(
            C_in,
            self.K,
            kernel_size=self.kernel_size,
            activation=self.activation,
            bn=self.batch_norm
        )

        # Residual dense blocks
        self.RDBs = nn.ModuleList()
        for i in range(self.M):
            self.RDBs.append(
                _RDB(
                    self.K,
                    self.N,
                    self.G,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    bn=self.batch_norm
                )
            )

        # Global Feature Fusion - 1x1 convolution followed by 3x3 convolution
        self.GFF = _GFFNet(
            self.M * self.K,
            self.K,
            kernel_size=self.kernel_size,
            activation=self.activation,
            bn=self.batch_norm
        )

        # Add a final 1x1 convolution to resize to number of desired output channels
        self.resize_out = _ConvLayer(
            n_channels_in=self.K,
            n_channels_out=self.output_shape[0],
            kernel_size=1,
            activation=self.act_out,
        )

    def forward(self, x):
        # Shallow feature extraction, gives two feature maps, F00 and F0
        F00, F0 = self.SFE(x)

        # F0 is fed into the RDB chain
        Fi = F0
        Fs = []
        for RDB in self.RDBs:
            Fi = RDB(Fi)
            Fs.append(Fi)
        Fs = torch.cat(Fs, 1)  # concatenate all feature maps F1..FM

        # Resize spatial dimensions to match target output shape
        Fs = F.interpolate(Fs, self.output_shape[1:], mode='bilinear', align_corners=False)
        F00 = F.interpolate(F00, self.output_shape[1:], mode='bilinear', align_corners=False)

        # Global feature fusion - combine all Fs with residual F00
        y = self.GFF(Fs) + F00

        # Resize output
        y = self.resize_out(y)

        return y


class _SFENet(nn.Module):
    def __init__(
            self,
            C_in,
            K,
            kernel_size=3,
            activation=False,
            bn=False,
    ):
        super().__init__()
        self.conv1 = _ConvLayer(
            n_channels_in=C_in,
            n_channels_out=K // 2,
            kernel_size=kernel_size,
            activation=activation,
            bn=bn,
        )
        self.conv2 = _ConvLayer(
            n_channels_in=K // 2,
            n_channels_out=K // 2,
            kernel_size=kernel_size,
            activation=activation,
            bn=bn,
        )

    def forward(self, x):
        F00 = self.conv1(x)
        F0 = self.conv2(F00)
        return F00, F0


class _GFFNet(nn.Module):
    def __init__(
            self,
            C_in,
            K,
            K_inter=None,
            kernel_size=3,
            activation=False,
            bn=False,
    ):
        super().__init__()
        K_inter = K_inter if K_inter is not None else K

        self.conv1 = _ConvLayer(
            n_channels_in=C_in,
            n_channels_out=K_inter,
            kernel_size=1,
            activation=activation,
            bn=bn
        )
        self.conv2 = _ConvLayer(
            n_channels_in=K_inter,
            n_channels_out=K,
            kernel_size=kernel_size,
            activation=activation,
            bn=bn
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class _RDB(nn.Module):
    def __init__(
            self,
            K,
            N,  # number of convs in each RDB
            G,  # growth rate in each RDB
            kernel_size=3,
            activation='relu',
            bn=False
    ):
        super().__init__()

        # Build dense convolutional layers
        convs = []
        for i in range(N):
            convs.append(
                _ConvLayer(
                    n_channels_in=K + i * G,
                    n_channels_out=G,
                    kernel_size=kernel_size,
                    activation=activation,
                    bn=bn,
                    cat_input=True
                )
            )
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion - 1x1 convolution
        self.LFF = _ConvLayer(
            n_channels_in=K + N * G,
            n_channels_out=K,
            kernel_size=1,
            activation=activation,
            bn=bn,
            cat_input=False
        )

    def forward(self, x):
        Fm = self.convs(x)
        return self.LFF(Fm) + x


class _ConvLayer(nn.Module):
    def __init__(
            self,
            n_channels_in,
            n_channels_out,
            kernel_size=3,
            activation=False,
            bn=False,
            cat_input=False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            n_channels_in,
            n_channels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

        if activation == 'relu':
            self.activation = nn.ReLU()  # inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'hardswish':
            self.activation = nn.Hardswish()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        if bn:
            self.bn = nn.BatchNorm2d(n_channels_out)
        else:
            self.bn = None

        self.cat_input = cat_input

    def forward(self, x):
        y = self.conv(x)
        if self.activation:
            y = self.activation(y)
        if self.bn:
            y = self.bn(y)
        if self.cat_input:
            return torch.cat((x, y), 1)
        else:
            return y
