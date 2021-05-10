from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wormlab3d import logger
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.nn.models.rdn import _SFENet, _ConvLayer


class _RDB(nn.Module):
    def __init__(
            self,
            spatial_shape,  # the image dimensions
            K_in,  # channels in
            K_out,  # channels out
            N,  # number of convs in each RDB
            G,  # growth rate in each RDB
            kernel_size=3,
            activation='relu',
            bn=False
    ):
        super().__init__()
        self.spatial_shape = spatial_shape

        # Build dense convolutional layers
        convs = []
        for i in range(N):
            convs.append(
                _ConvLayer(
                    n_channels_in=K_in + i * G,
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
            n_channels_in=K_in + N * G,
            n_channels_out=K_out,
            kernel_size=1,
            activation=activation,
            bn=bn,
            cat_input=False
        )

        # Shortcut
        self.shortcut = _ConvLayer(
            n_channels_in=K_in,
            n_channels_out=K_out,
            kernel_size=1,
            activation=False,
        )

    def forward(self, x):
        if x.shape[2:] != self.spatial_shape:
            x = F.interpolate(x, self.spatial_shape, mode='nearest')  # , align_corners=False)
        Fm = self.convs(x)

        # Input needs expanding/contracting to match number of output channels
        if x.shape[1] != self.LFF.conv.out_channels:
            x = self.shortcut(x)

        return self.LFF(Fm) + x


class RedNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int],
            latent_size: int,
            parameters_size: int,
            K: int,
            M: int,
            N: int,
            G: int,
            discriminator_layers: Tuple[int] = (),
            kernel_size: int = 3,
            activation: str = 'relu',
            batch_norm: bool = False,
            act_out: str = False,
            dropout_prob: float = 0.,
            build_model: bool = True

    ):
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        self.latent_size = latent_size
        self.parameters_size = parameters_size
        self.K = K  # initial number of channels
        self.M = M  # number of RDBs
        self.N = N  # number of convs in each RDB
        self.G = G  # growth rate in each RDB
        self.discriminator_layers = discriminator_layers  # Fully connected discriminator network
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

    def _build_model(self):

        # Calculate shapes
        K_in = self.K
        spatial_shape = self.input_shape[1:]
        K_m = []
        spatial_shape_m = []
        for m in range(self.M):
            K_m.append((K_in, K_in * 2))
            K_in = K_in * 2
            spatial_shape_m.append(spatial_shape)
            spatial_shape = (spatial_shape[0] // 2, spatial_shape[1] // 2)
        logger.debug(f'Spatial shapes: {spatial_shape_m}')
        logger.debug(f'Channel sizes: {K_m}')

        # -------- Encoder --------

        # Shallow Feature Extraction
        self.SFE = _SFENet(
            self.input_shape[0],
            self.K,
            kernel_size=self.kernel_size,
            activation=self.activation,
            bn=self.batch_norm
        )

        # Residual dense blocks
        self.RDBs_enc = nn.ModuleList()
        for m in range(self.M):
            self.RDBs_enc.append(
                _RDB(
                    spatial_shape=spatial_shape_m[m],
                    K_in=K_m[m][0],
                    K_out=K_m[m][1],
                    N=self.N,
                    G=self.G,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    bn=self.batch_norm
                )
            )

        # -------- Bottleneck --------
        C_in = sum([K_m[m][1] for m in range(self.M)])
        self.bottleneck = _ConvLayer(
            n_channels_in=C_in,
            n_channels_out=self.latent_size * 2,
            kernel_size=1,
            activation=None,
            bn=False,
            cat_input=False
        )

        # -------- Discriminator --------
        self.discriminator = nn.Sequential()
        size = self.latent_size
        for m, n in enumerate(self.discriminator_layers):
            self.discriminator.add_module(
                f'DiscriminatorLayer{m}',
                nn.Sequential(*[
                    nn.Linear(size, n),
                    nn.ReLU(inplace=True)
                ])
            )
            size = n
        self.discriminator.add_module(
            'DiscriminatorOutputLayer',
            nn.Linear(size, 1)
        )

        # -------- Decoder --------
        self.latent_to_parameters = nn.Linear(
            in_features=self.latent_size,
            out_features=self.parameters_size
        )
        self.parameters_expansion = _ConvLayer(
            self.parameters_size,
            K_m[-1][1],
            kernel_size=self.kernel_size,
            activation=self.activation,
            bn=self.batch_norm
        )

        # Residual dense blocks
        self.RDBs_dec = nn.ModuleList()
        for m in range(self.M - 1, -1, -1):
            self.RDBs_dec.append(
                _RDB(
                    spatial_shape=spatial_shape_m[m],
                    K_in=K_m[m][1],
                    K_out=K_m[m][0],
                    N=self.N,
                    G=self.G,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    bn=self.batch_norm
                )
            )

        # Add a final 1x1 convolution to resize to number of desired output channels
        C_in = sum([K_m[m][0] for m in range(self.M)])
        self.resize_out = _ConvLayer(
            n_channels_in=C_in,
            n_channels_out=self.output_shape[0],
            kernel_size=1,
            activation=self.act_out,
        )

    def _reparameterise(self, mu, log_var):
        """
        Sample from the latent space.
        https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, X_in):
        bs = X_in.shape[0]

        # Encode input
        F00, F0 = self.SFE(X_in)
        Fi = torch.cat([F00, F0], dim=1)
        Fs = []
        for RDB in self.RDBs_enc:
            Fi = RDB(Fi)
            Fs.append(F.adaptive_avg_pool2d(Fi, (1, 1)))
        Fs = torch.cat(Fs, 1)  # concatenate all feature maps F1..FM
        z_dist = self.bottleneck(Fs)
        z_dist = z_dist.reshape((bs, 2, -1))

        # Sample from the distribution
        z_mu, z_log_var = z_dist[:, 0], z_dist[:, 1]
        z = self._reparameterise(z_mu, z_log_var)

        # Discriminate
        disc = self.discriminator(z)

        # Decode the embedding
        parameters = self.latent_to_parameters(z)
        Fi = self.parameters_expansion(parameters.reshape((bs, self.parameters_size, 1, 1)))
        Fs = []
        for RDB in self.RDBs_dec:
            Fi = RDB(Fi)
            Fs.append(F.adaptive_avg_pool2d(Fi, self.output_shape[1:]))
        Fs = torch.cat(Fs, 1)  # concatenate all feature maps F1..FM
        X_out = self.resize_out(Fs)

        return z_mu, z_log_var, z, disc, parameters, X_out
