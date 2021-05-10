import torch
import torch.nn as nn
import torch.nn.functional as F

from wormlab3d.nn.models.rdn import RDN, _SFENet, _RDB, _GFFNet, _ConvLayer


class MSRDN(RDN):
    def __init__(
            self,
            *args,
            D: int = 3,
            **kwargs
    ):
        self.D = D
        super().__init__(*args, **kwargs)

    @property
    def id(self):
        id_ = f'MSRDN/' \
              f'D={self.D},' \
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
        C_out = self.output_shape[0]

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
        self.RDB_bottlenecks = nn.ModuleList()
        for i in range(self.M):
            self.RDBs.append(
                _RDB(
                    self.D * self.K,
                    self.N,
                    self.G,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    bn=self.batch_norm
                )
            )
            self.RDB_bottlenecks.append(
                nn.ModuleList(
                    [
                        _ConvLayer(
                            n_channels_in=self.D * self.K,
                            n_channels_out=self.K,
                            kernel_size=1,
                            activation=self.activation,
                            bn=self.batch_norm
                        )
                        for _ in range(self.D)
                    ]
                )
            )

        # Global Feature Fusion - 1x1 convolution followed by 3x3 convolution
        gff_c_in = self.D * self.M * self.K
        gff_c_out = self.K if C_out < self.K else min(gff_c_in, C_out)
        gff_c_inter = (gff_c_in + gff_c_out) // 2

        self.GFF = _GFFNet(
            gff_c_in,
            gff_c_out,
            K_inter=gff_c_inter,
            kernel_size=self.kernel_size,
            activation=self.activation,
            bn=self.batch_norm
        )

        # Add a final 1x1 convolution to resize to number of desired output channels
        self.resize_out = _ConvLayer(
            n_channels_in=gff_c_out,
            n_channels_out=C_out,
            kernel_size=1,
            activation=self.act_out,
            bn=self.batch_norm
        )

    def _get_shape(self, d):
        N0 = self.input_shape[1]
        T0 = self.input_shape[2]
        Nd = int(N0 + d / (self.D - 1) * (1 - N0))
        Td = int(T0 + d / (self.D - 1) * (1 - T0))
        return Nd, Td

    def _interpolate(self, x, d, mode='bilinear'):
        shape = self._get_shape(d)
        xi = F.interpolate(x, shape, mode=mode, align_corners=False)
        return xi

    def _multiscale(self, v):
        return [self._interpolate(v, d) for d in range(self.D)]

    def forward(self, x):
        # Create multiscale copies of x
        xs = self._multiscale(x)

        # Shallow feature extraction at all depths
        F00s = []
        F0s = []
        for d in range(self.D):
            F00, F0 = self.SFE(xs[d])

            F00ms = self._multiscale(F00)
            F0ms = self._multiscale(F0)

            F00s.append(F00ms)
            F0s.append(F0ms)

        # F0s are fed into the RDB chain
        F_prev = F0s
        Fs = []
        for RDB, bottleneck in zip(self.RDBs, self.RDB_bottlenecks):
            Fds = []
            for d in range(self.D):
                # Stack all feature maps produced at different scales, resized to fit current depth
                F_in = torch.cat([F_prev[d2][d] for d2 in range(self.D)], 1)
                Fd = RDB(F_in)
                Fd = bottleneck[d](Fd)
                Fdms = self._multiscale(Fd)
                Fds.append(Fdms)
            Fs.append(Fds)
            F_prev = Fds

        # Stack all feature maps F1..FM produced at different scales, resized to fit output shape
        F_in = torch.cat([
            F.interpolate(Fs[m][d2][d2], self.output_shape[1:], mode='bilinear', align_corners=False)
            for m in range(self.M)
            for d2 in range(self.D)
        ], 1)

        # Collect all F00s together, resized to output shape
        F00 = [
            F.interpolate(F00s[d2][d2], self.output_shape[1:], mode='bilinear', align_corners=False)
            for d2 in range(self.D)
        ]

        # Global feature fusion - combine all Fs with residual F00s
        y0 = self.GFF(F_in)  # + sum(F00)

        # Resize output channels
        y_out = self.resize_out(y0)

        return y_out
