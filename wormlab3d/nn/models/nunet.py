import torch
import torch.nn as nn

from wormlab3d.nn.models.basenet import BaseNet


class NuNet(BaseNet):
    def __init__(
            self,
            data_shape,
            n_init_channels,
            depth,
            up_mode,
            dropout_prob=0.,
            build_model=True
    ):
        super().__init__(data_shape)
        self.n_init_channels = n_init_channels
        self.depth = depth
        self.up_mode = up_mode
        self.dropout_prob = dropout_prob

        self.Z = None
        self.X_S = None

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'NuNet/{self.n_init_channels}i' \
               f'_depth={self.depth}' \
               f'_up={self.up_mode}' \
               f'_drop={self.dropout_prob}'

    def _build_model(self):
        # print('\n\n\nBUILD MODEL-------\n')
        spatial_size = self.input_shape[1]
        temporal_size = self.input_shape[2]
        # print('spatial_size', spatial_size)
        # print('temporal_size', temporal_size)

        n_channels = self.input_shape[0]
        n_channels_out = self.n_init_channels

        self.sizes = []

        # print('\n--DOWN PATH')
        self.down_path = nn.ModuleList()
        self.down_pools = nn.ModuleList()

        for depth in range(self.depth):
            # print(f'block {depth}')
            # print('before spatial_size', spatial_size)
            # print('before temporal_size', temporal_size)
            self.sizes.append((spatial_size, temporal_size))

            block = _NuNetConvBlock(n_channels, n_channels_out, spatial_size, temporal_size)
            self.down_path.append(block)
            n_channels = n_channels_out
            n_channels_out *= 2

            if depth < self.depth - 1:
                # k_size = (
                #     2 if block.spatial_size_out > 10 else 1,
                #     2 if block.temporal_size_out > 10 else 1,
                # )
                k_size = (
                    2 if spatial_size > 3 else 1,
                    2 if temporal_size > 3 else 1,
                )
                # print('ds k_size', k_size, spatial_size, temporal_size)
                ds = nn.MaxPool2d(k_size)
                self.down_pools.append(ds)
                # spatial_size = int(block.spatial_size_out / k_size[0])
                # temporal_size = int(block.temporal_size_out / k_size[1])
                spatial_size = int(spatial_size / k_size[0])
                temporal_size = int(temporal_size / k_size[1])
            # else:
            #     spatial_size = block.spatial_size_out
            #     temporal_size = block.temporal_size_out

            # print('after spatial_size', spatial_size)
            # print('after temporal_size', temporal_size)

        # print('\n--UP PATH')
        n_channels_out = int(n_channels / 2)
        self.up_path = nn.ModuleList()
        for depth in reversed(range(self.depth - 1)):
            # print('depth', depth)
            block = _NuNetUpBlock(n_channels, n_channels_out, self.sizes[depth][0], self.sizes[depth][1], self.up_mode)
            self.up_path.append(block)
            n_channels = n_channels_out
            n_channels_out = int(n_channels_out / 2)

            # spatial_size = block.spatial_size_out
            # temporal_size = block.temporal_size_out

        n_output_channels = 3 + 2 * 3 + 3  # xyz + e1/e2 + alpha/beta/gamma
        self.output_layer = nn.Conv2d(n_channels, n_output_channels,
                                      kernel_size=1, stride=1, padding=0)

        # print('\n\n', self.sizes)

    def forward(self, x):
        bridges = []

        # print('=== FORWARD ===')
        # print('x.shape', x.shape)   # [batch_size, channels, worm_len, n_frames]

        # print('\n---- DOWN ----')
        for depth, down in enumerate(self.down_path):
            # print(f'depth={depth}')
            x = down(x)
            # print('x.shape (after down) (=bridge)', x.shape)

            if depth != len(self.down_path) - 1:
                bridges.append(x)
                x = self.down_pools[depth](x)
                # print('downsampling, x.shape', x.shape)

                # k_size = (
                #     2 if n_spatial > 10 else 1,
                #     2 if n_temporal > 10 else 1,
                # )
                # print('k_size', k_size, n_spatial, n_temporal)
                # x = F.max_pool2d(x, k_size)

                # n_spatial = int(n_spatial / k_size[0])
                # n_temporal = int(n_temporal / k_size[1])

        # print('\n~~~bottom~~~')
        # print(x.shape)

        # print('\n---- UP ----')
        for depth, up in enumerate(self.up_path):
            # print(f'depth={depth}')
            # b = bridges[-depth - 1]
            # print(f'up with x.shape={x.shape} and b.shape={b.shape} (b idx={-depth - 1})')
            x = up(x, bridges[-depth - 1])
            # print('x.shape (after up)', x.shape)

        # print('\n~~~top again~~~')
        # print(x.shape)

        x = self.output_layer(x)

        # print('\n~~~output~~~')
        # print(x.shape)

        # Split output by channels into X_M and z

        X_S = x[:, :3]
        Z = x[:, 3:]

        # print('X_S.shape', X_S.shape)
        # print('z.shape', z.shape)

        self.Z = Z
        self.X_S = X_S

        return Z, X_S


class _NuNetConvBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, spatial_size_in, temporal_size_in, dropout_prob=0.):
        super().__init__()

        self.layers = nn.ModuleList()
        n_channels = n_channels_in
        spatial_size = spatial_size_in
        temporal_size = temporal_size_in

        for i in range(2):
            k_size = (
                3 if spatial_size >= 3 else 1,
                3 if temporal_size >= 3 else 1,
            )
            layer = _NuCompositeLayer(n_channels, n_channels_out, kernel_size=k_size,
                                      dropout_prob=dropout_prob)
            n_channels = n_channels_out
            self.layers.append(layer)
            spatial_size = spatial_size - (k_size[0] - 1)
            temporal_size = temporal_size - (k_size[1] - 1)

        # Resize back to original dimensions
        if spatial_size != spatial_size_in or temporal_size != temporal_size_in:
            k_size = (
                spatial_size_in - spatial_size + 1,
                temporal_size_in - temporal_size + 1,
            )
            # print('upconv k_size', k_size)
            up = nn.ConvTranspose2d(n_channels_out, n_channels_out, kernel_size=k_size, stride=1)
            self.layers.append(up)

        # self.spatial_size_out = spatial_size - (k2_size[0] - 1)
        # self.temporal_size_out = temporal_size - (k2_size[1] - 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _NuNetUpBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, spatial_size, temporal_size, up_mode, dropout_prob=0.):
        super().__init__()

        # self.spatial_size_in = spatial_size_in
        # self.temporal_size_in = temporal_size_in

        if up_mode == 'upconv':
            # k_size = (
            #     2 if spatial_size_in > 1 else 3,
            #     2 if temporal_size_in > 1 else 3,
            # )
            # print('upconv k_size', k_size)

            # self.up = nn.ConvTranspose2d(n_channels_in, n_channels_out, kernel_size=k_size, stride=2)
            self.up = nn.ConvTranspose2d(n_channels_in, n_channels_out, kernel_size=2, stride=2)

            # spatial_size = spatial_size_in * 2
            # temporal_size = temporal_size_in * 2
            # spatial_size = spatial_size_in * k_size[0]
            # temporal_size = temporal_size_in * k_size[1]

        elif up_mode == 'upsample':
            size_out = (spatial_size, temporal_size)
            self.up = nn.Sequential(
                # nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Upsample(mode='bilinear', size=size_out),
                nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1),
            )
            # spatial_size = size_out[0]
            # temporal_size =size_out[1]

        # print('spatial_size', spatial_size)
        # print('temporal_size', temporal_size)
        self.conv_block = _NuNetConvBlock(n_channels_in, n_channels_out, spatial_size, temporal_size,
                                          dropout_prob=dropout_prob)

        # self.spatial_size_out = self.conv_block.spatial_size_out
        # self.temporal_size_out = self.conv_block.temporal_size_out

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        # print('\n---UP forward')
        # print('x.shape', x.shape)
        # print('bridge.shape', bridge.shape)
        up = self.up(x)
        # print('up.shape', up.shape)
        crop1 = self.center_crop(bridge, up.shape[2:])
        # print('crop1.shape', crop1.shape)
        out = torch.cat([up, crop1], 1)
        # print('out.shape',out.shape)
        out = self.conv_block(out)
        # print('out conv.shape',out.shape)

        return out


class _NuCompositeLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=0, dropout_prob=0.):
        super().__init__()
        self.conv = nn.Conv2d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_channels_out)
        self.dropout = None
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.bn(x)
        return x
