from typing import Tuple

import torch
import torch.nn as nn

from wormlab3d.nn.models.basenet import BaseNet


class LSTMNet(BaseNet):
    def __init__(
            self,
            input_shape: tuple,
            output_shape: tuple,
            layers_config: Tuple[int],
            dropout_prob: float = 0.,
            build_model: bool = True,
    ):
        super().__init__(input_shape, output_shape)

        self.layers_config = layers_config
        self.dropout_prob = dropout_prob

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self) -> str:
        lc = ','.join([str(l) for l in self.layers_config])
        return f'LSTMNet/{lc}_d={self.dropout_prob}'

    def _build_model(self):
        """
        Configure the model with layers of LSTM cells feeding into a linear layer.
        """
        n_features, latent_size = self.input_shape

        # Configure LSTM cells
        cells = nn.ModuleList()
        input_size = n_features + latent_size
        for i, n in enumerate(self.layers_config):
            cell = nn.LSTMCell(input_size=input_size, hidden_size=n, bias=True)
            input_size = n
            cells.add_module(f'Layer{i}', cell)
        self.cells = cells

        # Configure linear output layer
        self.output_layer = nn.Linear(input_size, n_features)

        # Dropout between LSTM layers
        self.dropout = nn.Dropout(self.dropout_prob, inplace=True)

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Run the input data through the LSTM network and continue using the network outputs
        as new inputs until the required output size is reached. Context Z is provided along
        with the input at each time step.
        """
        bs, Nc, T_data = X.shape
        device = X.device
        T = self.output_shape[-1]

        # Set up initial hidden and cell states
        h_t = []
        c_t = []
        for i, n in enumerate(self.layers_config):
            h_t.append(torch.zeros(bs, n, dtype=torch.float32, device=device))
            c_t.append(torch.zeros(bs, n, dtype=torch.float32, device=device))

        # The outputs are all predictions so concatenate with the first data point so it lines up.
        outputs = [X[..., 0], ]

        # Process the input and generate the predictions
        output = torch.zeros(bs, Nc, dtype=torch.float32, device=device)
        for t in range(T - 1):
            if t < T_data:
                X_t = X[..., t]
                input_t = torch.cat([X_t, Z], dim=1)
            else:
                input_t = torch.cat([output, Z], dim=1)
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(input_t, (h_t[i], c_t[i]))
                if i < len(self.cells) - 1:
                    input_t = self.dropout(h_t[i])
                else:
                    input_t = h_t[i]
            output = self.output_layer(input_t)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=2)

        return outputs
