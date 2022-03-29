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
            build_model: bool=True,
    ):
        super().__init__(input_shape, output_shape)

        self.layers_config = layers_config
        self.dropout_prob = dropout_prob

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self) -> str:
        return f'LSTMNet/{",".join(map(str, self.layers_config))}_d={self.dropout_prob}'

    def _build_model(self):
        """
        Configure the model with layers of LSTM cells feeding into a linear layer.
        """
        n_features, _ = self.input_shape

        # Configure LSTM cells
        cells = nn.ModuleList()
        input_size = n_features
        for i, n in enumerate(self.layers_config):
            cell = nn.LSTMCell(input_size=input_size, hidden_size=n, bias=True)
            input_size = n
            cells.add_module(f'Layer{i}', cell)
        self.cells = cells

        # Configure linear output layer
        self.output_layer = nn.Linear(input_size, n_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the input data through the LSTM network and continue using the network outputs
        as new inputs until the required output size is reached.
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

        # Process the input and generate the predictions
        outputs = []
        for t in range(T - 1):
            if t < T_data:
                X_t = X[..., t]
                input_t = X_t
            else:
                input_t = output
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(input_t, (h_t[i], c_t[i]))
                input_t = h_t[i]
            output = self.output_layer(input_t)
            outputs.append(output)

        # The outputs are all predictions so concatenate with the first data point so it lines up.
        outputs = torch.stack([X[..., 0], *outputs], dim=2)

        return outputs
