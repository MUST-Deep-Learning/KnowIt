from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
import torch
from torch import nn
from torch.nn import LSTM, Linear, Module

available_tasks = ("regression", "classification")

HP_ranges_dict = {
    "num_layers": range(1, 21, 1),
    "width": range(2, 1025, 1),
    "dropout": np.arange(0, 1.1, 0.1),
    "output_activation": (None, "Sigmoid", "Softmax"),
}


class Model(Module):
    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        width: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        output_activation: None | str = None,
        *,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.task_name = task_name

        # self.model_in_dim = int(np.prod(input_dim))
        # self.model_out_dim = int(np.prod(output_dim))
        # if len(input_dim) < 3:
        #     input_dim.insert(0, 1)
        #     self.model_in_dim = input_dim
        # else:
        #     self.model_in_dim = input_dim
        self.num_input_components = input_dim[-1]
        self.model_out_dim = int(np.prod(output_dim))
        self.final_out_dim = output_dim

        # input layer
        rnn_layers: list[Module] = []

        rnn_layers.append(
            LSTM(
                input_size=self.num_input_components,
                hidden_size=width,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            ),
        )

        self.lstm_layers = nn.Sequential(*rnn_layers)

        output_layer: list[Module] = []
        output_layer.append(
            Linear(
                in_features=width * input_dim[0],
                out_features=self.model_out_dim,
                bias=False,
            ),
        )

        if self.task_name == "classification":
            if output_activation == "Softmax":
                output_layer_activation = getattr(nn, output_activation)(dim=1)
            elif output_activation is None:
                output_layer_activation = nn.Identity()
            else:
                output_layer_activation = getattr(nn, output_activation)()
            output_layer.append(output_layer_activation)

        self.model_output = nn.Sequential(*output_layer)

    def _regression(self, x: Tensor) -> Tensor:
        """Return model output for an input batch for a regression task.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk * in_components)

        Returns
        -------
            (Tensor):       Model output of shape
                            (batch_size, out_chunk, out_components)

        """
        hidden = self.lstm_layers(x)[0]
        hidden = torch.reshape(hidden, (hidden.shape[0], hidden.shape[1] * hidden.shape[2]))
        out = self.model_output(hidden)

        return out.view(
            out.shape[0],
            self.final_out_dim[0],
            self.final_out_dim[1],
        )

    def _classification(self, x: Tensor) -> Tensor:
        """Return model output for an input batch for a classification task.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk * in_components)

        Returns
        -------
            (Tensor):       Model output of shape
                            (batch_size, num_classes)

        """
        hidden = self.lstm_layers(x)[0]
        hidden = torch.reshape(hidden, (hidden.shape[0], hidden.shape[1] * hidden.shape[2]))
        
        return self.model_output(hidden)

    def forward(self, x: Tensor) -> Tensor:
        """Return model output for an input batch.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk, in_components).

        Returns
        -------
            (Tensor):       Model output of shape
                            (batch_size, out_chunk, out_components) if regress-
                            ion.

                            Model output of shape
                            (batch_size, num_classes) if classification.

        """
        # x = x.view(x.shape[0], self.model_in_dim)

        if self.task_name == "regression":
            return self._regression(x)

        return self._classification(x)
