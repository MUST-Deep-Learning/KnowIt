"""
----
LSTM
----

An example of an LSTM architecture using Pytorch.

The LSTM consists of recurrent cells that contain hidden states and cell
states. When in stateless mode, the hidden and cell states are re-initialized for each batch.
When in stateful mode, hidden and cell states are maintained across batches only if prediction points at
corresponding indices across batches are contiguous in time.

The architecture consists of a set of basic uni-directional LSTM layers, followed by a fully connected layer and an
optional output activation function.

"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "tiantheunissen@gmail.com, randlerabe@gmail.com"
__description__ = "Contains an example of a LSTM architecture."

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import LSTM, Linear, Module

from helpers.logger import get_logger

logger = get_logger()

available_tasks = ("regression", "classification")

HP_ranges_dict = {"width": range(2, 1025, 1),
                  "depth": range(2, 1025, 1),
                  "dropout": np.arange(0, 1.1, 0.1),
                  "output_activation": (None, "Sigmoid", "Softmax"),
                  "stateful": (False, True),
                  "hc_init_method": ('zeros', 'random'),
                  "bidirectional": (False, True),
                  "residual": (False, True)}

class Model(Module):

    task_name = None
    model_in_dim = None
    model_out_dim = None
    final_out_dim = None
    stateful = False
    last_ist_idx = None

    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        *,
        width: int = 256,
        depth: int = 2,
        dropout: float = 0.5,
        output_activation: str | None = None,
        stateful: bool = False,
        hc_init_method: str = 'zeros',
        layernorm: bool = True,
        bidirectional: bool = False,
        residual: bool = False
    ) -> None:
        super().__init__()

        self.task_name = task_name
        self.input_size = input_dim[-1]
        self.model_out_dim = int(np.prod(output_dim))
        self.final_out_shape = output_dim
        self.stateful = stateful

        self.lstm_layers = nn.ModuleList()
        for d in range(depth):
            self.lstm_layers.append(
                LSTM_Block(
                    input_size=self.input_size if d == 0 else width,
                    hidden_size=width,
                    num_layers=1,
                    dropout=dropout,
                    batch_first=True,
                    hc_init_method=hc_init_method,
                    layernorm=layernorm,
                    bidirectional=bidirectional,
                    residual= residual
                )
            )

        self.output_layers = []
        self.output_layers.append(Linear(in_features=width * input_dim[-2],
                                        out_features=self.model_out_dim,
                                        bias=False))
        if output_activation is not None:
            self.output_layers.append(get_output_activation(output_activation))
        self.output_layers = nn.Sequential(*self.output_layers)

    def _reset_all_layer_states(self, batch_size, device, changed_idx=None):
        for layer in self.lstm_layers:
            layer.reset_states(batch_size=batch_size, device=device, changed_idx=changed_idx)

    def _detach_all_layer_hidden_states(self):
        for layer in self.lstm_layers:
            layer.hidden_state = (layer.hidden_state[0].detach(), layer.hidden_state[1].detach())

    def _handle_states(self, ist_idx, device) -> None:

        def _is_contiguous(a, b):
            same_i = a[:, 0] == b[:, 0]
            same_s = a[:, 1] == b[:, 1]
            next_t = a[:, 2]+1 == b[:, 2]
            result = same_i & same_s & next_t
            return result

        if self.stateful:
            if not self.last_ist_idx is None:
                # this is not the first pass of data
                self._detach_all_layer_hidden_states()
                if self.last_ist_idx.shape == ist_idx.shape:
                    # new batch is same shape as last batch
                    changed = ~_is_contiguous(self.last_ist_idx, ist_idx)
                    self._reset_all_layer_states(ist_idx.shape[0], device, changed)
                else:
                    # new batch is different shape to last batch
                    self._reset_all_layer_states(ist_idx.shape[0], device)
            else:
                self._reset_all_layer_states(ist_idx.shape[0], device)
            self.last_ist_idx = ist_idx.clone()
        else:
            self._reset_all_layer_states(ist_idx.shape[0], device)

    def forward(self, batch: dict) -> Tensor:

        x = batch['x']
        self._handle_states(batch['ist_idx'][0], x.device)

        hidden = x
        for i, layer in enumerate(self.lstm_layers):
            hidden = layer(hidden)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1] * hidden.shape[2])
        out = self.output_layers(hidden)
        out = out.view(hidden.shape[0], self.final_out_shape[0], self.final_out_shape[1])

        if self.task_name == 'regression':
            out = out.view(hidden.shape[0], self.final_out_shape[0], self.final_out_shape[1])
        elif self.task_name == 'classification':
            pass
        else:
            logger.error('Unknown task name %s, for LSTM.', self.task_name)
            exit(101)

        return out

class LSTM_Block(nn.Module):

    hidden_size = None
    num_layers = None
    hc_init_method = None
    hidden_state = None
    bidirectional = None

    def __init__(self, input_size, hidden_size,
                 num_layers, dropout, batch_first, hc_init_method,
                 layernorm, bidirectional, residual):
        super(LSTM_Block, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hc_init_method = hc_init_method
        self.bidirectional = bidirectional
        self.residual = residual

        self.lstm_layer = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=batch_first,
            bidirectional=bidirectional)

        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size) if layernorm else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_states(batch_size=1, device='cuda')

    def reset_states(self, batch_size, device, changed_idx=None):
        if changed_idx is None:
            d = 2 if self.bidirectional else 1
            if self.hc_init_method == 'zeros':
                h = torch.zeros(self.num_layers * d, batch_size, self.hidden_size).to(device)
                c = torch.zeros(self.num_layers * d, batch_size, self.hidden_size).to(device)
            elif self.hc_init_method == 'random':
                h = torch.randn(self.num_layers * d, batch_size, self.hidden_size).to(device)
                c = torch.randn(self.num_layers * d, batch_size, self.hidden_size).to(device)
            else:
                logger.error('Unknown hc_init_method %s ', self.hc_init_method)
                exit(101)
            self.hidden_state = (h, c)
        else:
            if self.hc_init_method == 'zeros':
                self.hidden_state[0][:, changed_idx, :] = 0.
                self.hidden_state[1][:, changed_idx, :] = 0.
            elif self.hc_init_method == 'random':
                self.hidden_state[0][:, changed_idx, :] = torch.randn_like(self.hidden_state[0][:, changed_idx, :])
                self.hidden_state[1][:, changed_idx, :] = torch.randn_like(self.hidden_state[0][:, changed_idx, :])
            else:
                logger.error('Unknown hc_init_method %s ', self.hc_init_method)
                exit(101)

    def forward(self, x) -> Tensor:
        res = x if x.shape[-1] == self.hidden_size and self.residual else None
        out, self.hidden_state = self.lstm_layer(x, self.hidden_state)
        out = self.layer_norm(out)
        if res is not None:
            out = out + res
        out = self.dropout(out)
        return out


def get_output_activation(output_activation: None | str) -> Module:
    """Fetch output activation function from Pytorch."""
    if output_activation == "Softmax":
        return getattr(nn, output_activation)(dim=1)
    if output_activation == "Sigmoid":
        return getattr(nn, output_activation)()
    return nn.Identity()