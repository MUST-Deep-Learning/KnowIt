"""
-----
LSTM
-----

As an example, we define a multilayer perceptron (MLP) architecture using the
well-known Pytorch library.

The example shows one way of constructing the MLP which can either be shallow
or deep. The MLP is capable of handling regression or classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import LSTM, Linear, Module

from helpers.logger import get_logger

logger = get_logger()

available_tasks = ("regression", "classification")

HP_ranges_dict = {
    "arch_args": {
        "width": range(2, 1025, 1),
        "depth": range(2, 1025, 1),
        "dropout": np.arange(0, 1.1, 0.1),
        "output_activation": (None, "Sigmoid", "Softmax"),
        "bidirectional": (False, True),
    },
    "internal_state": {
        "init_hidden_state": (None, "random", Tensor),
        "init_cell_state": (None, "random", Tensor),
        "tracking": (False, True),
    },
}


@dataclass(frozen=True)
class ArchArgs(yaml.YAMLObject):
    yaml_tag = "!arch_args"

    width: int = 256
    depth: int = 1
    dropout: float = 0.5
    output_activation: None | str = None
    bidirectional: bool = False

@dataclass(frozen=True)
class Internal(yaml.YAMLObject):
    yaml_tag = "!internal"

    init_hidden_state: None | str | Tensor=None
    init_cell_state: None | str | Tensor=None
    tracking: bool=False


class Model(Module):
    def __init__(
        self,
        input_dim: tuple[int, int, int],
        output_dim: list[int],
        task_name: str,
        arch_args: None | ArchArgs = None,
        internal_state: None | Internal = None,
    ) -> None:
        super().__init__()

        # incomponents and out_components
        self.input_size = input_dim[-1]
        self.model_out_dim = int(np.prod(output_dim))
        self.final_out_shape = output_dim

        # remember to move h0 and c0 to device

        if not arch_args:
            # revert to default arg values in ArchArgs
            arch_args = ArchArgs()

        # config for cell and hidden state initialization for each batch
        if internal_state:
            self._internal = InternalState(
                width=arch_args.width,
                depth=arch_args.depth,
                bidirect=arch_args.bidirectional,
                init_hidden_state=internal_state.init_hidden_state,
                init_cell_state=internal_state.init_cell_state,
                track_hidden=internal_state.tracking,
            )
        else:
            # revert to default vals if no internal args provided
            internal_state = Internal()
            self._internal = InternalState(
                width=arch_args.width,
                depth=arch_args.depth,
                bidirect=arch_args.bidirectional,
                init_hidden_state=internal_state.init_hidden_state,
                init_cell_state=internal_state.init_cell_state,
                track_hidden=internal_state.tracking,
            )

        # rnn layer (input x will return three tensors: output, (hn, cn))
        self.lstm_layers = LSTM(
                input_size=self.input_size,
                hidden_size=arch_args.width,
                num_layers=arch_args.depth,
                dropout=arch_args.dropout,
                bidirectional=arch_args.bidirectional,
                batch_first=True,
            )

        output_layer: list[Module] = []
        output_layer.append(
            Linear(
                in_features=self._internal.d * arch_args.width * input_dim[-2],
                out_features=self.model_out_dim,
                bias=False,
            ),
        )

        self.task_name = task_name
        if task_name == "classification":
            output_layer.append(
                get_output_activation(arch_args.output_activation),
            )

        self.model_output = nn.Sequential(*output_layer)


    def _regression(self, x: Tensor) -> Tensor:
        """Return model output for an input batch for a regression task.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk, in_components)

        Returns
        -------
            (Tensor):       Model output of shape
                            (batch_size, out_chunk, out_components)

        """
        self._internal.initialize(batch_size=x.shape[0])

        hidden, (h, c) = self.lstm_layers(
            x,
            (self._internal.h0.to(x.device), self._internal.c0.to(x.device)),
        )

        # TODO(randle): Tracking hidden states across batches is work in
        # progress.
        # https://github.com/MUST-Deep-Learning/KnowIt/issues/131
        if self._internal.tracking:
            self._internal.update(hn=h, cn=c)

        hidden = hidden.reshape(
            hidden.shape[0], hidden.shape[1] * hidden.shape[2],
        )
        out = self.model_output(
            hidden,
        )
        return out.view(
            x.shape[0], self.final_out_shape[0], self.final_out_shape[1],
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
        self._internal.initialize(batch_size=x.shape[0])

        hidden, (h, c) = self.lstm_layers(
            x,
            (self._internal.h0.to(x.device), self._internal.c0.to(x.device)),
        )

        # TODO(randle): Tracking hidden states across batches is work in
        # progress.
        # https://github.com/MUST-Deep-Learning/KnowIt/issues/131
        if self._internal.tracking:
            self._internal.update(hn=h, cn=c)

        hidden = hidden.reshape(
            hidden.shape[0], hidden.shape[1] * hidden.shape[2],
        )

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
        if self.task_name == "regression":
            return self._regression(x)

        return self._classification(x)


class InternalState:
    """Initialize hidden node and cell states.

    The class will initialize custom hidden and cell states for time step
    t0 = t1 - 1. The states can be initialized on each batch or the final
    states of a previous batch can be passed as the initial states of a batch.
    """

    def __init__(
            self,
            width: int,
            depth: int,
            init_hidden_state: None | str | Tensor,
            init_cell_state: None | str | Tensor,
            *,
            track_hidden: bool,
            bidirect: bool,
    ) -> None:
        self.width = width
        self.depth = depth
        self.init_hidden_state = init_hidden_state
        self.init_cell_state = init_cell_state

        # TODO(randle): Tracking hidden states across batches is work in
        # progress.
        # https://github.com/MUST-Deep-Learning/KnowIt/issues/131
        self.tracking = track_hidden
        if track_hidden:
            logger.info("Tracking internal LSTM state set to True.")
        self.d = 2 if bidirect else 1


    def initialize(self, batch_size: int) -> None:
        """Initialize hidden and cell states."""
        if not self.init_hidden_state or self.init_hidden_state == "zeros":
            h0 = torch.zeros(
                size=(self.d * self.depth, batch_size, self.width),
            )
        elif self.init_hidden_state == "random":
            h0 = torch.rand(
                size=(self.d * self.depth, batch_size, self.width),
            )
        else:
            logger.error("Choice for initial state must be: zeros or random\
                        (default: zeros).")
            sys.exit()

        if not self.init_cell_state or self.init_cell_state == "zeros":
            c0 = torch.zeros(
                size=(self.d * self.depth, batch_size, self.width),
            )
        elif self.init_cell_state == "random":
            c0 = torch.randn(
                size=(self.d * self.depth, batch_size, self.width),
            )
        else:
            logger.error("Choice for initial state must be: zeros or random\
                        (default: zeros).")
            sys.exit()

        self.update(cn=c0, hn=h0)

    def update(self, cn: Tensor, hn: Tensor) -> None:
        """Update values for the cell and the hidden state."""
        self.c0 = cn
        self.h0 = hn


def get_output_activation(
        output_activation: None | str,
) -> Module:
    """Fetch output activation function from Pytorch."""
    if output_activation == "Softmax":
        return getattr(nn, output_activation)(dim=1)
    if output_activation == "Sigmoid":
        return getattr(nn, output_activation)()
    return nn.Identity()
