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

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import LSTM, Linear, Module

available_tasks = ("regression", "classification")

HP_ranges_dict = {
    "arch_args": {
        "width": range(2, 1025, 1),
        "depth": range(2, 1025, 1),
        "num_hidden_to_out": "varies according to width",
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

    num_hidden_to_out: int
    width: int = 256
    depth: int = 1
    dropout: float = 0.5
    output_activation: None | str = None
    bidirectional: bool = False
    path: str = __file__



@dataclass(frozen=True)
class Internal(yaml.YAMLObject):
    yaml_tag = "!internal"

    init_hidden_state: None | str | Tensor=None
    init_cell_state: None | str | Tensor=None
    tracking: bool=False
    path: str = __file__


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
        self.final_out_dim = output_dim

        # remember to move h0 and c0 to device

        if not arch_args:
            # revert to default arg values in ArchArgs
            arch_args = ArchArgs(num_hidden_to_out=1)

        # input layer (input x will return three tensors: output, (hn, cn))
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
                in_features=arch_args.width,
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
        if len(self._internal.h0.shape) == 2:
            # add batch dim
            # bho & ch0 resulting shapes (depth, batch, width)
            hn = torch.broadcast_to(
                input=self._internal.h0,
                size=(
                    self._internal.h0.shape[0],
                    x.shape[0],
                    self._internal.h0.shape[1],
                ),
            )
            cn = torch.broadcast_to(
                input=self._internal.c0,
                size=(
                    self._internal.c0.shape[0],
                    x.shape[0],
                    self._internal.c0.shape[1],
                ),
            )
            self._internal.update(hn=hn, cn=cn)
        elif self._internal.h0.shape[1] > x.shape[0]:
            # batch dim exists but needs to be modified according to x.
            hn = torch.narrow(
                input=self._internal.h0,
                dim=1,
                start=0,
                length=int(x.shape[0]),
            )
            cn = torch.narrow(
                input=self._internal.c0,
                dim=1,
                start=0,
                length=int(x.shape[0]),
            )
            self._internal.update(hn=hn, cn=cn)
        elif self._internal.h0.shape[1] < x.shape[0]:
            # batch dim exists but needs to be modified according to x.
            hn = self._internal.h0[:, 1, :]
            hn = torch.unsqueeze(hn, -1)
            hn = torch.transpose(hn, 1, 2)
            hn = torch.broadcast_to(
                input=hn,
                size=(
                    self._internal.h0.shape[0],
                    x.shape[0],
                    self._internal.h0.shape[2],
                ),
            )
            cn = self._internal.c0[:, 1, :]
            cn = torch.unsqueeze(cn, -1)
            cn = torch.transpose(cn, 1, 2)
            cn = torch.broadcast_to(
                input=cn,
                size=(
                    self._internal.c0.shape[0],
                    x.shape[0],
                    self._internal.c0.shape[2],
                ),
            )
            self._internal.update(hn=hn, cn=cn)

        hidden, (h, c) = self.lstm_layers(
            x,
            (self._internal.h0.to(x.device), self._internal.c0.to(x.device)),
        )
        if self._internal.tracking:
            self._internal.update(hn=h, cn=c)

        return self.model_output(
            hidden,
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
        if self.handc:
            x_tuple = (x, (self.h_0, self.c_0))
            hidden = self.lstm_layers(*x_tuple)[0]
        else:
            hidden = self.lstm_layers(x)[0]
        hidden = torch.reshape(
            hidden, (hidden.shape[0], hidden.shape[1] * hidden.shape[2]),
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

        self._current_batch: int=0
        self.tracking: bool = track_hidden
        _d = 2 if bidirect else 1

        # initialize
        if self._current_batch == 0:
            if not init_hidden_state:
                h0: Tensor = torch.zeros(size=(_d * depth, width))
            elif init_hidden_state == "random":
                h0: Tensor = torch.randn(size=(_d * depth, width))
            elif isinstance(init_hidden_state, Tensor):
                h0: Tensor = init_hidden_state

            if not init_cell_state:
                c0: Tensor = torch.zeros(size=(_d * depth, width))
            elif init_cell_state == "random":
                c0: Tensor = torch.randn(size=(_d * depth, width))
            elif isinstance(init_cell_state, Tensor):
                c0: Tensor = init_cell_state

            self.update(cn=c0, hn=h0)
            self._current_batch += 1

    def update(self, cn: Tensor, hn: Tensor) -> None:
        """Update init values for the cell and the hidden state."""
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
