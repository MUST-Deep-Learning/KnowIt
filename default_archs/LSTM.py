"""
-----
LSTM
-----

An example of a LSTM architecture using Pytorch.

The LSTM consists of recurrent cells that contain hidden states and cell
states. The hidden and cell states are initialized for each batch to a zero
tensor or a random tensor. Additionally, the LSTM can be set to bidirectional.

The example also shows how dataclasses can be used with KnowIt.

Example usage:
-------------

from knowit import KnowIt
from default_archs.LSTM import ArchArgs, Internal

KI = KnowIt()

model_name = "my_new_penguin_model"
data_args = {'name': 'penguin_42_debug',
             'task': 'classification',
             'in_components': ['accX', 'accY', 'accZ'],
             'out_components': ['PCE'],
             'in_chunk': [-25, 25],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 256,
             'split_method': 'chronological',
             'scaling_tag': 'in_only',
             'min_slice': 100}
arch_args = {'task': 'classification',
             'name': 'LSTM',
             'arch_hps': {
                 'arch_args': ArchArgs(
                    dropout=0.5,
                    width=256,
                    bidirectional=False,
                 ),
                'internal_state': Internal(
                    init_cell_state='zeros',
                    init_hidden_state='zeros',
                ),
             }}
trainer_args = {'loss_fn': 'weighted_cross_entropy',
                'optim': 'Adam',
                'max_epochs': 30,
                'learning_rate': 0.01,
                'learning_rate_scheduler': {
                    'ReduceLROnPlateau': {'mode': 'min', 'patience': 15},
                },
                'task': 'classification'}
KI.train_model(
    model_name=model_name,
    kwargs={'data': data_args, 'arch': arch_args, 'trainer': trainer_args},
)
"""

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Contains an example of a LSTM architecture."

import sys
from dataclasses import dataclass

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


@dataclass
class ArchArgs(yaml.YAMLObject):
    """Container for architecture parameters.

    Parameters
    ----------
    width : int, optional
        The width of the architecture (default is 256).
    depth : int, optional
        The depth of the architecture (default is 1).
    dropout : float, optional
        The dropout rate (default is 0.5).
    output_activation : None or str, optional
        The activation function for the output layer (default is None).
    bidirectional : bool, optional
        Indicates whether the architecture is bidirectional (default is False).

    Attributes
    ----------
    yaml_tag : str
        A unique YAML tag for identifying the architecture parameters.
    """

    yaml_tag = "!arch_args"

    width: int = 256
    depth: int = 1
    dropout: float = 0.5
    output_activation: None | str = None
    bidirectional: bool = False


@dataclass
class Internal(yaml.YAMLObject):
    """Container for internal states and configuration.

    Parameters
    ----------
    init_hidden_state : None or str or Tensor, optional
        The initial hidden state for the model (default is None).
    init_cell_state : None or str or Tensor, optional
        The initial cell state for the model (default is None).
    tracking : bool, optional
        Flag to indicate if tracking is enabled (default is False).

    Attributes
    ----------
    yaml_tag : str
        A unique YAML tag for identifying the internal configuration.
    """

    yaml_tag = "!internal"

    init_hidden_state: None | str | Tensor = None
    init_cell_state: None | str | Tensor = None
    tracking: bool = False


class Model(Module):
    """Define an LSTM architecture.

    The long short-term memory architecture is a type of recurrent gated neural
    network with nonlinear activation functions.

    For more information, see:

    [1] https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext

    Parameters
    ----------
    input_dim : list[int], shape = (in_chunk, in_components)
        The shape of the input data. The "time axis" is along the first
        dimension.
    output_dim : list[int], shape = (in_chunk, in_components)
        The shape of the output data. The "time axis" is along the first
        dimension.
    task_name : str
        The type of task (classification or regression).
    arch_args : None | ArchArgs, default=None
        The parameters that define the architecture (see ArchArgs).
    internal_state : None | Internal, default=None
        The parameters that define the internal state of the LSTM
        (see Internal).

    Attributes
    ----------
    input_size : int
        The size of the input features.
    model_out_dim : int
        The total number of output features.
    final_out_shape : list[int]
        The shape of the final output.
    lstm_layers : LSTM
        The LSTM layers of the model.
    model_output : nn.Sequential
        The sequential output layers, including the final activation function
        if the task is classification.
    _internal : InternalState
        The internal state configuration of the LSTM.
    task_name : str
        The name of the task for which the model is configured.

    Notes
    -----
    This class builds a multi-layer LSTM network, which can be used for
    various sequence prediction tasks, including classification and regression.
    The architecture is configurable through the `arch_args` and
    `internal_state` parameters, allowing for flexibility in model design.
    """

    def __init__(
        self,
        input_dim: list[int],
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
            hidden.shape[0],
            hidden.shape[1] * hidden.shape[2],
        )
        out = self.model_output(
            hidden,
        )
        return out.view(
            x.shape[0],
            self.final_out_shape[0],
            self.final_out_shape[1],
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
            hidden.shape[0],
            hidden.shape[1] * hidden.shape[2],
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

    This class manages the initialization and updating of the hidden and
    cell states for an LSTM architecture, allowing for both custom
    initialization and tracking of states across batches.

    Parameters
    ----------
    width : int
        The number of features in the hidden state.
    depth : int
        The number of LSTM layers.
    init_hidden_state : None | str | Tensor
        The initial hidden state. Can be "zeros", "random", or a Tensor.
    init_cell_state : None | str | Tensor
        The initial cell state. Can be "zeros", "random", or a Tensor.
    track_hidden : bool
        Flag to indicate if hidden states should be tracked across batches.
    bidirect : bool
        Flag to indicate if the LSTM is bidirectional.

    Attributes
    ----------
    width : int
        The number of features in the hidden state.
    depth : int
        The number of LSTM layers.
    init_hidden_state : None | str | Tensor
        The initial hidden state configuration.
    init_cell_state : None | str | Tensor
        The initial cell state configuration.
    tracking : bool
        Indicates if hidden states are being tracked.
    d : int
        The number of directions in the LSTM (1 for unidirectional, 2 for
        bidirectional).
    c0 : Tensor
        The current cell state.
    h0 : Tensor
        The current hidden state.
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
        """Initialize hidden and cell states.

        This method sets the initial values for the hidden and cell states
        based on the specified initialization strategy.

        Parameters
        ----------
        batch_size : int
            The size of the batch for which the hidden and cell states are
            initialized.

        Raises
        ------
        SystemExit
            If the initialization choice for hidden or cell states is invalid.
        """
        if not self.init_hidden_state or self.init_hidden_state == "zeros":
            h0 = torch.zeros(
                size=(self.d * self.depth, batch_size, self.width),
            )
        elif self.init_hidden_state == "random":
            h0 = torch.rand(
                size=(self.d * self.depth, batch_size, self.width),
            )
        else:
            logger.error(
                "Choice for initial state must be: zeros or random\
                        (default: zeros).",
            )
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
            logger.error(
                "Choice for initial state must be: zeros or random\
                        (default: zeros).",
            )
            sys.exit()

        self.update(cn=c0, hn=h0)

    def update(self, cn: Tensor, hn: Tensor) -> None:
        """Update values for the cell and the hidden state.

        This method updates the current cell state and hidden state with
        the provided values.

        Parameters
        ----------
        cn : Tensor
            The new cell state to set.
        hn : Tensor
            The new hidden state to set.
        """
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
