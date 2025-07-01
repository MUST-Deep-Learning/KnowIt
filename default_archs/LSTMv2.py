"""This is a set of stacked LSTM blocks,
with optional normalization, dropout, bidirectionality, and residual connections,
followed by a final linear layer, with optional output activation.
Additionally, the architecture can be stateful or stateless.

---------
LSTMBlock
---------

This block consists of 2 to 4 layers.
* is optional

[lstm] -> [layernorm*] -> [dropout*] -> [residual*]

    -   [lstm] = torch.nn.LSTM
    -   [layernorm*] = torch.nn.LayerNorm
    -   [dropout*] = torch.nn.Dropout
    -   [residual*] = a residual connection

Notes
-----
    - The LSTM is capable of handling regression or classification tasks.
    - All LSTMBlocks have bias parameters.
    - All LSTMBlocks have the same number of hidden units, defined by the ``width`` parameter.

""" # noqa: INP001, D415, D400, D212, D205

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "tiantheunissen@gmail.com, randlerabe@gmail.com"
__description__ = "Contains an example of a LSTM architecture."

from numpy import arange, prod
from torch import Tensor, zeros, randn, randn_like, nn, ones, cat
from torch.nn import (LSTM, Linear, Module, LayerNorm,
                      Dropout, ModuleList, Sequential, Identity)

from helpers.logger import get_logger

logger = get_logger()

available_tasks = ("regression", "classification")

HP_ranges_dict = {"width": range(2, 1025, 1),
                  "depth": range(2, 1025, 1),
                  "dropout": arange(0, 1.1, 0.1),
                  "output_activation": (None, "Sigmoid", "Softmax"),
                  "stateful": (False, True),
                  "hc_init_method": ('zeros', 'random'),
                  "bidirectional": (False, True),
                  "residual": (False, True)}

class Model(Module):
    """A stacked LSTM model for sequence processing with configurable depth,
    bidirectional processing, and output layers.

    This model consists of multiple LSTMBlock layers followed by a linear output layer and an optional activation function.
    It supports stateful processing for sequential data, handling tasks like regression or classification.

    Parameters
    ----------
    input_dim : list[int], shape=[in_chunk, in_components]
        The shape of the input data. The "time axis" is along the first dimension.
    output_dim : list[int], shape=[out_chunk, out_components]
        The shape of the output data. The "time axis" is along the first dimension.
    task_name : str
        The type of task (classification or regression).
    width : int, default=256
        The number of features in the hidden state of each LSTMBlock.
    depth : int, default=2
        The number of LSTMBlocks.
    dropout : float, default=0.0
        Dropout probability applied in each LSTMBlock.
    output_activation : str or None, default=None
        The activation function for the output layer ('Sigmoid', 'Softmax', or None).
    stateful : bool, default=False
        If True, maintains hidden states across batches for contiguous sequences.
    hc_init_method : str, default='zeros'
        Method for initializing hidden and cell states in LSTMBlocks ('zeros' or 'random').
    layernorm : bool, default=True
        If True, applies layer normalization in each LSTMBlock.
    bidirectional : bool, default=False
        If True, uses bidirectional LSTMBlocks.
    residual : bool, default=False
        If True, applies residual connections in LSTMBlocks if sizes match.

    Attributes
    ----------
    task_name : str
        The task type ('regression' or 'classification').
    model_in_dim : int
        The input feature size (last dimension of input_dim).
    model_out_dim : int
        The flattened output size (product of output_dim).
    final_out_shape : list[int]
        The desired output shape.
    stateful : bool
        Whether the model maintains hidden states across batches.
    last_ist_idx : torch.Tensor or None
        The last batch indices used for stateful processing, or None if not set.
    lstm_layers : torch.nn.ModuleList
        List of LSTMBlock instances.
    output_layers : torch.nn.Sequential
        Sequential container of the output linear layer and optional activation function.

    Notes
    -----
    - The output shape is reshaped to match `output_dim` (e.g., [batch_size, out_chunk, out_components]).
    - Stateful processing requires `ist_idx` in the input batch to track sequence continuity.
    - The `output_activation` is applied only if specified, typically for classification tasks.
    """

    task_name = None
    model_in_dim = None
    model_out_dim = None
    final_out_shape = None
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
        dropout: float = 0.0,
        output_activation: str | None = None,
        stateful: bool = False,
        hc_init_method: str = 'zeros',
        layernorm: bool = True,
        bidirectional: bool = False,
        residual: bool = True
    ) -> None:
        super().__init__()

        self.task_name = task_name
        self.model_in_dim = input_dim[-1]
        self.model_out_dim = int(prod(output_dim))
        self.final_out_shape = output_dim
        self.stateful = stateful

        self.lstm_layers = ModuleList()
        direction = 2 if bidirectional else 1
        for d in range(depth):
            self.lstm_layers.append(
                LSTMBlock(
                    input_size=self.model_in_dim if d == 0 else direction * width,
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
        self.output_layers.append(Linear(in_features=direction * width * input_dim[-2],
                                        out_features=self.model_out_dim,
                                        bias=False))
        if output_activation is not None:
            self.output_layers.append(get_output_activation(output_activation))
        self.output_layers = Sequential(*self.output_layers)

    def force_reset(self):
        """ A function for external modules to manually signal that all hidden and internal states need to be reset."""
        self.last_ist_idx = None

    def _reset_all_layer_states(self, batch_size, device, changed_idx=None) -> None:
        """Reset the hidden and cell states of all LSTMBlock layers.

        Parameters
        ----------
        batch_size : int
            The batch size for the hidden and cell states.
        device : str
            The device to place the hidden and cell states on (e.g., 'cuda' or 'cpu').
        changed_idx : torch.Tensor, default=None
            Boolean mask tensor of shape (batch_size,) indicating which batch indices to reset.
        """
        for layer in self.lstm_layers:
            layer.reset_states(batch_size=batch_size, device=device, changed_idx=changed_idx)

    def _detach_all_layer_hidden_states(self) -> None:
        """Detach the hidden and cell states of all LSTMBlock layers to stop gradient tracking.

        Notes
        -----
        This is typically used in stateful mode to break gradient computation across batches.
        """
        for layer in self.lstm_layers:
            layer.hidden_state = (layer.hidden_state[0].detach(), layer.hidden_state[1].detach())

    def update_states(self, ist_idx, device) -> None:
        """Manage hidden state continuity for stateful processing based on batch indices.

        Parameters
        ----------
        ist_idx : torch.Tensor
            Tensor of shape (batch_size, 3) containing batch indices for tracking sequence continuity.
        device : str
            The device to place the hidden and cell states on (e.g., 'cuda' or 'cpu').

        Notes
        -----
        - If `stateful=True`, hidden states are preserved for contiguous sequences and reset for non-contiguous ones.
        - `ist_idx` is expected to have columns [instance_id, slice_id, time_step_id].
        - Non-contiguous sequences are detected by comparing `ist_idx` with `last_ist_idx`.
        """

        def _is_contiguous(a, b):
            """ Determine which prediction points are contiguous with last batch. """
            same_i = a[:, 0] == b[:, 0]
            same_s = a[:, 1] == b[:, 1]
            next_t = a[:, 2] + 1 == b[:, 2]
            result = same_i & same_s & next_t
            return result

        if self.stateful:
            if not self.last_ist_idx is None:
                # data has been passed before, detach all hidden states
                self._detach_all_layer_hidden_states()
                if self.last_ist_idx.shape == ist_idx.shape:
                    # new batch is same shape as last batch, just reset at breaks in contiguousness
                    changed = ~_is_contiguous(self.last_ist_idx, ist_idx)
                elif self.last_ist_idx.shape[0] > ist_idx.shape[0]:
                    # new batch is smaller than last batch
                    changed = ~_is_contiguous(self.last_ist_idx[:ist_idx.shape[0], :], ist_idx[:ist_idx.shape[0], :])
                else:
                    # new batch is larger than last batch
                    changed = ~_is_contiguous(self.last_ist_idx, ist_idx[:self.last_ist_idx.shape[0], :])
                    this = ones(size=(ist_idx.shape[0] - self.last_ist_idx.shape[0],), dtype=bool).to(device)
                    changed = cat((changed, this))
                self._reset_all_layer_states(ist_idx.shape[0], device, changed)
            else:
                # first time data is passed through model, reset everything
                self._reset_all_layer_states(ist_idx.shape[0], device)
            self.last_ist_idx = ist_idx.clone()
        else:
            self._reset_all_layer_states(ist_idx.shape[0], device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model, processing input through LSTM layers and output layers.

        Parameters
        ----------
        x : Tensor, shape=[batch_size, in_chunk, in_components]
            An input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_size) matching `final_out_shape`.

        Raises
        ------
        SystemExit
            If `task_name` is neither 'regression' nor 'classification', exits with code 101.

        Notes
        -----
        - The input is processed sequentially through each LSTMBlock, followed by reshaping and output layers.
        - For 'regression', the output is reshaped to match `final_out_shape`.
        - For 'classification', the output is returned as is (assumes activation like softmax is in output_layers).
        """

        if x.shape[0] != self.lstm_layers[0].hidden_state[0].shape[1]:
            logger.warning('Unplanned reset of hidden state in LSTMv2! '
                           'This was done due to missmatch in batch size and hidden state expectation. '
                           'Statefulness at least partially lost.')
            self._reset_all_layer_states(x.shape[0], x.device)

        hidden = x
        for i, layer in enumerate(self.lstm_layers):
            hidden = layer(hidden)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1] * hidden.shape[2])
        out = self.output_layers(hidden)

        if self.task_name == 'regression':
            out = out.view(hidden.shape[0], self.final_out_shape[0], self.final_out_shape[1])
        elif self.task_name == 'classification':
            pass
        else:
            logger.error('Unknown task name %s, for LSTM.', self.task_name)
            exit(101)

        return out


class LSTMBlock(Module):
    """A customizable LSTM block with optional bidirectional processing, layer normalization,
    dropout, and residual connections.

    This module wraps a PyTorch LSTM layer with additional features such as layer normalization, dropout, and residual
    connections. It also supports customizable initialization of hidden and cell states.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int
        Number of recurrent layers.
    dropout : float
        Dropout probability to apply to the LSTM output. If 0, no dropout is applied.
    batch_first : bool
        If True, input and output tensors are provided as (batch, seq, feature). Otherwise, (seq, batch, feature).
    hc_init_method : str
        Method for initializing hidden and cell states: 'zeros' or 'random'.
    layernorm : bool
        If True, applies layer normalization to the LSTM output.
    bidirectional : bool
        If True, uses a bidirectional LSTM.
    residual : bool
        If True, applies a residual connection between input and output if their sizes match.

    Attributes
    ----------
    hidden_size : int
        The number of features in the hidden state of the LSTM.
    num_layers : int
        The number of LSTM layers.
    hc_init_method : str
        The method for initializing hidden and cell states ('zeros' or 'random').
    bidirectional : bool
        If True, the LSTM is bidirectional.
    residual : bool
        If True, applies a residual connection if input and output sizes match.
    hidden_state : tuple of Tensor
        The hidden and cell states of the LSTM, with shapes (num_layers * num_directions, batch_size, hidden_size).
    lstm_layer : torch.nn.LSTM
        The underlying LSTM layer.
    layer_norm : torch.nn.Module
        The layer normalization module (LayerNorm if layernorm=True, else Identity).
    dropout : torch.nn.Module | Identity
        The dropout module (Dropout if dropout > 0, else Identity).

    Notes
    -----
    - The LSTM output size is `hidden_size * 2` if bidirectional, otherwise `hidden_size`.
    - Residual connections are only applied if `residual=True` and the input size matches the output size.
    - Hidden and cell states are initialized on the specified device (default: 'cuda') during `reset_states`.
    """

    hidden_size = None
    num_layers = None
    hc_init_method = None
    bidirectional = None
    residual = None
    hidden_state = None

    def __init__(self, input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 batch_first: bool,
                 hc_init_method: str,
                 layernorm: bool,
                 bidirectional: bool,
                 residual: bool) -> None:
        super(LSTMBlock, self).__init__()

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

        d = 2 if self.bidirectional else 1
        self.layer_norm = LayerNorm(normalized_shape=d*hidden_size) if layernorm else Identity()

        self.dropout = Dropout(dropout) if dropout > 0 else Identity()

        self.reset_states(batch_size=1, device='cuda')

    def reset_states(self, batch_size, device, changed_idx=None) -> None:
        """Initialize or reset the hidden and cell states of the LSTM.

        Parameters
        ----------
        batch_size : int
            The batch size for the hidden and cell states.
        device : str
            The device to place the hidden and cell states on (e.g., 'cuda' or 'cpu').
        changed_idx : int, optional
            If provided, reset only the specified indices of the hidden and cell states. Default is None.

        Raises
        ------
        SystemExit
            If `hc_init_method` is neither 'zeros' nor 'random', exits with code 101.
        """
        if changed_idx is None or changed_idx.sum().item() == batch_size:
            # either the first pass of data through the model or all indices need resetting
            d = 2 if self.bidirectional else 1
            if self.hc_init_method == 'zeros':
                h = zeros(self.num_layers * d, batch_size, self.hidden_size).to(device)
                c = zeros(self.num_layers * d, batch_size, self.hidden_size).to(device)
            elif self.hc_init_method == 'random':
                h = randn(self.num_layers * d, batch_size, self.hidden_size).mul_(0.1).to(device)
                c = randn(self.num_layers * d, batch_size, self.hidden_size).mul_(0.1).to(device)
            else:
                logger.error('Unknown hc_init_method %s ', self.hc_init_method)
                exit(101)
            self.hidden_state = (h, c)
        elif changed_idx.sum().item() == 0 and self.hidden_state[0].shape[1] == batch_size:
            # no indices need resetting and new batch size is old batch size, just ensure correct device
            self.hidden_state = (self.hidden_state[0].to(device), self.hidden_state[1].to(device))
        else:
            # some partial resetting required
            self._partial_reset(batch_size, device, changed_idx)

    def _partial_reset(self, batch_size: int, device: str, changed_idx: Tensor) -> None:
        """Helper method to reset specific indices of the hidden and cell states.

        Parameters
        ----------
        batch_size : int
            The batch size for the next hidden and cell states.
        device : str
            The device to place the hidden and cell states on (e.g., 'cuda' or 'cpu').
        changed_idx : torch.Tensor
            Boolean mask tensor of shape (batch_size,) indicating which batch indices to reset.

        Raises
        ------
        ValueError
            If `hc_init_method` is invalid or `hidden_state` is not initialized.
        """

        def _apply_sub_reset(t, mask, mode):
            """ Directly resets specific indices based on a mask. """
            if mode == 'zeros':
                t[:, mask, :] = 0.
            elif mode == 'random':
                t[:, mask, :] = randn_like(t[:, mask, :]).mul_(0.1)
            else:
                logger.error('Unknown hc_init_method %s ', self.hc_init_method)
                exit(101)
            return t

        def _gen_extra(t, l, mode):
            """ Generates a tensor to add based on the given tensor and length. """
            if mode == 'zeros':
                to_add = zeros(size=(t.shape[0], l, t.shape[2])).mul_(0.1).to(device)
            elif mode == 'random':
                to_add = randn(size=(t.shape[0], l, t.shape[2])).mul_(0.1).to(device)
            else:
                logger.error('Unknown hc_init_method %s ', self.hc_init_method)
                exit(101)
            return to_add

        if self.hidden_state is None:
            self.reset_states(batch_size, device)

        h = self.hidden_state[0].to(device)
        c = self.hidden_state[1].to(device)
        old_batch_size = self.hidden_state[0].shape[1]

        if old_batch_size == batch_size:
            h = _apply_sub_reset(h, changed_idx, self.hc_init_method)
            c = _apply_sub_reset(c, changed_idx, self.hc_init_method)
        elif batch_size < old_batch_size:
            h = h[:, :batch_size, :]
            c = c[:, :batch_size, :]
            h = _apply_sub_reset(h, changed_idx, self.hc_init_method)
            c = _apply_sub_reset(c, changed_idx, self.hc_init_method)
        else:
            h = _apply_sub_reset(h, changed_idx[:old_batch_size], self.hc_init_method)
            c = _apply_sub_reset(c, changed_idx[:old_batch_size], self.hc_init_method)
            h = cat((h, _gen_extra(h, batch_size - old_batch_size, self.hc_init_method)), dim=1)
            c = cat((c, _gen_extra(c, batch_size - old_batch_size, self.hc_init_method)), dim=1)
        self.hidden_state = (h, c)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the LSTM block.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (batch, seq, input_size) if batch_first=True, else (seq, batch, input_size).

        Returns
        -------
        Tensor
            Output tensor with shape (batch, seq, hidden_size * num_directions) if batch_first=True,
            else (seq, batch, hidden_size * num_directions).

        Notes
        -----
        - Applies the LSTM layer, followed by optional layer normalization, residual connection, and dropout.
        - The residual connection is only applied if `residual=True` and the input size matches the output size.
        """
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
    return Identity()