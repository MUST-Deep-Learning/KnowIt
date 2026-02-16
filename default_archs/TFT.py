"""
This module implements a simplified Temporal Fusion Transformer (TFT) architecture.

Unlike the original TFT,
this version omits the encoding of exogenous inputs (e.g. static covariates or future known inputs).
It is tailored for regression, classification, or variable length regression tasks,
using temporal variable selection, gating mechanisms, LSTM-based encoding, and Interpretable Multi-Head Attention.

The following diagram depicts the overall architecture:

    X → [EmbeddingLayer] → [VariableSelectionNetwork] → [LSTM]* → [GateAddNorm] →
    [InterpretableMultiHeadAttention]* → [GateAddNorm] → [GatedResidualNetwork] → [Dense] → Y

"*" indicates a skip connection going bypassing this module from the past to the next.

See:
[1] "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"  
    Lim et al., 2019 — https://arxiv.org/abs/1912.09363

Inspiration for this implementation has been taken from `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_.

Note: The LSTM stage of the TFT makes use of the default LSTMv2 architecture within KnowIt. All stateful processing is handled by
this underlying LSTM module.

--------------
EmbeddingLayer
--------------

Performs the initial embedding of input components into a representation space for further processing.
Linear embeddings are done in one of two modes:

* independent (default): input components are independently embedded,
* mixed: the embedding of each input component depends on all other input component.

----
Gate
----

    → [Dropout] → [Linear] → [GLU] →

The main gating mechanism in the TFT.
It consists of a linear layer and a Gated Linear Unit (GLU) function,
with optional dropout before the linear layer.

This module is used as a basic building block by other modules in the architecture.

-----------
GateAddNorm
-----------

    → → [Gate] → [LayerNorm] →
        → → → → → → → ↑

Applies the gating mechanism, an optional residual connection from outside the module, and layer normalization.

This module is used as both a building block and as part of the main architectural flow.
Specifically, it is used after the LSTM encoder and Attention blocks to skip over these components
and dynamically calibrate the complexity of the overall architecture.

--------------------
GatedResidualNetwork
--------------------

    → → [Linear] → [ELU] → [Linear] → [GateAddNorm] →
      ↓ → → → → → → → → → → → → → → → → → → ↑

A gated feedforward block:
    - Enables non-linear transformations with gating control.
    - Residual connection allows gradient flow and feature reuse.
    - LayerNorm stabilizes outputs.

This module is used as both a building block and as part of the main architectural flow.
Specifically, it is used towards the end of the architecture as a final feedforward stage.

------------------------
VariableSelectionNetwork
------------------------

    → → → [GatedResidualNetwork] → → → → → → → → [Weighted sum] →
      ↓ → [GatedResidualNetwork] → [Softmax] → → → → → ↑

This module performs a feature selection step before passing on the input to the LSTM encoder.
Specifically, a weighted sum of (processed by GatedResidualNetwork) input values are returned,
where the weights are determined by a variable selection mechanism done by a different dedicated
GatedResidualNetwork and a softmax function.

-------------------------
ScaledDotProductAttention
-------------------------

Performs basic dot product self attention.
Optionally incorporates attention dropout and causality masking.

-------------------------------
InterpretableMultiHeadAttention
-------------------------------

Performs interpretable multi-head attention, where the query and key operations are done for each head separately,
but heads share a value component. This is said to improve the interpretability of the resulting attention weights.

While the LSTM is meant to model local information (and encode positional information) the
InterpretableMultiHeadAttention module is intended to capture long-term dependencies in the data.


Notes
-----
    - TFT handles **only temporal inputs**, and is suitable for forecasting, regression, and classification.
    - The architecture **does not use static encodings**.
    - All weights (excluding biases) are initialized with `nn.init.kaiming_uniform_` if dimensions allow,
      otherwise with `nn.init.normal_`.
    - All bias parameters are initialized to zero.
"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "potgieterharmen@gmail.com, tiantheunissen@gmail.com"
__description__ = "Example of an TFT model architecture."

import torch.nn.functional
from torch import (Tensor, zeros, bool, arange, cat, float, stack, ones, as_tensor, bmm, mean, nn, tril, finfo)
from torch.nn import (Module, LSTM, Parameter, Sigmoid, Linear, ModuleList, LayerNorm, ELU, Dropout, Softmax, Identity, Sequential)
from torch.nn.functional import glu, softmax
from numpy import prod
from torch.nn.init import (zeros_, kaiming_uniform_, normal_)
from typing import Dict, Tuple, List, Optional
from helpers.logger import get_logger
import numpy as np
from default_archs.LSTMv2 import Model as KnowItLSTM

logger = get_logger()

available_tasks = ("regression", "classification", "vl_regression")

HP_ranges_dict = {"depth": range(1, 512, 1),
                  "lstm_depth": range(1, 64, 1),
                  "num_attention_heads": range(1, 4, 1),
                  "dropout": np.arange(0, 1.1, 0.1),
                  "output_activation": (None, "Sigmoid", "Softmax"),
                  "decoder_time_steps": range(1, 1025, 1),
                  "stateful": (False, True),
}

class Model(Module):
    """
    Temporal Fusion Transformer (TFT)-style model.

    This module implements a streamlined Temporal Fusion Transformer architecture
    consisting of:

    - Per-feature embedding via ``EmbeddingLayer``
    - Feature-wise selection through a ``VariableSelectionNetwork``
    - Recurrent sequence modeling using a custom ``LSTM`` encoder
    - Static skip connections using ``GateAddNorm``
    - Interpretable multi-head self-attention
    - Decoder refinement via ``GatedResidualNetwork``
    - Task-dependent output projection layer

    The model supports standard regression, variable-length regression
    (``'vl_regression'``), and classification tasks.

    Parameters
    ----------
    input_dim : list[int], shape=[in_chunk_size, in_components]
        The shape of the input data. The "time axis" is along the first dimension.
        If variable length data is processed, then in_chunk_size must be 1.
    output_dim : list[int], shape=[out_chunk_size, out_components]
        The shape of the output data. The "time axis" is along the first dimension.
        If variable length data is processed, then out_chunk_size must be 1.
    task_name : str
        The type of task ('classification', 'regression', or 'vl_regression').
    embedding_mode : str, default='independent'
        Embedding strategy. Must be either "independent" or "mixed".
        The former will embed input components independently and the latter will mix information during embedding.
    hidden_dim : int, default=32
        Hidden dimensionality used throughout embeddings, attention,
        and gated residual blocks.
    lstm_depth : int, default=4
        Number of stacked LSTM layers in the encoder. See LSTMv2 for details.
    lstm_width : int, default=64
        Internal width of the LSTM layers. See LSTMv2 for details.
    lstm_hc_init_method : str, default='zeros'
        Initialization strategy for LSTM hidden and cell states. See LSTMv2 for details.
    lstm_layernorm : bool, default=True
        If True, applies layer normalization inside the LSTM. See LSTMv2 for details.
    lstm_bidirectional : bool, default=False
        If True, uses a bidirectional LSTM encoder. See LSTMv2 for details.
    num_attention_heads : int, default=4
        Number of heads in the interpretable multi-head attention module.
    dropout : float or None, default=0.2
        Dropout probability applied throughout the network. If None,
        dropout is disabled.
    output_activation : str or None, default=None
        Optional activation applied after the final linear output layer
        (only used for non-``'vl_regression'`` tasks).
    lstm_stateful : bool, default=True
        If True, the LSTM maintains internal states across forward passes.

    Attributes
    ----------
    embedder : EmbeddingLayer
        Per-variable embedding module.
    vsn : VariableSelectionNetwork
        Learns dynamic feature importance weights.
    lstm_encoder : LSTMV2
        Recurrent sequence encoder.
    attention : InterpretableMultiHeadAttention
        Self-attention mechanism operating over temporal dimension.
    decoder_grn : GatedResidualNetwork
        Decoder refinement block.
    output_layers : torch.nn.Module
        Final task-dependent projection layers.
    """

    def __init__(self,
                 input_dim: list,
                 output_dim: list,
                 task_name: str,
                 *,
                 embedding_mode: str = 'independent',
                 hidden_dim: int = 32,
                 lstm_depth: int = 4,
                 lstm_width: int = 64,
                 lstm_hc_init_method: str='zeros',
                 lstm_layernorm: bool = True,
                 lstm_bidirectional: bool = False,
                 num_attention_heads: int = 4,
                 dropout: float | None = 0.2,
                 output_activation: str | None = None,
                 lstm_stateful: bool = True,
    ) -> None:

        super().__init__()

        self.task_name = task_name
        self.model_in_dim = input_dim[-1]
        self.model_out_dim = int(prod(output_dim))
        self.final_out_shape = output_dim

        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.output_activation = output_activation

        self.embedder = EmbeddingLayer(n_inputs=self.model_in_dim,
                                       hidden_dim=self.hidden_dim,
                                       mode=embedding_mode)

        self.vsn = VariableSelectionNetwork(
            n_inputs=self.model_in_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.lstm_encoder = KnowItLSTM(input_dim=[1, self.hidden_dim],
                                       output_dim=[1, self.hidden_dim],
                                       task_name='vl_regression',
                                       stateful=lstm_stateful,
                                       depth=lstm_depth,
                                       dropout=self.dropout if self.dropout is not None else 0.0,
                                       width=lstm_width,
                                       hc_init_method=lstm_hc_init_method,
                                       layernorm=lstm_layernorm,
                                       bidirectional=lstm_bidirectional,
                                       residual=False) # <- False since there is a residual connection in TFT

        self.lstm_out_gan = GateAddNorm(
            n_inputs=self.hidden_dim,
            dropout=self.dropout,
        )

        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            n_heads=self.num_attention_heads,
            dropout=self.dropout,
        )

        self.decoder_gan = GateAddNorm(
            n_inputs=self.hidden_dim,
            dropout=self.dropout,
        )

        self.decoder_grn = GatedResidualNetwork(
            n_inputs=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_outputs=self.hidden_dim,
            dropout=self.dropout,
        )

        if not self.task_name == 'vl_regression':
            self.output_layers = []
            self.output_layers.append(Linear(in_features=self.hidden_dim * input_dim[-2],
                                            out_features=self.model_out_dim,
                                            bias=False))
            if output_activation is not None:
                self.output_layers.append(get_output_activation(output_activation))
            self.output_layers = Sequential(*self.output_layers)
        else:
            self.output_layers = nn.Linear(self.hidden_dim, self.model_out_dim)

    def forward(self, x: Tensor, *internal_states) -> Tensor:
        """
        Forward pass of the TFT model.

        The computation pipeline consists of:

        1. Feature-wise embedding
        2. Variable selection
        3. LSTM encoding
        4. Residual gating
        5. Interpretable multi-head attention
        6. Decoder gated residual refinement
        7. Task-specific output projection

        If multiple ``internal_states`` are provided, they overwrite the
        internal states of the stateful LSTM before processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, n_features).
        *internal_states : tuple of torch.Tensor, optional
            Optional hidden and cell states used to overwrite the internal
            LSTM states before forward propagation. Only used when the
            encoder is stateful.

        Returns
        -------
        torch.Tensor
            Model output tensor.

            - For ``'regression'``:
              Shape (batch_size, output_dim[0], output_dim[1])
            - For ``'vl_regression'``:
              Shape (batch_size, time_steps, model_out_dim)
            - For ``'classification'``:
              Shape (batch_size, model_out_dim)

        Raises
        ------
        SystemExit
            If an unsupported ``task_name`` is provided.
        """

        if len(internal_states) > 1:
            self.lstm_encoder._overwrite_internal_states(internal_states)

        # EmbeddingLayer
        x_embedded = self.embedder(x)

        # VariableSelectionNetwork
        embeddings_varying_encoder, _ = self.vsn(x=x_embedded)

        # LSTM
        encoder_out = self.lstm_encoder(embeddings_varying_encoder)

        # First GateAddNorm
        lstm_out = self.lstm_out_gan(encoder_out, residual_connect=embeddings_varying_encoder)

        # Attention
        attn_out, _ = self.attention(lstm_out)

        # Second GateAddNorm
        attn_out = self.decoder_gan(x=attn_out, residual_connect=lstm_out)

        # GatedResidualNetwork
        grn_decoder_out = self.decoder_grn(x=attn_out)

        # Dense
        if not self.task_name == 'vl_regression':
            grn_decoder_out = grn_decoder_out.reshape(grn_decoder_out.shape[0], grn_decoder_out.shape[1] * grn_decoder_out.shape[2])
        out = self.output_layers(grn_decoder_out)

        if self.task_name in ('regression', 'vl_regression'):
            if not self.task_name == 'vl_regression':
                out = out.view(grn_decoder_out.shape[0], self.final_out_shape[0], self.final_out_shape[1])
        elif self.task_name == 'classification':
            pass
        else:
            logger.error('Unknown task name %s, for LSTM.', self.task_name)
            exit(101)

        return out

    def force_reset(self) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm_encoder.force_reset()

    def get_internal_states(self) -> list:
        """ Wrapper for the underlying LSTM's corresponding function."""
        return self.lstm_encoder.get_internal_states()

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm_encoder.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm_encoder.update_states(ist_idx, device)

class Gate(Module):
    """
    A gating mechanism using Gated Linear Unit (GLU) activations with optional dropout.

    This module applies a linear transformation to the input, doubling the output
    features to produce gates and values, then uses the GLU activation function to
    gate the output. Dropout is applied before the linear layer if specified.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_outputs : int, optional (default=None)
        Number of output features. If None, them the number of outputs will be the number of inputs.
    dropout : float, optional (default=None)
        Dropout probability applied before the linear transformation. If None, no dropout is applied.

    Attributes
    ----------
    fc : Linear
        Linear layer producing gates and values with output dimension `n_outputs * 2`.
    dropout : Dropout or None
        Dropout layer applied before linear transformation.

    Methods
    -------
    forward(x)
        Applies dropout (if any), linear transformation, and GLU gating to the input tensor.
    """

    dropout = None

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int = None,
        dropout: float = None,
    ) ->None:
        super().__init__()

        if dropout is not None:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = dropout

        n_outputs = n_outputs or n_inputs
        self.fc = Linear(n_inputs, n_outputs * 2)

        init_mod(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Gating mechanism.

        Parameters
        ----------
        x : Tensor of shape (batch_size, sequence_length, n_input)
            Input tensor.

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, depth)
            Output tensor after gating, where `depth` is half the output size of the linear layer.
        """

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)
        out = glu(x, dim=-1)

        return out


class GateAddNorm(Module):
    """
    Gates the input (using *Gate*), implements a residual connection, and applies layer normalization.

    This module applies a *Gate* transformation to the input tensor,
    adds a residual connection (projected if necessary to match dimensions), and
    normalizes the result using LayerNorm.

    Parameters
    ----------
    n_inputs : int
        Number of input features (last dimension of `x`).
    n_outputs : int, optional
        Number of output features. Defaults to `n_input`.
    n_residuals : int, optional
        Dimensionality of the residual input. If different from `n_outputs`, a linear projection
        is applied to align dimensions. Defaults to `n_outputs`.
    dropout : float, optional
        Dropout rate applied inside the GLU. If None, no dropout is applied.

    Attributes
    ----------
    gate : Gate
        The Gate transformation module.
    norm : LayerNorm
        Layer normalization applied after residual addition.

    Methods
    -------
    forward(x, residual_connect)
        Applies *Gate*, adds residual connection (with projection if needed), and normalizes output.

    """

    n_outputs = None
    n_residuals = None

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int = None,
        n_residuals: int = None,
        dropout: float = None,
    ) -> None:
        super().__init__()

        self.n_outputs = n_outputs or n_inputs
        self.n_residuals = n_residuals or self.n_outputs

        self.gate = Gate(
            n_inputs=n_inputs,
            n_outputs=self.n_outputs,
            dropout=dropout
        )

        self.projection = None
        if self.n_outputs != self.n_residuals:
            self.projection = Linear(n_residuals, n_outputs)

        self.norm = LayerNorm(self.n_outputs)

    def forward(self, x, residual_connect) -> Tensor:
        """
        Forward pass applying *Gate* transformation, residual addition, and normalization.

        Parameters
        ----------
        x : Tensor of shape (batch_size, sequence_length, n_input)
            Input tensor to transform.
        residual_connect : Tensor of shape (batch_size, sequence_length, n_residuals)
            Residual tensor to add after transformation. Projected if necessary.

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, n_outputs)
            The output tensor after gated transformation, residual addition, and layer normalization.
        """
        x = self.gate(x)
        if residual_connect is not None:
            if self.projection is not None:
                residual_connect = self.projection(residual_connect)
            x = self.norm(x + residual_connect)
        else:
            x = self.norm(x)

        return x


class GatedResidualNetwork(Module):
    """
    This module implements a Gated Residual Network (GRN) block that consists of
    two linear layers with ELU activations inbetween, followed by a *GateAddNorm* module.
    There is a residual connection to the *GateAddNorm* module.

    Parameters
    ----------
    n_inputs : int
        Number of input features (last dimension of `x`).
    hidden_dim : int
        Hidden layer dimensionality within the GRN block.
    n_outputs : int, optional
        Number of output features. Defaults to `n_input`.
    dropout : float, optional (default=None)
        Dropout rate applied in the gated residual normalization block.

    Methods
    -------
    forward(x, residual_connect=None)
        Forward pass through the GRN, applying linear transformations, activation,
        gating, residual connection, and normalization.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_dim: int,
        n_outputs: int,
        dropout: float = None,
    ) -> None:
        super().__init__()

        self.fc1 = Linear(n_inputs, hidden_dim)
        self.elu = ELU()
        # Note this second layer is probably redundant since the gating mechanisms also
        # starts with a linear layer, but keeping in to be consistent with source material.
        self.fc2 = Linear(hidden_dim, hidden_dim)
        init_mod(self)

        self.gate_norm = GateAddNorm(
            n_inputs = hidden_dim,
            n_residuals = n_inputs,
            n_outputs = n_outputs,
            dropout=dropout,
        )

    def forward(self, x) -> Tensor:
        """
        Forward pass through the Gated Residual Network.

        Parameters
        ----------
        x : Tensor of shape (batch_size, sequence_length, n_input)
            Input tensor.

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, n_output)
            Output tensor after processing through GRN block with gated residual connection and normalization.
        """

        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        out = self.gate_norm(h, x)

        return out


class EmbeddingLayer(Module):
    """
    Embedding layer supporting independent or fully mixed feature embedding.

    This module embeds a multivariate time series input of shape
    (batch_size, time_steps, input_components) into a structured
    representation of shape

        (batch_size, time_steps, input_components, hidden_dim)

    Two embedding strategies are supported:

    1. "independent"
        Each input component is embedded separately using its own
        Linear(1 → hidden_dim) layer. No cross-feature interaction occurs
        during embedding.

    2. "mixed"
        All input components at each time step are jointly projected using
        a single Linear(n_inputs → n_inputs * hidden_dim) layer. The output
        is then reshaped to recover a per-feature embedding structure.

        In this mode, each feature embedding depends on the full input
        feature vector at that time step.

    Parameters
    ----------
    n_inputs : int
        Number of input components (features) in the last dimension of
        the input tensor.
    hidden_dim : int
        Dimensionality of the embedding space for each input component.
    mode : str, default="independent"
        Embedding strategy. Must be either "independent" or "mixed".

    Attributes
    ----------
    embedders : nn.ModuleList
        Present when mode="independent". Contains n_inputs separate
        Linear(1, hidden_dim) layers.
    embedder : nn.Linear
        Present when mode="mixed". A single Linear layer mapping
        n_inputs → n_inputs * hidden_dim.

    Notes
    -----
    Input tensor shape:
        (batch_size, time_steps, input_components)

    Output tensor shape (both modes):
        (batch_size, time_steps, input_components, hidden_dim)

    In "mixed" mode, the parameter count scales approximately as:

        n_inputs × (n_inputs × hidden_dim)

    which grows quadratically with the number of input components.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_dim: int,
        mode: str = "independent",
    ) -> None:
        super().__init__()

        self.mode = mode
        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim

        if mode == "independent":
            self.embedders = nn.ModuleList(
                [nn.Linear(1, hidden_dim) for _ in range(n_inputs)]
            )

        elif mode == "mixed":
            self.embedder = nn.Linear(
                n_inputs,
                n_inputs * hidden_dim
            )

        else:
            logger.error("mode must be 'independent' or 'mixed'")
            exit(101)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the embedding transformation.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            (batch_size, time_steps, input_components).

        Returns
        -------
        Tensor
            Embedded tensor of shape
            (batch_size, time_steps, input_components, hidden_dim).

            - In "independent" mode, each feature is embedded separately.
            - In "mixed" mode, each feature embedding depends on the
              entire feature vector at the corresponding time step.
        """

        if self.mode == "independent":
            x_embedded = []
            for j, layer in enumerate(self.embedders):
                x_j = x[:, :, j:j + 1]
                x_j_emb = layer(x_j)
                x_embedded.append(x_j_emb)

            return stack(x_embedded, dim=2)

        else:
            B, T, _ = x.shape

            x_proj = self.embedder(x)
            x_proj = x_proj.view(
                B, T,
                self.n_inputs,
                self.hidden_dim
            )

            return x_proj


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).

    Implements a component-wise variable selection mechanism.
    Each input component is processed by a component-specific nonlinear transformation.
    A joint gating network computes importance weights across features, which
    are used to produce a weighted aggregation of feature representations.

    Parameters
    ----------
    n_inputs : int
        Number of input components.
    hidden_dim : int
        Dimensionality of the shared hidden representation space.
    dropout : float, optional
        Dropout rate applied within internal GatedResidualNetwork modules.

    Notes
    -----
    - Cross-feature interaction occurs within the variable selection
      network.
    - Selection weights are normalized using a softmax over the feature
      dimension.
    - The output is a convex combination of processed feature representations.
    """


    def __init__(
        self,
        n_inputs: int,
        hidden_dim: int,
        dropout: float = None,
    ) -> None:
        super().__init__()

        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim

        self.variable_selector = GatedResidualNetwork(n_inputs=n_inputs * hidden_dim,
                                                      hidden_dim=n_inputs * hidden_dim, # <- Chosen heuristically, can be an HP later
                                                      n_outputs=n_inputs,
                                                      dropout=dropout)

        self.softmax = Softmax(dim=-1)

        self.non_linear_procs = ModuleList()
        for idx in range(n_inputs):
            self.non_linear_procs.append(GatedResidualNetwork(
                n_inputs=hidden_dim,
                hidden_dim=hidden_dim,
                n_outputs=hidden_dim,
                dropout=dropout,
            ))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Variable Selection Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, F, H), where
            B is the batch size,
            T is the sequence length,
            F is the number of input features.
            H is the number of hidden dimensions.

        Returns
        -------
        torch.Tensor
            Aggregated feature representation of shape
            (B, T, hidden_dim).
        output : Tensor
            Aggregated feature representation of shape
            (B, T, hidden_dim).
        variable_selection : Tensor
            Tensor of shape (B, T, n_inputs) containing the
            soft attention weights assigned to each input component.
            Meant for interpretability.

        Notes
        -----
        The forward computation consists of:

        1. Independent per-feature projection into the shared
           representation space.
        2. Computation of context-dependent variable selection weights.
        3. Feature-wise nonlinear processing.
        4. Softmax-weighted aggregation across the feature dimension.
        """

        B, T, F, H = x.shape

        # calculate the variable selection weights
        x_flattened = x.reshape(B, T, F * self.hidden_dim)
        variable_selection = self.variable_selector(x_flattened)
        variable_selection = self.softmax(variable_selection)

        # process the inputs
        procd_outputs = []
        for j, layer in enumerate(self.non_linear_procs):
            x_j = x[:, :, j, :]
            x_j_out = layer(x_j)
            procd_outputs.append(x_j_out)
        procd_outputs = stack(procd_outputs, dim=2)

        # weight the processed inputs by the variable selection weights
        variable_selection = variable_selection.unsqueeze(-1)
        weighted = procd_outputs * variable_selection
        output = weighted.sum(dim=2)

        return output, variable_selection


class ScaledDotProductAttention(Module):
    """
    Scaled dot-product attention mechanism.

    This module computes attention scores using the dot product between queries and keys,
    optionally scales the scores, optionally applies causal masking,
    and applies a softmax function to obtain attention weights.
    These weights are used to aggregate values.
    Optionally includes dropout after softmax.

    Parameters
    ----------
    attention_dropout : float, optional
        Dropout probability to apply after the attention softmax. If None, no dropout is used.
    scale : bool, optional
        If True, scales attention scores by the square root of the key dimension. Default is True.
    masking : bool, optional
        If True, attention values are masked to prevent temporal dimensions from attending to future values.
        Default True.

    Attributes
    ----------
    attention_dropout : Dropout or None
        Dropout layer applied to the attention weights, if specified.
    scale : bool
        Whether to apply scaling to the attention logits.
    masking : bool
        If True, attention values are masked to prevent temporal dimensions from attending to future values.

    Methods
    -------
    forward(q, k, v)
        Computes attention-weighted values and attention weights from input tensors.
    """

    scale = None

    def __init__(
        self,
        attention_dropout: float = None,
        scale: bool = True,
        masking: bool = True,
    ) -> None:
        super().__init__()

        self.attention_dropout = attention_dropout
        if self.attention_dropout is not None:
            self.attention_dropout = Dropout(p=attention_dropout)

        self.scale = scale
        self.masking = masking

    def forward(self, q, k, v) -> Tensor:
        """
        Forward pass of the scaled dot-product attention mechanism.

        Computes attention scores as the dot product between queries and keys,
        optionally scales them, optionally masks for causality, then normalizes
        with softmax. The resulting attention weights are used to compute
        a weighted sum over the values.

        Parameters
        ----------
        q : Tensor, shape [batch_size, sequence_length, embedding_dim]
            Query tensor.
        k : Tensor, shape [batch_size, sequence_length, embedding_dim]
            Key tensor.
        v : Tensor, shape [batch_size, sequence_length, embedding_dim]
            Value tensor.

        Returns
        -------
        out : Tensor, shape [batch_size, sequence_length, embedding_dim]
            Resulting attention-weighted sum of values.
        attn : Tensor, shape [batch_size, sequence_length, sequence_length]
            Attention weights after softmax.
        """

        attn = bmm(q, k.permute(0, 2, 1))
        if self.scale:
            attn = attn / (k.size(-1) ** 0.5)

        if self.masking:
            # TODO: This mask is generated every forward pass. This is inefficient, but currently required for
            #  variable length input processing. To think if we can get around it.
            mask = tril(ones(attn.size(-2), attn.size(-1), device=attn.device, dtype=bool)).unsqueeze(0)
            attn = attn.masked_fill(~mask, finfo(attn.dtype).min)

        attn = softmax(attn, dim=-1)

        if self.attention_dropout is not None:
            attn = self.attention_dropout(attn)

        out = bmm(attn, v)

        return out, attn


class InterpretableMultiHeadAttention(Module):
    """
    Implements an interpretable multi-head attention mechanism where each attention
    head shares the same value vector but uses separate query and key projections.

    This variant simplifies interpretability by decoupling the query/key learning across
    heads while enforcing a shared value representation. Useful in architectures where
    clarity of attention allocation is important (e.g., temporal attention).

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    hidden_dim : int
        Total feature dimension of the input and output.
    dropout : float, optional
        Dropout probability applied after attention and after output projection. Default is None.
    masking : bool, optional
        If True, attention values are masked to prevent temporal dimensions from attending to future values.
        Default True.

    Attributes
    ----------
    v_layer : nn.Linear
        Shared linear projection for values across all heads.
    q_layer : nn.ModuleList
        Per-head linear projections for queries.
    k_layer : nn.ModuleList
        Per-head linear projections for keys.
    attention : ScaledDotProductAttention
        Core attention mechanism computing attention weights and outputs.
    w_h : nn.Linear
        Final projection layer mapping averaged head output back to model dimension.
    dropout : nn.Dropout
        Dropout probability applied after attention and after output projection.
    """
    def __init__(
        self,
        n_heads: int,
        hidden_dim: int,
        dropout: float = None,
        masking: bool = True,
    ) -> None:
        super().__init__()

        assert hidden_dim % n_heads == 0, \
            "hidden_dim must be divisible by n_heads in TFT."

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.d_k = self.d_q = self.d_v = hidden_dim // n_heads

        self.dropout = dropout
        if self.dropout is not None:
            self.dropout = Dropout(p=dropout)

        self.v_layer = Linear(self.hidden_dim, self.d_v)
        self.q_layer = ModuleList([Linear(self.hidden_dim, self.d_q) for _ in range(n_heads)])
        self.k_layer = ModuleList([Linear(self.hidden_dim, self.d_k) for _ in range(n_heads)])
        self.attention = ScaledDotProductAttention(attention_dropout=dropout, scale=True, masking=masking)
        self.w_h = Linear(self.d_v, self.hidden_dim, bias=False)

        init_mod(self)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Interpretable Multi-Head Attention module.

        Each attention head uses a separate query and key projection but shares the same
        value projection across heads. Attention is computed per head, optionally masked
        for causality, and optionally uses dropout. The outputs of all heads are averaged
        (not concatenated) to produce the final attention output, which is then projected
        back to the model dimension and passed through a final dropout.

        Parameters
        ----------
        x : Tensor, shape [batch_size, sequence_length, hidden_dim]
            Input tensor containing the sequence to attend over. Each element in the sequence
            is a hidden vector of dimension `hidden_dim`.

        Returns
        -------
        out : Tensor, shape [batch_size, sequence_length, hidden_dim]
            The attention-weighted output sequence after averaging across heads and applying
            the final linear projection.

        attn : Tensor, shape [batch_size, sequence_length, sequence_length, n_heads]
            Attention weights for each head. Each slice `attn[:, :, :, i]` corresponds
            to the attention map of head `i`, after softmax and optional dropout.
            If `n_heads == 1`, the head dimension is omitted.

        Notes
        -----
        - Causal masking ensures that each position can only attend to previous and current
          time steps when `masking=True`.
        - The value projection is shared across all heads to enforce interpretability,
          following the Temporal Fusion Transformer design.
        """

        heads = []
        attns = []

        # global value
        v = self.v_layer(x)
        for i in range(self.n_heads):
            # head-specific query
            q = self.q_layer[i](x)
            # head-specific key
            k = self.k_layer[i](x)
            # calculate head attention
            head, attn = self.attention(q, k, v)
            # store head output
            heads.append(head)
            # store head attention values
            attns.append(attn)

        # concatenate per-head variables
        head = stack(heads, dim=2) if self.n_heads > 1 else heads[0]
        attn = stack(attns, dim=2)

        # average across heads
        out = mean(head, dim=2) if self.n_heads > 1 else head
        # apply final projection
        out = self.w_h(out)
        # apply dropout after final projection
        if self.dropout is not None:
            out = self.dropout(out)

        return out, attn


def init_mod(mod):
    """
    Initializes the parameters of the given module using suitable initialization schemes.

    This function iterates over the named parameters of the provided module and applies:
    - Kaiming uniform initialization for parameters containing 'weight' in their name, if applicable.
    - Standard normal initialization for 'weight' parameters where Kaiming initialization is unsuitable.
    - Zero initialization for parameters containing 'bias' in their name.

    Parameters
    ----------
    mod : nn.Module
        The PyTorch module whose parameters will be initialized.

    Notes
    -----
    This function is used to prepare layers for training by setting their initial weights and biases
    to suitable values, which can improve convergence rates.
    """

    for name, parameters in mod.named_parameters():
        if 'weight' in name:
            try:
                kaiming_uniform_(parameters)
            except:
                normal_(parameters)
        elif 'bias' in name:
            zeros_(parameters)


def get_output_activation(output_activation: None | str) -> Module:
    """Fetch output activation function from Pytorch."""
    if output_activation == "Softmax":
        return getattr(nn, output_activation)(dim=1)
    if output_activation == "Sigmoid":
        return getattr(nn, output_activation)()
    return Identity()