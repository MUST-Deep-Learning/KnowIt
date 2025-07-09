"""
This module implements a simplified Temporal Fusion Transformer (TFT) architecture for
sequence-to-sequence modeling with only temporal inputs.

Unlike the original TFT, this version omits all static input processing and static context encoding.
It is tailored for forecasting, regression, or classification tasks over temporal sequences,
using temporal variable selection, LSTM-based encoding/decoding, and temporal self-attention.

See:
[1] "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"  
    Lim et al., 2019 — https://arxiv.org/abs/1912.09363

------------------------
VariableSelectionNetwork
------------------------

Learns per-timestep soft attention over temporal input variables.
    - Each variable passes through a shared GatedResidualNetwork.
    - Importance weights are computed using softmax across input variables.
    - Used separately for encoder and decoder inputs.

--------------------
GatedResidualNetwork
--------------------

A gated feedforward block:
    [Linear] → [ELU] → [Linear] → [Gating + Residual + Norm]

    - Enables non-linear transformations with gating control.
    - Residual connection allows gradient flow and feature reuse.
    - LayerNorm stabilizes outputs.

-----------
GateAddNorm
-----------

Applies gating, residual connection, and normalization:
    Output = LayerNorm(residual + gate * candidate)

Used after GRNs and attention to improve stability.

----------
TFTCore
----------

The main temporal modeling stage:

    - LSTM Encoder:
        Processes full input sequence.

    - LSTM Decoder:
        Processes final `decoder_time` steps from input sequence.

    - Temporal Self-Attention:
        Applies attention over decoder steps to capture long-range dependencies.

    - Position-wise GRN:
        Final per-timestep transformation before output.

----------
FinalBlock
----------

After the TFTCore we obtain a tensor T(batch_size, decoder_steps, model_dim).

If task_name = 'regression':
    - Linear projection → (batch_size, decoder_steps, target_dim)
    - Output activation (optional)

If task_name = 'classification':
    - Linear projection → (batch_size, decoder_steps, num_classes)

If task_name = 'forecast':
    - Sequence output over decoder_time steps.

Notes
-----
    - TFT handles **only temporal inputs**, and is suitable for forecasting, regression, and classification.
    - The architecture **does not use static encodings**.
    - All weights (excluding biases) are initialized with `nn.init.kaiming_uniform_` if dimensions allow,
      otherwise with `nn.init.normal_`.
    - All bias parameters are initialized to zero.
    - `decoder_time_steps` controls the number of time steps passed to the decoder.
      It must be > 0 and constitutes the final segment of the input sequence
      [batch_size, -decoder_time_steps:, num_features].
"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "potgieterharmen@gmail.com"
__description__ = "Example of an TFT model architecture."

from torch import (Tensor, zeros, bool, arange, cat, float, stack, ones, as_tensor, bmm, mean, nn)
from torch.nn import (Module, LSTM, Parameter, Sigmoid, Linear, ModuleList, LayerNorm, ELU, Dropout, Softmax)
from torch.nn.functional import glu
from torch.nn.init import (zeros_, kaiming_uniform_, normal_)
from typing import Dict, Tuple, List, Optional
from numpy import arange

from helpers.logger import get_logger
logger = get_logger()

available_tasks = ("regression", "classification", "forcast")

HP_ranges_dict = {"depth": range(1, 512, 1),
                  "lstm_depth": range(1, 64, 1),
                  "num_attention_heads": range(1, 4, 1),
                  "dropout": arange(0, 1.1, 0.1),
                  "full_attention": (True, False),
                  "output_activation": (None, "Sigmoid", "Softmax"),
                  "decoder_time_steps": range(1, 1025, 1),
                  "stateful": (False, True),
}

class Model(Module):
    """
    A hybrid temporal model combining variable selection, LSTM encoding/decoding,
    gated residual processing, and interpretable attention. Supports regression,
    classification, and (WIP) forecasting tasks.

    Architecture Overview
    ---------------------
    The model uses a standard encoder-decoder pattern with LSTMs as the backbone,
    augmented by learned variable selection, gating mechanisms, and a multi-head
    attention block. It optionally includes full or causal attention masking.

    Block Summary:
        - VariableSelectionNetwork: Selects and embeds input features at each time step.
        - LSTM Encoder: Processes historical sequence data.
        - LSTM Decoder: Processes future steps or zero-padded inputs during inference.
        - GateAddNorm: Combines LSTM output and embeddings.
        - GatedResidualNetwork: Injects static enrichment before attention.
        - Multi-Head Attention: Produces interpretable attention-weighted summaries.
        - FinalBlock: Maps the decoded output to task-specific predictions.

    Parameters
    ----------
    input_dim : list[int]
        Shape of the input sequence as [num_time_steps, num_features].

    output_dim : list[int]
        Shape of the output sequence as [num_time_steps, num_features].

    task_name : str
        One of {"regression", "classification", "forecast"}. Defines the output format.

    depth : int, default=16
        Dimensionality of intermediate representations.

    lstm_depth : int, default=4
        Number of stacked LSTM layers for both encoder and decoder.

    num_attention_heads : int, default=4
        Number of attention heads in the multi-head attention layer.

    dropout : float or None, default=0.1
        Dropout probability applied throughout the model.

    full_attention : bool, default=True
        If True, decoder attention can access all decoder positions.
        If False, attention is limited to causal (past) steps only.

    output_activation : str or None, default=None
        Optional activation to apply to the model's output.

    decoder_time_steps : int, default=10
        Number of future time steps predicted (used to split encoder/decoder).

    stateful : bool, default=False
        Enables persistent LSTM hidden state across batches (not fully implemented).

    Notes
    -----
    - In inference mode, the decoder receives zero-filled inputs.
    - Attention masking is recomputed dynamically if batch size changes.
    - Hidden states are reset unless `stateful=True`, but persistent state
      handling is not robust.
    - Forecasting mode is marked WIP and may produce unstable outputs.
    - All submodules assume consistent input dimensionality (`depth`).
    - The final output shape depends on `task_name`:
        - "regression": (B, T_out, C_out)
        - "classification": (B, C_out)
        - "forecast" (WIP): (B, T_out, C_out)

    """

    task_name = None
    full_attention = True
    output_activation = None

    def __init__(self,
                 input_dim: list,
                 output_dim: list,
                 task_name: str,
                 *,
                 depth: int = 16,
                 lstm_depth: int = 4,
                 num_attention_heads: int = 4,
                 dropout: float | None = 0.1,
                 full_attention: bool = True,
                 output_activation: str | None = None,
                 decoder_time_steps: int = 10,
                 stateful: bool = False,
    ) -> None:

        super().__init__()

        self.num_model_out_time_steps = output_dim[0]
        self.num_model_out_channels = output_dim[1]
        self.num_model_in_time_steps = input_dim[0]
        self.num_model_in_channels = input_dim[1]
        self.decoder_time_steps = decoder_time_steps
        self.encoder_time_steps = input_dim[0] - decoder_time_steps

        self.task_name = task_name
        self.depth = depth
        self.lstm_depth = lstm_depth
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.full_attention = full_attention
        self.state_full = stateful

        self.output_activation = output_activation

        self.batch_size_last = -1
        self.attention_mask = None
        self.inference = False

        self._attn_out_weights = None
        self._static_covariate_var = None
        self._encoder_sparse_weights = None
        self._decoder_sparse_weights = None

        #Encoder
        self.lstm_encoder_vsn = _VariableSelectionNetwork(
            n_input_dim=1,
            n_input_features=self.num_model_in_channels,
            depth=self.depth,
            dropout=self.dropout,
        )

        self.lstm_decoder_vsn = _VariableSelectionNetwork(
            n_input_dim=1,
            n_input_features=self.num_model_in_channels,
            depth=self.depth,
            dropout=self.dropout,
        )

        self.lstm_encoder = LSTM(
            input_size=self.depth,
            hidden_size=self.depth,
            num_layers=self.lstm_depth,
            dropout=self.dropout if self.lstm_depth > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.depth,
            hidden_size=self.depth,
            num_layers=self.lstm_depth,
            dropout=self.dropout if self.lstm_depth > 1 else 0,
            batch_first=True,
        )

        self.lstm_out_gan = _GateAddNorm(
            n_input=self.depth,
            dropout=self.dropout,
        )

        # Decoder
        self.static_enrichment_grn = _GatedResidualNetwork(
            n_input=self.depth,
            depth=self.depth,
            n_output=self.depth,
            dropout=self.dropout,
        )
        self.attention = _InterpretableMultiHeadAttention(
            d_model=self.depth,
            n_head=self.num_attention_heads,
            dropout=self.dropout,
        )
        self.decoder_gan = _GateAddNorm(
            n_input=self.depth,
            dropout=self.dropout,
        )
        self.decoder_grn = _GatedResidualNetwork(
            n_input=self.depth,
            depth=self.depth,
            n_output=self.depth,
            dropout=self.dropout,
        )

        #Output layer
        self.pre_out_gan = _GateAddNorm(
            n_input=self.depth,
            dropout=None,
        )

        self.final_output_layer = FinalBlock(
            num_model_in_time_steps=self.decoder_time_steps,
            num_model_out_channels=self.num_model_out_channels,
            num_model_out_time_steps=self.num_model_out_time_steps,
            output_activation=self.output_activation,
            depth=self.depth,
            task=self.task_name
        )

    @staticmethod
    def get_attention_mask_future(
            encoder_length: int,
            decoder_length: int,
            batch_size: int,
            device: str,
            full_attention: bool
    ) -> Tensor:
        """
        Constructs a boolean attention mask for encoder-decoder attention in sequence models.

        Parameters
        ----------
        encoder_length : int
            Number of time steps in the encoder sequence.

        decoder_length : int
            Number of time steps in the decoder sequence (forecast horizon).

        batch_size : int
            Batch size, used to expand mask across the batch dimension.

        device : str
            Device identifier (e.g., 'cpu' or 'cuda') for mask tensor allocation.

        full_attention : bool
            If True, allow decoder to attend to all decoder time steps (no causal masking).
            If False, enforce causal masking so each decoder step only attends to
            past and present steps (no peeking into the future).

        Returns
        -------
        Tensor
            A boolean tensor mask of shape (batch_size, decoder_length, encoder_length + decoder_length),
            where True indicates positions **masked out** (ignored) by the attention mechanism.

        Notes
        -----
        - Encoder positions are never masked (fully visible).
        - Decoder mask depends on `full_attention`:
          - True: no mask (all False).
          - False: mask out future decoder steps to enforce causality.
        - The mask concatenates encoder and decoder masks along the last dimension
          to produce a combined attention mask for cross-attention.
        """

        if full_attention:
            decoder_mask = zeros((decoder_length, decoder_length), dtype=bool, device=device)
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step

        encoder_mask = zeros(batch_size, encoder_length, dtype=bool, device=device)
        # combine masks along attended time - first encoder and then decoder
        mask = cat((
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1)
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Tensor) -> Tensor: #, reset_lstm_state: bool = True) -> Tensor:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, total_time_steps, num_input_channels).
            The sequence includes both encoder (past) and decoder (future) time steps.

        Returns
        -------
        Tensor
            Output tensor shaped according to the task, typically
            (batch_size, decoder_time_steps, num_output_channels).

        Details
        -------
        - Splits input into encoder and decoder sequences.
        - Processes both sequences through variable selection networks and LSTMs.
        - Applies gating and residual connections.
        - Uses multi-head attention on decoder output with masking.
        - Produces final output via output layers, applying activation if specified.
        - Handles stateful LSTM hidden states internally.
        """

        reset_lstm_state = True
        batch_size = x.shape[0]

        input_vectors_past = x[:, :self.encoder_time_steps, :]
        if self.inference:
            input_vectors_future = zeros((batch_size, self.decoder_time_steps, self.num_model_in_channels),
                                         device=x.device)
        else:
            input_vectors_future = x[:, self.encoder_time_steps:, :]

        if batch_size != self.batch_size_last:
            self.attention_mask = self.get_attention_mask_future(
                self.encoder_time_steps,
                self.decoder_time_steps,
                batch_size,
                x.device,
                self.full_attention
            )
            self.batch_size_last = batch_size


        embeddings_varying_encoder, encoder_sparse_weights = self.lstm_encoder_vsn(x=input_vectors_past)
        embeddings_varying_decoder, decoder_sparse_weights = self.lstm_decoder_vsn(x=input_vectors_future)

        #LSTM
        if reset_lstm_state or self.hidden_state is None:
            h0 = zeros(self.lstm_depth, batch_size, self.depth, device=x.device)
            c0 = zeros(self.lstm_depth, batch_size, self.depth, device=x.device)
        else:
            h0, c0 = self.hidden_state, self.cell_state

        encoder_out, (hn, cn) = self.lstm_encoder(input=embeddings_varying_encoder, hx=(h0, c0))

        decoder_out, _ = self.lstm_decoder(input=embeddings_varying_decoder, hx=(hn, cn))

        lstm_layer = cat([encoder_out, decoder_out], dim=1)
        input_embeddings = cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)

        self.hidden_state = hn.detach()
        self.cell_state = cn.detach()

        lstm_out = self.lstm_out_gan(lstm_layer, input_embeddings)

        # Attention head
        attn_in = self.static_enrichment_grn(x=lstm_out)

        attn_out, att_out_weights = self.attention(
            q=attn_in[:, self.encoder_time_steps:],
            k=attn_in,
            v=attn_in,
            mask=self.attention_mask,
        )

        attn_out = self.decoder_gan(x=attn_out, residual_connect=attn_in[:, self.encoder_time_steps:])
        decoder_out = self.decoder_grn(x=attn_out)

        #Output
        pre_out_gan = self.pre_out_gan(decoder_out, lstm_out[:, self.encoder_time_steps:])

        final_out = self.final_output_layer(pre_out_gan)

        self._attn_out_weights = att_out_weights
        self._encoder_out_weights = encoder_sparse_weights
        self._decoder_out_weights = decoder_sparse_weights

        return final_out

################################# Sub-Modules #################################################

class _GateAddNorm(Module):
    """
    Gated residual block with optional projection and layer normalization.

    This module applies a Gated Linear Unit (GLU) transformation to the input tensor,
    adds a residual connection (projected if necessary to match dimensions), and
    normalizes the result using LayerNorm.

    Parameters
    ----------
    n_input : int
        Number of input features (last dimension of `x`).
    depth : int, optional
        Output feature dimension after GLU transformation. Defaults to `n_input`.
    residual : int, optional
        Dimensionality of the residual input. If different from `depth`, a linear projection
        is applied to align dimensions. Defaults to `depth`.
    dropout : float, optional
        Dropout rate applied inside the GLU. If None, no dropout is applied.

    Attributes
    ----------
    glu : _GatedLinearUnit
        The GLU transformation module.
    norm : LayerNorm
        Layer normalization applied after residual addition.

    Methods
    -------
    forward(x, residual_connect)
        Applies GLU, adds residual connection (with projection if needed), and normalizes output.

    """


    n_input = None
    depth = None
    residual = None
    dropout = None

    def __init__(
        self,
        n_input: int,
        depth: int = None,
        residual: int = None,
        dropout: float = None,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.depth = depth or n_input
        self.residual = residual or self.depth
        self.dropout = dropout

        self.glu = _GatedLinearUnit(self.n_input, depth=self.depth, dropout=self.dropout)
        self.norm = LayerNorm(self.depth)

    def forward(self, x, residual_connect) -> Tensor:
        """
        Forward pass applying gated transformation, residual addition, and normalization.

        Parameters
        ----------
        x : Tensor of shape (batch_size, sequence_length, n_input)
            Input tensor to transform.
        residual_connect : Tensor of shape (batch_size, sequence_length, residual)
            Residual tensor to add after transformation. Projected if necessary.

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, depth)
            The output tensor after gated transformation, residual addition, and layer normalization.
        """
        x = self.glu(x)
        if self.depth != self.residual:
            residual_connect = self.resample(residual_connect)

        x = self.norm(x + residual_connect)
        return x


class _GatedResidualNetwork(Module):
    """
    Gated Residual Network (GRN) module with optional residual projection and normalization.

    This module implements a GRN block that consists of two linear layers with ELU activation,
    followed by a gated residual connection with layer normalization. It supports projecting
    the residual input to match the output dimension if needed.

    Parameters
    ----------
    n_input : int
        Number of input features.
    depth : int
        Hidden layer dimensionality within the GRN block.
    n_output : int
        Number of output features. Residual input is projected if this differs from `n_input`.
    dropout : float, optional (default=0.1)
        Dropout rate applied in the gated residual normalization block.
    residual_connect : bool, optional (default=True)
        Whether to use residual connections.

    Attributes
    ----------
    linear : Linear, optional
        Linear layer projecting residual input if `n_input` != `n_output`.
    norm : LayerNorm
        Layer normalization applied to the residual connection.
    fc1 : Linear
        First linear layer in the GRN block.
    elu : ELU
        ELU activation function.
    fc2 : Linear
        Second linear layer in the GRN block.
    gate_norm : _GateAddNorm
        Gated residual block that applies gating, residual addition, and normalization.

    Methods
    -------
    forward(x, residual_connect=None)
        Forward pass through the GRN, applying linear transformations, activation,
        gating, residual connection, and normalization.
    """

    def __init__(
        self,
        n_input: int,
        depth: int,
        n_output: int,
        dropout: float = 0.1,
        residual_connect: bool = True,
    ) -> None:
        super().__init__()

        self.n_input = n_input
        self.depth = depth
        self.n_output = n_output
        self.dropout = dropout
        self.residual_connect = residual_connect

        if self.n_output != self.n_input:
            self.linear = Linear(self.n_input, self.n_output)

        self.norm = LayerNorm(self.n_output) # Not in OG

        self.fc1 = Linear(self.n_input, self.depth)
        self.elu = ELU()
        self.fc2 = Linear(self.depth, self.depth)
        init_mod(self)

        self.gate_norm = _GateAddNorm(
            n_input = self.depth,
            residual=self.n_output,
            depth=self.n_output,
            dropout=self.dropout,
        )

    def forward(self, x, residual_connect = None) -> Tensor:
        """
        Forward pass through the Gated Residual Network.

        Parameters
        ----------
        x : Tensor of shape (batch_size, sequence_length, n_input)
            Input tensor.
        residual_connect : Tensor of shape (batch_size, sequence_length, n_input) or None, optional
            Residual tensor to be added. If None, defaults to `x`.

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, n_output)
            Output tensor after processing through GRN block with gated residual connection and normalization.
        """

        if residual_connect is None:
            residual_connect = x

        if self.n_input != self.n_output:
            residual_connect = self.linear(residual_connect)

        residual_connect = self.norm(residual_connect)

        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        out = self.gate_norm(x, residual_connect)

        return out

class _GatedLinearUnit(Module):
    """
    Gated Linear Unit (GLU) module with optional dropout.

    This module applies a linear transformation to the input, doubling the output
    features to produce gates and values, then uses the GLU activation function to
    gate the output. Dropout is applied before the linear layer if specified.

    Parameters
    ----------
    n_input : int
        Number of input features.
    depth : int
        Output feature dimension after gating (half of linear layer output features).
    dropout : float, optional (default=0.1)
        Dropout probability applied before the linear transformation. If None, no dropout is applied.

    Attributes
    ----------
    fc : Linear
        Linear layer producing gates and values with output dimension `depth * 2`.
    dropout : Dropout or None
        Dropout layer applied before linear transformation.

    Methods
    -------
    forward(x)
        Applies dropout (if any), linear transformation, and GLU gating to the input tensor.
    """

    def __init__(
        self,
        n_input: int,
        depth: int,
        dropout: float = 0.1,
    ) ->None:
        super().__init__()

        if dropout is not None:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = dropout

        self.depth = depth or n_input
        self.fc = Linear(n_input, self.depth * 2)
        init_mod(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Gated Linear Unit.

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

################################### VSN Block ##########################################

class _VariableSelectionNetwork(Module):
    """
    Variable Selection Network for dynamic feature selection for the TFT.

    This module assigns dynamic, data-dependent importance weights to input features,
    enabling the model to focus on the most relevant variables at each time step.
    It uses a combination of per-variable Gated Residual Networks (GRNs) to transform
    each input separately and a joint GRN on the flattened embedding to compute
    soft attention weights over all input features.

    This approach improves interpretability and efficiency in models like the
    Temporal Fusion Transformer (TFT), especially when dealing with high-dimensional
    and mixed-type inputs.

    Parameters
    ----------
    n_input_dim : int
        Dimensionality of each individual input feature (i.e., feature size).
    n_input_features : int
        Total number of input features (variables).
    depth : int
        Number of hidden units used inside each GRN.
    dropout : float, optional
        Dropout probability used within the GRNs. Default is 0.1.

    Attributes
    ----------
    flattened_grn : _GatedResidualNetwork
        GRN applied to the concatenated feature embeddings, used to generate feature importance weights.
    single_variable_grns : ModuleList
        List of GRNs applied independently to each feature.
    softmax : Softmax
        Softmax function for converting attention scores into a probability distribution.

    Methods
    -------
    forward(x)
        Applies per-variable GRNs and computes attention-based feature selection.
        Returns a weighted sum of GRN outputs and the corresponding feature weights.
    """

    def __init__(
        self,
        n_input_dim: int,
        n_input_features: int,
        depth: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_input_dim = n_input_dim
        self.n_input_features = n_input_features
        self.depth = depth
        self.dropout = dropout

        if self.n_input_features > 1:
            self.flattened_grn = _GatedResidualNetwork(
                self.n_input_dim * self.n_input_features,
                min(self.depth, self.n_input_features),
                self.n_input_features,
                self.dropout,
                residual_connect=False,
            )

        self.single_variable_grns = ModuleList()

        for idx in range(self.n_input_features):
            self.single_variable_grns.append(_GatedResidualNetwork(
                n_input=self.n_input_dim,
                depth=min(self.n_input_dim, self.depth),
                n_output=self.depth,
                dropout=dropout,
            ))

        self.softmax = Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Variable Selection Network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, n_input).

        Returns
        -------
        out : Tensor
            Output tensor of shape (batch_size, sequence_length, depth), where each time step
            contains a weighted combination of the transformed features.
        sparse_weights : Tensor
            Tensor of shape (batch_size, sequence_length, 1, n_input_features) containing the
            soft attention weights assigned to each input feature.
        """

        if self.n_input_features > 1:
            var_outputs = []
            weight_inputs = []
            for idx in range(self.n_input_features):
                variable_embedding = x[:, :, idx].unsqueeze(-1)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[idx](variable_embedding))
            var_outputs = stack(var_outputs, dim=-1)

            flat_embedding = cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            out = var_outputs * sparse_weights
            out = out.sum(dim=-1)

        elif self.n_input_features == 1:
            variable_embedding = x[:, :, 0].unsqueeze(-1)
            out = self.single_variable_grns[0](variable_embedding)
            if out.ndim == 3:
                sparse_weights = ones(out.size(0), out.size(1), 1, 1, device=out.device)
            else:
                sparse_weights = ones(out.size(0), 1, 1, device=out.device)

        else:
            out = zeros(x.size(), device=x.device)
            if out.ndim == 3:
                sparse_weights = zeros(out.size(0), out.size(1), 1, 0, device=out.device)
            else:
                sparse_weights = ones(out.size(0), 1, 0, device=out.device)

        return out, sparse_weights

############################################## Attention head ############################################

class _ScaledDotProductAttention(Module):
    """
    Scaled dot-product attention mechanism.

    This module computes attention scores using the dot product between queries and keys,
    optionally scales the scores, applies a softmax function to obtain attention weights,
    and uses these weights to aggregate values. Optionally includes dropout after softmax.

    Parameters
    ----------
    dropout : float, optional
        Dropout probability to apply after the attention softmax. If None, no dropout is used.
    scale : bool, optional
        If True, scales attention scores by the square root of the key dimension. Default is True.

    Attributes
    ----------
    dropout : Dropout or None
        Dropout layer applied to the attention weights, if specified.
    softmax : Softmax
        Softmax function applied to the attention logits along the last dimension.
    scale : bool
        Whether to apply scaling to the attention logits.

    Methods
    -------
    forward(q, k, v, mask=None)
        Computes attention-weighted values and attention weights from input tensors.
    """
    def __init__(
        self,
        dropout: float = None,
        scale: bool = True,
    ) -> None:
        super().__init__()

        if dropout is not None:
            self.dropout = Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None) -> Tensor:
        """
        Forward pass of the scaled dot-product attention mechanism.

        Computes attention scores as the dot product between queries and keys,
        optionally scales them, applies a mask if provided, then normalizes
        with softmax. The resulting attention weights are used to compute
        a weighted sum over the values.

        Parameters
        ----------
        q : Tensor, shape [batch_size, sequence_length, model_depth // number_heads]
            Query tensor.
        k : Tensor, shape [batch_size, sequence_length, model_depth // number_heads]
            Key tensor.
        v : Tensor, shape [batch_size, sequence_length, model_depth // number_heads]
            Value tensor.
        mask : Tensor, optional, shape broadcastable to [batch_size, sequence_length, sequence_length]
            Boolean mask indicating positions to be masked (e.g., padding or future tokens).

        Returns
        -------
        out : Tensor, shape [batch_size, sequence_length, model_depth // number_heads]
            Resulting attention-weighted sum of values.
        attn : Tensor, shape [batch_size, sequence_length, sequence_length]
            Attention weights after softmax.
        """
        attn = bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        out = bmm(attn, v)

        return out, attn


class _InterpretableMultiHeadAttention(Module):
    """
    Implements an interpretable multi-head attention mechanism where each attention
    head shares the same value vector but uses separate query and key projections.

    This variant simplifies interpretability by decoupling the query/key learning across
    heads while enforcing a shared value representation. Useful in architectures where
    clarity of attention allocation is important (e.g., temporal attention).

    Parameters
    ----------
    n_head : int
        Number of attention heads.
    d_model : int
        Total feature dimension of the input and output.
    dropout : float, optional
        Dropout probability applied after attention and output projection. Default is 0.1.

    Attributes
    ----------
    v_layer : nn.Linear
        Shared linear projection for values across all heads.
    q_layer : nn.ModuleList
        Per-head linear projections for queries.
    k_layer : nn.ModuleList
        Per-head linear projections for keys.
    attention : _ScaledDotProductAttention
        Core attention mechanism computing attention weights and outputs.
    w_h : nn.Linear
        Final projection layer mapping averaged head output back to model dimension.
    dropout : nn.Dropout
        Dropout applied after attention and final projection.
    """
    def __init__(
        self,
        n_head: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = Dropout(p=dropout)

        self.v_layer = Linear(self.d_model, self.d_v)
        self.q_layer = ModuleList([Linear(self.d_model, self.d_q) for _ in range(n_head)])
        self.k_layer = ModuleList([Linear(self.d_model, self.d_k) for _ in range(n_head)])
        self.attention = _ScaledDotProductAttention()
        self.w_h = Linear(self.d_v, self.d_model, bias=False)

        init_mod(self)

    def forward(self, q, v, k, mask=None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for interpretable multi-head attention.

        Projects inputs into per-head queries and keys and shared values,
        computes attention per head, averages head outputs, and projects the result.

        Parameters
        ----------
        q : Tensor, shape [batch_size, seq_len, d_model]
            Query tensor.
        v : Tensor, shape [batch_size, seq_len, d_model]
            Value tensor.
        k : Tensor, shape [batch_size, seq_len, d_model]
            Key tensor.
        mask : Tensor, optional, shape broadcastable to [batch_size, seq_len, seq_len]
            Boolean mask indicating positions to ignore during attention.

        Returns
        -------
        out : Tensor, shape [batch_size, seq_len, d_model]
            Output of attention after head averaging and projection.
        attn : Tensor, shape [batch_size, seq_len, seq_len, n_head]
            Raw attention weights for each head.
        """

        heads = []
        attns = []

        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layer[i](q)
            ks = self.k_layer[i](k)
            head, attn = self.attention(qs, ks, vs, mask=mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = stack(attns, dim=2)

        out = mean(head, dim=2) if self.n_head > 1 else head
        out = self.w_h(out)
        out = self.dropout(out)

        return out, attn

class FinalBlock(Module):
    """
    Final processing block for TFT-based models for classification, regression, or forecasting tasks.

    This class applies final transformations to the TFT model output to prepare it for a specific task,
    such as classification, regression, or forecasting. For classification and regression, the output
    undergoes linear transformation, while for forecasting, it directly returns the selected output steps.

    Parameters
    ----------
    num_model_in_time_steps : int
        The number of input time steps.
    num_model_out_channels : int
        The number of output components (e.g., feature channels).
    num_model_out_time_steps : int
        The number of output time steps for forecasting or regression tasks.
    output_activation : str | None
        The output activation function to be applied (e.g., 'Softmax', 'Sigmoid').
    task : str
        Specifies the task type ('classification', 'regression', or 'forecasting').

    Attributes
    ----------
    expected_in_t : int
        Number of input time steps expected by the model.
    expected_in_c : int
        Number of input channels expected by the model.
    desired_out_c : int
        Number of output components for the model's final output.
    desired_out_t : int
        Number of output time steps required in the model output.
    task : str
        Task being performed by the model.
    act : nn.Module | None
        Activation function applied to the output if specified.
    trans : nn.Module
        Linear transformation layer applied to the input.

    Methods
    -------
    classify(x)
        Processes input for classification tasks.
    regress(x)
        Processes input for regression tasks.
    forward(x)
        Processes input and returns output depending on task type.
    """

    expected_in_t = None
    expected_in_c = None
    desired_out_t = None
    desired_out_c = None
    task = None

    def __init__(self, num_model_in_time_steps, num_model_out_channels,
                 num_model_out_time_steps, output_activation, depth, task):

        super(FinalBlock, self).__init__()

        self.expected_in_t = num_model_in_time_steps
        self.expected_in_c = num_model_out_channels
        self.desired_out_t = num_model_out_time_steps
        self.desired_out_c = num_model_out_channels
        self.depth = depth
        self.task = task

        self.act = None
        if output_activation is not None:
            if output_activation == 'Softmax':
                self.act = getattr(nn, output_activation)(dim=1)
            else:
                self.act = getattr(nn, output_activation)

        if task == 'classification':
            self.trans = Linear(self.depth * self.expected_in_t, self.desired_out_c, bias=False)
            init_mod(self.trans)
        elif task == 'regression':
            self.trans = Linear(self.depth * self.expected_in_t, self.desired_out_c * self.desired_out_t, bias=False)
            init_mod(self.trans)

    def classify(self, x):
        """
        Process input for classification tasks.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, expected_in_t, depth).

        Returns
        -------
        Tensor
            Classification output tensor, possibly with applied activation function.
        """
        x = x.reshape(x.shape[0], self.depth * self.expected_in_t)
        out = self.trans(x)
        if self.act is not None:
            out = self.act(out)
        return out

    def regress(self, x):
        """
        Process input for regression tasks.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, expected_in_t, depth).

        Returns
        -------
        Tensor
            Regression output tensor, reshaped as (batch_size, desired_out_t, desired_out_c).
        """
        x = x.reshape(x.shape[0], self.depth * self.expected_in_t)
        out = self.trans(x)
        if self.act is not None:
            out = self.act(out)
        out = out.reshape(out.shape[0], self.desired_out_t, self.desired_out_c)
        return out

    def forward(self, x):
        """
        Return output for an input batch, based on the specified task type.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, in_chunk, out_components).

        Returns
        -------
        Tensor
            Model output with shape depending on task type:
            - Classification: (batch_size, desired_out_c)
            - Regression: (batch_size, desired_out_t, desired_out_c)
            - Forecasting: (batch_size, desired_out_t, expected_in_c)

        Raises
        ------
        ValueError
            If the task type specified is not valid.
        """
        if self.task == 'classification':
            return self.classify(x)
        elif self.task == 'regression':
            return self.regress(x)
        elif self.task == 'forecasting':
            return x[:, -self.desired_out_t:, :]
        else:
            logger.error(self.task + " not a valid task type!")
            raise ValueError(f"Invalid task type: {self.task}")

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