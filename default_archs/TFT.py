"""


------------------------
VariableSelectionNetwork
------------------------

--------------------
GatedResidualNetwork
--------------------

-----------
GateAddNorm
-----------

----------
FinalBlock
----------

After the TCN stage we have a tensor T(batch_size, num_input_time_steps, num_output_components).

If task_name = 'regression'
    -   T is flattened to T(batch_size, num_input_time_steps * num_output_components)
        a linear layer and output activation is applied, and it is reshaped to the desired output
        T(batch_size, num_output_time_steps, num_output_components).

If task_name = 'classification'
    -   T is flattened to T(batch_size, num_input_time_steps * num_output_components)
        a linear layer is applied, which outputs T(batch_size, num_output_components).

If task_name = 'forecast' (WIP)
    -   T(batch_size, num_output_time_steps, num_output_components) is return where the
        num_output_time_steps is the last chunk from num_input_time_steps.

Notes
-----
    -
    -
    -
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

HP_ranges_dict = {
    "n_loss": range(1, 6),
    "n_targets": range(1, 6),
    "depth": range(1, 6),
    "lstm_depth": range(1, 6),
    "num_attention_heads": range(1, 6),
    "dropout": arange(0, 1.1, 0.1),
    "num_static_components": range(1, 6),
    "hidden_continuous_size": range(1, 6),
}

class Model(Module):

    task_name = None
    depth = 16
    lstm_depth = 16
    num_attention_heads = 1
    dropout = 0.1
    full_attention = True
    output_activation = None

    def __init__(self,
                 input_dim: list,
                 output_dim: list,
                 task_name: str,
                 *,
                 depth: int = 64,
                 lstm_depth: int = 16,
                 num_attention_heads: int = 4,
                 dropout: float | None = 0.1,
                 full_attention: bool = True,
                 quantiles: list = None,
                 output_activation: str | None = None,
    ) -> None:

        super().__init__()

        self.num_model_out_time_steps = output_dim[0]
        self.num_model_out_channels = output_dim[1]
        self.num_model_in_time_steps = input_dim[0]
        self.num_model_in_channels = input_dim[1]
        self.decoder_time_steps = output_dim[0]
        self.encoder_time_steps = input_dim[0] - output_dim[0]

        self.task_name = task_name
        self.depth = depth
        self.lstm_depth = lstm_depth
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.full_attention = full_attention

        self.output_activation = output_activation

        self.batch_size_last = -1
        self.attention_mask = None

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
            context_size=self.depth,
        )

        self.lstm_decoder_vsn = _VariableSelectionNetwork(
            n_input_dim=1,
            n_input_features=self.num_model_in_channels,
            depth=self.depth,
            dropout=self.dropout,
            context_size=self.depth,
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
            context_size=self.depth,
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
            num_model_in_time_steps=self.num_model_in_time_steps,
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

        reset_lstm_state = True
        batch_size = x.shape[0]

        input_vectors_past = x[:,:self.encoder_time_steps,:]
        input_vectors_future = x[:,self.encoder_time_steps:,:]

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

class _ResampleNorm(Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        trainable: bool = True
    ):
        super().__init__()
        self.input_size = n_input
        self.n_output = n_output
        self.trainable = trainable

        if self.input_size != self.n_output:
            self.resample = Linear(self.input_size, self.n_output)

        if self.trainable: #Used for learning which features to suppress and emphasise
            self.mask = Parameter(zeros(self.n_output, dtype=float))
            self.gate = Sigmoid()
        self.norm = LayerNorm(self.n_output)

    def forward(self, x):
        if self.input_size != self.n_output:
            x = self.resample(x)

        if self.trainable:
            x = x * self.gate(self.mask) * 2.0
        x = self.norm(x)
        return x

class _AddNorm(Module):
    def __init__(
        self,
        n_input: int,
        residual: int = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_input = n_input
        self.residual = residual or n_input
        self.trainable = trainable

        if self.n_input != self.residual:
            self.resample = Linear(self.n_input, self.residual)

        if self.trainable:
            self.mask = Parameter(zeros(self.n_input, dtype=float))
            self.gate = Sigmoid()

        self.norm = LayerNorm(self.n_input)

    def forward(self, x: Tensor, residual_connect: Tensor):
        if self.n_input != self.residual:
            residual_connect = self.resample(residual_connect)

        if self.trainable:
            residual_connect = residual_connect * self.gate(self.mask) * 2.0
        out = self.norm(x + residual_connect)

        return out

class _GateAddNorm(Module):
    def __init__(
        self,
        n_input: int,
        depth: int = None,
        residual: int = None,
        trainable: bool = True,
        dropout: float = None,
    ):
        super().__init__()
        self.n_input = n_input
        self.depth = depth or n_input
        self.residual = residual or self.depth
        self.dropout = dropout
        self.trainable = trainable

        self.glu = _GatedLinearUnit(self.n_input, depth=self.depth, dropout=self.dropout)
        self.add_norm = _AddNorm(self.depth, residual=self.residual, trainable=self.trainable)

    def forward(self, x, residual_connect):
        x = self.glu(x)
        x = self.add_norm(x, residual_connect)
        return x


class _GatedResidualNetwork(Module):
    def __init__(
        self,
        n_input: int,
        depth: int,
        n_output: int,
        dropout: float = 0.1,
        context_size: int =None,
        residual_connect: bool = True,
    ):

        super().__init__()

        self.n_input = n_input
        self.depth = depth
        self.n_output = n_output
        self.dropout = dropout
        self.context_size = context_size
        self.residual_connect = residual_connect

        if self.n_output != self.n_input:
            self.resample_norm = _ResampleNorm(self.n_input, self.n_output)

        self.fc1 = Linear(self.n_input, self.depth)
        self.elu = ELU()

        if self.context_size is not None:
            self.context = Linear(self.context_size, self.depth, bias=False)

        self.fc2 = Linear(self.depth, self.depth)
        init_mod(self)

        self.gate_norm = _GateAddNorm(
            n_input = self.depth,
            residual=self.n_output,
            depth=self.n_output,
            dropout=self.dropout,
            trainable=False
        )

    def forward(self, x, context = None, residual_connect = None):

        if residual_connect is None:
            residual_connect = x

        if self.n_input != self.n_output:
            residual_connect = self.resample_norm(residual_connect)

        x = self.fc1(x)
        if context != None:
            x = x + context

        x = self.elu(x)
        x = self.fc2(x)
        out = self.gate_norm(x, residual_connect)

        return out

class _GatedLinearUnit(Module):
    def __init__(
        self,
        n_input: int,
        depth: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dropout is not None:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = dropout

        self.depth = depth or n_input
        self.fc = Linear(n_input, self.depth * 2)
        init_mod(self)

    def forward(self, x: Tensor):
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)
        out = glu(x, dim=-1)

        return out

################################### VSN Block ##########################################

class _VariableSelectionNetwork(Module):
    """
    Implements a variable selection network used for prioritizing input features
    in complex models.

    This class serves as a specialized neural network module designed for variable
    selection by assigning weights to input features, such that the importance of
    each input feature in subsequent layers is dynamically determined. The model
    leverages Gated Residual Networks (GRNs) for processing both individual
    features and flattened embeddings. The selective weighting mechanism enables
    dimensionality reduction and improved generalization in tasks requiring input
    selection.

    Attributes
    ----------
    n_input_dim : int
        Dimensionality of the input features.
    n_input_features : int
        Number of input features being processed by the network.
    depth : int
        Number of hidden units in each layer of the Gated Residual Networks (GRNs).
    dropout : float
        Dropout rate applied within the Gated Residual Networks.
    context_size : int, optional
        Size of the context vector used in gated operations, if applicable.
    """
    def __init__(
        self,
        n_input_dim: int,
        n_input_features: int,
        depth: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        self.n_input_dim = n_input_dim
        self.n_input_features = n_input_features
        self.depth = depth
        self.dropout = dropout
        self.context_size = context_size

        if self.n_input_features > 1:
            if self.context_size is not None:
                self.flattened_grn = _GatedResidualNetwork(
                    self.n_input_dim * self.n_input_features,
                    min(self.depth, self.n_input_features),
                    self.n_input_features,
                    self.dropout,
                    self.context_size,
                    residual_connect=False,
                )
            else:
                self.flattened_grn = _GatedResidualNetwork(
                    self.n_input_dim * self.n_input_features,
                    min(self.depth, self.n_input_features),
                    self.n_input_features,
                    self.dropout,
                    residual_connect=False,
                )

        self.single_variable_grns = ModuleList()
        self.prescalers = ModuleList()

        for idx in range(self.n_input_features):
            self.single_variable_grns.append(_GatedResidualNetwork(
                n_input=self.n_input_dim,
                depth=min(self.n_input_dim, self.depth),
                n_output=self.depth,
                dropout=dropout,
            ))

        self.softmax = Softmax(dim=-1)

    def forward(self, x: Tensor, context: Tensor = None):

        if self.n_input_features > 1:
            var_outputs = []
            weight_inputs = []
            for idx in range(self.n_input_features):
                variable_embedding = x[:, :, idx].unsqueeze(-1)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[idx](variable_embedding))
            var_outputs = stack(var_outputs, dim=-1)

            flat_embedding = cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
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
            out = zeros(context.size(), device=context.device)
            if out.ndim == 3:
                sparse_weights = zeros(out.size(0), out.size(1), 1, 0, device=out.device)
            else:
                sparse_weights = ones(out.size(0), 1, 0, device=out.device)

        return out, sparse_weights

############################################## Attention head ############################################

class _ScaledDotProductAttention(Module):
    def __init__(
            self,
            dropout: float = None,
            scale: bool = True,
    ):
        super().__init__()
        if dropout is not None:
            self.dropout = Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):

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
    def __init__(
            self,
            n_head: int,
            d_model: int,
            dropout: float = 0.1,
            ):
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

    This class applies final transformations to the TCN model output to prepare it for a specific task,
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

        # if task == 'classification':
        #     self.trans = Linear(self.expected_in_t * self.expected_in_c, self.desired_out_c, bias=False)
        #     init_mod(self.trans)
        # elif task == 'regression':
        #     self.trans = Linear(self.expected_in_t * self.expected_in_c,
        #                            self.desired_out_c * self.desired_out_t, bias=False)

        if task == 'classification':
            self.trans = Linear(self.depth * self.desired_out_t, self.desired_out_c, bias=False)
            init_mod(self.trans)
        elif task == 'regression':
            self.trans = Linear(self.depth * self.desired_out_t, self.desired_out_c * self.desired_out_t, bias=False)
            init_mod(self.trans)

    def classify(self, x):
        """
        Process input for classification tasks.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, expected_in_t, expected_in_c).

        Returns
        -------
        Tensor
            Classification output tensor, possibly with applied activation function.
        """
        x = x.reshape(x.shape[0], self.depth * self.desired_out_t)
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
            Input tensor of shape (batch_size, expected_in_t, expected_in_c).

        Returns
        -------
        Tensor
            Regression output tensor, reshaped as (batch_size, desired_out_t, desired_out_c).
        """
        x = x.reshape(x.shape[0], self.depth * self.desired_out_t)
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