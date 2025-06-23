from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "potgieterharmen@gmail.com"
__description__ = "Example of an TFT model architecture."

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import LSTM, Parameter, Sigmoid
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from copy import deepcopy
import numpy as np
import math

from torch.nn.functional import layer_norm
from torch.onnx.symbolic_opset9 import embedding
from torchvision.prototype.models import depth

from helpers.logger import get_logger
logger = get_logger()

available_tasks = ("regression", "classification", "forcast")

HP_ranges_dict = {
    "n_loss": range(1, 6),
    "n_targets": range(1, 6),
    "depth": range(1, 6),
    "lstm_depth": range(1, 6),
    "num_attention_heads": range(1, 6),
    "dropout": np.arange(0, 1.1, 0.1),
    "num_static_componenets": range(1, 6),
    "hidden_continuous_size": range(1, 6),
}

class Model(nn.Module):

    task_name = None
    model_out_dim = None
    final_out_dim = None

    def __init__(self,
                 input_dim: list,
                 output_dim: list,
                 task_name: str,
                 *,
                 depth: int = 16,
                 lstm_depth: int = 1,
                 num_attention_heads: int = 1,
                 dropout: float | None = 0.1,
                 full_attention: bool = False,
                 hidden_embedding_size: int = 1,
                 norm_type: str = "LayerNorm",
                 quantiles: list = None,
                 output_activation: str | None = None,

                 ):

        super().__init__()
        n_targets = len(output_dim)
        n_loss = len(output_dim)
        self.num_model_out_time_steps = output_dim[0]
        self.num_model_out_channels = output_dim[1]
        self.num_model_in_time_steps = input_dim[0]
        self.num_model_in_channels = input_dim[1]
        self.decoder_time_steps = output_dim[0]
        self.encoder_time_steps = input_dim[0] - output_dim[0]
        # self.decoder_steps = num_encoder_steps

        self.task_name = task_name
        self.n_targets, self.n_loss = output_dim
        self.n_targets = n_targets
        self.n_loss = n_loss
        self.depth = depth
        self.lstm_depth = lstm_depth
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_embedding_size = hidden_embedding_size  # Vector size of embedding
        self.full_attention = full_attention

        self.output_activation = output_activation
        self.layer_norm = nn.LayerNorm

        # self.reals = input_dim[-1] # Length of real valuead inputs
        # self.encoder_variables = input_dim[-1] # Input data before prediction

        self.batch_size_last = -1
        self.attention_mask = None

        self._attn_out_weights = None
        self._static_covariate_var = None
        self._encoder_sparse_weights = None
        self._decoder_sparse_weights = None

        # Continuous variable processing
        self.prescalers = [nn.Linear(1, self.hidden_embedding_size) for _ in range(self.num_model_in_channels)]
        # static_input_sizes = {
        #     str(x): self.input_embeddings.output_size[x]
        #     for x in range(self.static_cat_components)
        # }
        # # Add static components which does not need embeddings(reals)
        # static_input_sizes.update(
        #     {
        #         str(x): self.hidden_embedding_size
        #         for x in range(self.static_real_components)
        #     }
        # )

        # self.static_covariate_vsn = _VariableSelectionNetwork(
        #     n_input=static_input_sizes,
        #     depth=self.depth,
        #     input_embedding_flags={name: True for name in range(self.static_cat_components)},
        #     dropout=self.dropout,
        #     prescalers=self.prescalers
        # )

        # Static encoders
        # self.variable_selection_sce = _GatedResidualNetwork(
        #     n_input=depth,
        #     depth=self.depth,
        #     n_output=self.depth,
        #     dropout=self.dropout,
        #     )
        # self.lstm_vs_sce = _GatedResidualNetwork(
        #     n_input=self.depth,
        #     depth=self.depth,
        #     n_output=self.depth,
        #     dropout=self.dropout,
        #     )
        # self.lstm_hidden_sce = _GatedResidualNetwork(
        #     n_input=self.depth,
        #     depth=self.depth,
        #     n_output=self.depth,
        #     dropout=self.dropout,
        #     )
        # self.static_context_enrichment = _GatedResidualNetwork(
        #     n_input=self.depth,
        #     depth=self.depth,
        #     n_output=self.depth,
        #     dropout=self.dropout,
        #     )
        # encoder_input_sizes = {
        #     str(name): self.hidden_embedding_size for name in range(self.encoder_variables)
        # }
        # encoder_input_sizes = [self.hidden_embedding_size for _ in range(self.encoder_variables)]
        # decoder_input_sizes = [self.hidden_embedding_size for _ in range(self.decoder_variables)]

        #Encoder
        self.lstm_encoder_vsn = _VariableSelectionNetwork(
            n_input_dim=self.hidden_embedding_size,
            n_input_features=self.num_model_in_channels,
            depth=self.depth,
            dropout=self.dropout,
            context_size=self.depth,
            prescalers=self.prescalers,
        )

        self.lstm_decoder_vsn = _VariableSelectionNetwork(
            n_input_dim=self.hidden_embedding_size,
            n_input_features=self.num_model_in_channels,
            depth=self.depth,
            dropout=self.dropout,
            context_size=self.depth,
            prescalers=self.prescalers,
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
            layer_norm=self.layer_norm
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
            layer_norm=self.layer_norm,
        )

        #Output layer
        self.pre_out_gan = _GateAddNorm(
            n_input=self.depth,
            dropout=None,
        )
        self.output_layer = nn.Linear(
            self.depth,
            self.n_targets * self.n_loss
        )

        self.final_output_layer = FinalBlock(self.num_model_in_time_steps, self.num_model_out_channels,
                                             self.num_model_out_time_steps, self.output_activation, self.depth, self.task_name)
    @property
    def num_static_component(self):

        return len(self.static_variables)

    @staticmethod
    def get_attention_mask_future(
            encoder_length: int,
            decoder_length: int,
            batch_size: int,
            device: str,
            full_attention: bool
    ) -> torch.Tensor:
        if full_attention:
            decoder_mask = torch.zeros((decoder_length, decoder_length), dtype=torch.bool, device=device)
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step

        encoder_mask = torch.zeros(batch_size, encoder_length, dtype=torch.bool, device=device)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1)
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Tensor) -> Tensor: #, reset_lstm_state: bool = True) -> Tensor:

        reset_lstm_state = True
        # Batch, time, variable

        batch_size = x.shape[0]
        # input_vectors_past = [
        #     x[..., idx].unsqueeze(-1) for idx in range(self.encoder_variables)
        # ]

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
            h0 = torch.zeros(self.lstm_depth, batch_size, self.depth, device=x.device)
            c0 = torch.zeros(self.lstm_depth, batch_size, self.depth, device=x.device)
        else:
            h0, c0 = self.hidden_state, self.cell_state

        encoder_out, (hn, cn) = self.lstm_encoder(input=embeddings_varying_encoder, hx=(h0, c0))

        decoder_out, _ = self.lstm_decoder(input=embeddings_varying_decoder, hx=(hn, cn))

        lstm_layer = torch.cat([encoder_out, decoder_out], dim=1)
        input_embeddings = torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)

        self.hidden_state = hn.detach()
        self.cell_state = cn.detach()

        lstm_out = self.lstm_out_gan(lstm_layer, input_embeddings)

        # Attention head
        # static_context_enriched = self.static_context_enrichment(static_embedding)

        attn_in = self.static_enrichment_grn(x=lstm_out) #, context=self.expand_static_context(context=static_context_enriched, time=time_steps))

        attn_out, att_out_weights = self.attention(
            q=attn_in[:, self.encoder_time_steps:],
            k=attn_in,
            v=attn_in,
            mask=self.attention_mask,
        )

        attn_out = self.decoder_gan(x=attn_out, residual_connect=attn_in[:, self.encoder_time_steps:])
        decoder_out = self.decoder_grn(x=attn_out)

        #Out
        pre_out_gan = self.pre_out_gan(decoder_out, lstm_out[:, self.encoder_time_steps:])

        # Output layer
        out = self.output_layer(pre_out_gan)

        final_out = self.final_output_layer(pre_out_gan)

        self._attn_out_weights = att_out_weights
        self._encoder_out_weights = encoder_sparse_weights
        self._decoder_out_weights = decoder_sparse_weights

        return final_out

################################# Sub-Modules #################################################

# Work in progress

# class _MultiEmbedding(nn.Module):
#     def __init__(self, embedding_sizes, variable_names):
#         super().__init__()
#         self.embedding_sizes = embedding_sizes
#         self.variable_names = variable_names
#
#         self.embeddings = nn.ModuleDict({
#             name: nn.Embedding(*embedding_sizes[name]) for name in variable_names
#         })
#
#     def forward(self, x: Tensor) -> Tensor:
#         embeddings = {name: self.embeddings[name](x)}
#         return

##################################### GRU BLOCK ###############################################
class _TimeDistributedInterpolation(nn.Module):
    def __init__(
        self,
        n_output: int,
        batch_first: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.n_output = n_output
        self.batch_first = batch_first
        self.trainable = trainable

        if self.trainable:
            self.mask = Parameter(torch.zeros(self.n_output, dtype=torch.float32))
            self.gate = Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(
            x.unsqueeze(1), self.n_output, mode='linear', align_corners=True
        ).squeeze(1)

        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0

        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:

            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            ) # samples, timesteps, output_size
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # timesteps, samples, output_size

        return y


class _ResampleNorm(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        layer_norm: nn.LayerNorm = nn.LayerNorm,
        trainable: bool = True
    ):
        super().__init__()
        self.input_size = n_input
        self.n_output = n_output
        self.layer_norm = layer_norm
        self.trainable = trainable

        if self.input_size != self.n_output:
            # self.resample = _TimeDistributedInterpolation(
            #     self.n_output, batch_first=True, trainable=False
            # )
            self.resample = nn.Linear(self.input_size, self.n_output)

        if self.trainable: #Used for learning which features to suppress and emphasise
            self.mask = nn.Parameter(torch.zeros(self.n_output, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = self.layer_norm(self.n_output)

    def forward(self, x):
        if self.input_size != self.n_output:
            x = self.resample(x)

        if self.trainable:
            x = x * self.gate(self.mask) * 2.0
        x = self.norm(x)
        return x

class _AddNorm(nn.Module):
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
            # self.resample = _TimeDistributedInterpolation(
            #     self.n_input,
            #     batch_first=True,
            #     trainable=False
            # )
            self.resample = nn.Linear(self.n_input, self.residual)
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.n_input, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.n_input)

    def forward(self, x: torch.Tensor, residual_connect: torch.Tensor):
        if self.n_input != self.residual:
            residual_connect = self.resample(residual_connect)

        if self.trainable:
            residual_connect = residual_connect * self.gate(self.mask) * 2.0
        out = self.norm(x + residual_connect)

        return out

class _GateAddNorm(nn.Module):
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


class _GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        n_input: int,
        depth: int,
        n_output: int,
        dropout: float = 0.1,
        context_size: int =None,
        residual_connect: bool = True,
        layer_norm: nn.LayerNorm = nn.LayerNorm,
    ):

        super().__init__()

        self.n_input = n_input
        self.depth = depth
        self.n_output = n_output
        self.dropout = dropout
        self.context_size = context_size
        self.residual_connect = residual_connect

        # if self.n_input != self.n_output:
        #     residual_size = self.n_input
        # else:
        #     residual_size = self.n_output

        if self.n_output != self.n_input:
            self.resample_norm = _ResampleNorm(self.n_input, self.n_output, layer_norm=layer_norm)

        self.fc1 = nn.Linear(self.n_input, self.depth)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.depth, bias=False)

        self.fc2 = nn.Linear(self.depth, self.depth)
        self.initialize()

        self.gate_norm = _GateAddNorm(
            n_input = self.depth,
            residual=self.n_output,
            depth=self.n_output,
            dropout=self.dropout,
            trainable=False
        )

    def initialize(self):
        for name, parameters in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(parameters)
            elif 'fc1' in name or 'fc2' in name:
                try:
                    nn.init.kaiming_uniform_(parameters)
                except:
                    nn.init.normal_(parameters)

    def forward(self, x, context = None, residual_connect = None):

        if residual_connect is None:
            residual_connect = x

        if self.n_input != self.n_output:# and not self.residual_connect:
            residual_connect = self.resample_norm(residual_connect)

        x = self.fc1(x)
        if context != None:
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        out = self.gate_norm(
            x,
            residual_connect)

        return out

class _GatedLinearUnit(nn.Module):
    def __init__(
        self,
        n_input: int,
        depth: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.depth = depth or n_input
        self.fc = nn.Linear(n_input, self.depth * 2)
        self.initialize()

    def initialize(self):
        for name, parameters in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(parameters)
            elif 'fc' in name:
                try:
                    nn.init.kaiming_uniform_(parameters)
                except:
                    nn.init.normal_(parameters)

    def forward(self, x: torch.Tensor):
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)
        out = F.glu(x, dim=-1)

        return out


################################### VSN Block ##########################################

class _VariableSelectionNetwork(nn.Module):
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
        prescalers: List[nn.Linear] = None,
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

        self.single_variable_grns = nn.ModuleList()
        self.prescalers = nn.ModuleList()

        for idx in range(self.n_input_features):
            self.single_variable_grns.append(_GatedResidualNetwork(
                n_input=self.n_input_dim,
                depth=min(self.n_input_dim, self.depth),
                n_output=self.depth,
                dropout=dropout,
            ))

        if prescalers is not None:
            self.prescalers = nn.ModuleList(prescalers)

        self.softmax = nn.Softmax(dim=-1)

    # @property
    # def input_size_total(self):
    #     return sum(self.n_input)

    # @property
    # def num_inputs(self):
    #     return len(self.n_input)

    def forward(self, x: Tensor, context: torch.Tensor = None):

        if self.n_input_features > 1:
            var_outputs = []
            weight_inputs = []
            for idx in range(self.n_input_features):
                variable_embedding = x[:, :, idx].unsqueeze(-1)
                variable_embedding = self.prescalers[idx](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[idx](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            out = var_outputs * sparse_weights
            out = out.sum(dim=-1)

        elif self.n_input_features == 1:
            variable_embedding = x[:, :, 0].unsqueeze(-1)
            variable_embedding = self.prescalers[0](variable_embedding)
            out = self.single_variable_grns[0](variable_embedding)
            if out.ndim == 3:
                sparse_weights = torch.ones(out.size(0), out.size(1), 1, 1, device=out.device)
            else:
                sparse_weights = torch.ones(out.size(0), 1, 1, device=out.device)

        else:
            out = torch.zeros(context.size(), device=context.device)
            if out.ndim == 3:
                sparse_weights = torch.zeros(out.size(0), out.size(1), 1, 0, device=out.device)
            else:
                sparse_weights = torch.ones(out.size(0), 1, 0, device=out.device)

        return out, sparse_weights

class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        max_seq_len=160,
    ):
        super().__init__()
        assert (d_model % 2 == 0), "d_model must be even"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for position in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[position, i] = (torch.sin(position / (10000 ** ((2 * i) / d_model))))
                pe[position, i + 1] = (torch.cos(position / (10000 ** ((2 * (i + 1)) / d_model))))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            out = x + pe

            return out

############################################## Attention head ############################################

class _ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            dropout: float = None,
            scale: bool = True,
    ):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        out = torch.bmm(attn, v)
        return out, attn


class _InterpretableMultiHeadAttention(nn.Module):
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
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layer = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(n_head)])
        self.k_layer = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(n_head)])
        self.attention = _ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

    def initialize(self):
        for name, parameter in self.named_parameters():
            if 'bias' not in name:
                try:
                    nn.init.kaiming_uniform_(parameter)
                except:
                    nn.init.normal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(self, q, v, k, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        out = torch.mean(head, dim=2) if self.n_head > 1 else head
        out = self.w_h(out)
        out = self.dropout(out)

        return out, attn

class FinalBlock(nn.Module):
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
        #     self.trans = nn.Linear(self.expected_in_t * self.expected_in_c, self.desired_out_c, bias=False)
        #     init_mod(self.trans)
        # elif task == 'regression':
        #     self.trans = nn.Linear(self.expected_in_t * self.expected_in_c,
        #                            self.desired_out_c * self.desired_out_t, bias=False)

        if task == 'classification':
            self.trans = nn.Linear(self.depth * self.desired_out_t, self.desired_out_c, bias=False)
            init_mod(self.trans)
        elif task == 'regression':
            self.trans = nn.Linear(self.depth * self.desired_out_t, self.desired_out_c * self.desired_out_t, bias=False)
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
                nn.init.kaiming_uniform_(parameters)
            except:
                nn.init.normal_(parameters)
        elif 'bias' in name:
            nn.init.zeros_(parameters)