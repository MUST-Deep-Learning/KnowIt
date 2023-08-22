__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains an example template for a custom model. This is an MLP.'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

available_tasks = ('regression', 'classification')

HP_dict = {'depth': 3,
           'width': 256,
           'batchnorm': True,
           'dropout': 0.5,
           'activations': 'ReLU'}


class Model(nn.Module):
    """
    Model Name
    Paper link: ???
    """

    def __init__(self, hp_dict):
        super(Model, self).__init__()
        self.task_name = hp_dict['task']
        self.input_dim = hp_dict['input_dim']
        self.output_dim = hp_dict['output_dim']
        self.hidden_dim = hp_dict['hidden_dim']
        self.batchnorm = hp_dict['batchnorm']

        self.hidden_block = [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                             nn.BatchNorm1d(self.hidden_dim),
                             nn.ReLU(),
                             nn.Dropout(p=)]

        if self.task_name == 'regression':
            self.projection = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        if self.task_name == 'classification':
            self.projection = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def regression(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None