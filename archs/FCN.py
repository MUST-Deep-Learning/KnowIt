__author__ = 'tiantheunissen@gmail.com'
__description__ = 'This is a 1D Fully Convolutional Network.'

""" This is basically a TCN without the causal convolutions. """

from typing import Union
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

model_name = "FCN"

available_tasks = ('regression', 'classification')

# Default hyperparameter values
HP_defaults_dict = {'depth': 3,
                    'num_filters': 32,
                    'kernel_size': 3,
                    'normalization': 'weight',
                    'dropout': 0.3,
                    'activations': 'ReLU',
                    'output_activation': None,
                    'dilation_base': 2,
                    'stride': 1}

# The ranges for each hyperparameter (used later for Knowit Tuner module)
HP_ranges_dict = {'depth': range(1, 21, 1),
                  'num_filters': range(3, 1025, 1),
                  'kernel_size': range(3, 31, 1),
                  'normalization': ('batch', 'weight', None),
                  'dropout': np.arange(0, 1.1, 0.1),
                  'activations': ('ReLU', 'LeakyReLU', 'Tanh', 'GLU'),  # see Pytorch docs for more options
                  'output_activation': (None, 'Sigmoid', 'Softmax'),
                  'dilation_base': range(2, 8, 1),
                  'stride': range(1, 11, 1)}


class Model(nn.Module):
    """
    Temporal convolutional network
    Paper link: ???
    """

    def __init__(self,
                 input_dim: tuple,
                 output_dim: tuple,
                 task_name: str,
                 depth: int = HP_defaults_dict['depth'],
                 num_filters: int = HP_defaults_dict['num_filters'],
                 kernel_size: int = HP_defaults_dict['kernel_size'],
                 normalization: bool = HP_defaults_dict['normalization'],
                 dropout: float = HP_defaults_dict['dropout'],
                 activations: str = HP_defaults_dict['activations'],
                 output_activation: Union[str, None] = HP_defaults_dict['output_activation'],
                 dilation_base: int = HP_defaults_dict['dilation_base'],
                 stride: int = HP_defaults_dict['stride']
                 ):

        super(Model, self).__init__()

        if task_name not in available_tasks:
            raise TypeError(task_name + " must be of type " + str(available_tasks))

        # Input data type check
        self.__check_and_add_arg('task_name', task_name, str)
        self.__check_and_add_arg('depth', depth, int)
        self.__check_and_add_arg('num_filters', num_filters, int)
        self.__check_and_add_arg('kernel_size', kernel_size, int)
        self.__check_and_add_arg('dropout', dropout, float)
        self.__check_and_add_arg('activations', activations, str)
        if output_activation is not None:
            self.__check_and_add_arg('output_activation', output_activation, str)
        else:
            self.output_activation = None
        self.__check_and_add_arg('dilation_base', dilation_base, int)
        self.__check_and_add_arg('stride', stride, int)
        if normalization is not None:
            self.__check_and_add_arg('normalization', normalization, str)
        else:
            self.normalization = None

        self.model_in_dim = input_dim[1]
        if self.task_name == 'classification':
                self.model_out_dim = output_dim[1]
        else:
                self.model_out_dim = output_dim[0]

        self.latent_dim = input_dim[0]


        # build model arch
        self.network = self.__build_fcn()

    def __check_and_add_arg(self, name, val, expected):
        if not isinstance(val, expected):
            raise TypeError(name + " must be of type " + str(expected))
        else:
            self.__setattr__(name, val)

    def __build_fcn(self):

        layers = []
        for i in range(self.depth):
            layers += [ConvBlock(n_inputs=self.model_in_dim if i == 0 else self.num_filters,
                                 n_outputs=self.model_out_dim if i == self.depth - 1 else self.num_filters,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 dilation=self.dilation_base ** i,
                                 padding=(self.kernel_size - 1) * (self.dilation_base ** i),
                                 dropout=self.dropout,
                                 normalization=self.normalization,
                                 activations=self.activations)]

        layers += [FinalBlock(self.latent_dim, self.model_out_dim, self.output_activation)]

        return nn.Sequential(*layers)

    def regression(self, input):
        """Performs a forward pass over the regression TCN.

        Returns a tensor of shape (batch_size, output_dim)"""

        return self.network(input)

    def classification(self, input):
        """Performs a forward pass over the classifier TCN.

        Returns a tensor of shape (batch_size, output_dim)"""

        return self.network(input)

    def forward(self, input):
        """Based on the user task, passes the input to the relevant forward function.

        Returns an output tensor of shape (batch_size, output_dim)"""

        if self.task_name == 'regression':
            output = self.regression(input)
            return output
        if self.task_name == 'classification':
            output = self.classification(input)
            return output


class FinalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, output_activation):
        super(FinalBlock, self).__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        self.output_activation = output_activation
        self.lin = nn.Linear(n_inputs*n_outputs, n_outputs)
        if output_activation:
            self.act = getattr(nn, self.output_activation)()

    def init_weights(self):
        self.lin.weight.data.normal_(0, 0.01)

    def forward(self, x):

        x = x.reshape(x.shape[0], self.n_inputs * self.n_outputs)

        out = self.lin(x)
        if self.output_activation:
            out = self.act(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout, normalization, activations):
        super(ConvBlock, self).__init__()

        if normalization == 'weight':
            self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation))
        elif normalization == 'batch':
            self.conv = nn.Sequential(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                              stride=stride, padding=dilation, dilation=dilation),
                                      nn.BatchNorm1d(n_outputs))
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                              stride=stride, padding=dilation, dilation=dilation)

        self.activation = getattr(nn, activations)()
        self.dropout = nn.Dropout(p=dropout)

        self.down_sample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None

        self.net = nn.Sequential(self.conv, self.activation, self.dropout)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.down_sample is not None:
            self.down_sample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        x = x.transpose(1, 2)

        out = self.net(x)
        res = x if self.down_sample is None else self.down_sample(x)


        out = out + res

        out = out.transpose(1, 2)

        return out
