__author__ = 'tiantheunissen@gmail.com'
__description__ = 'This is a Temporal Convolutional Network.'

""" 

This is an architecture that performs 1D convolutions over the time domain, 
and regards components as channels. It uses dilated causal convolutions and padding, 
to ensure that there is no information leakage from future values.
The convolutional stage outputs a sequence equal in length to the output.

It contains a number of ConvBlock modules followed by a FinalBlock module.


------------
ConvBlock
------------

This module consists of 5 layers.

[convolution] -> [normalization*] -> [activation] -> [dropout] -> [residual connection*]

    -   [convolution] = nn.Conv1d(num_input_components, num_filters)    ... if at input
                        nn.Conv1d(num_filters, num_filters)             ... if in between
                        nn.Conv1d(num_filters, num_output_components)   ... if at the end
            -   This layer performs 1D convolution over the time steps, with the the input components
                as channels.
            -   It outputs a tensor of (batch_size, num_time_steps, num_filters)
    -   [normalization] = depends on the normalization hyperparameter
            - nn.utils.weight_norm if normalization='weight'
            - nn.BatchNorm1d if normalization='batch'
            - skipped if normalization=None
    -   [activation] = depends on the 'activations' hyperparameter
    -   [dropout] = nn.Dropout
    -   [residual connection] = The input to the block is added to the output. 
            A 1x1 conv is used to resize the input if the input size != output size


"""

from typing import Union
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

model_name = "TCN"

available_tasks = ('regression', 'classification')

# Default hyperparameter values
HP_defaults_dict = {'depth': 3,
                    'num_filters': 64,
                    'kernel_size': 3,
                    'normalization': 'weight',
                    'dropout': 0.3,
                    'activations': 'ReLU',
                    'output_activation': None,
                    'residual_connect': True,
                    'latent_dim': 256}

# The ranges for each hyperparameter (used later for Knowit Tuner module)
HP_ranges_dict = {'depth': range(1, 21, 1),
                  'num_filters': range(3, 1025, 1),
                  'kernel_size': range(3, 31, 1),
                  'normalization': ('batch', 'weight', None),
                  'dropout': np.arange(0, 1.1, 0.1),
                  'activations': ('ReLU', 'LeakyReLU', 'Tanh', 'GLU'),  # see Pytorch docs for more options
                  'output_activation': (None, 'Sigmoid', 'Softmax'),
                  'residual_connect': (True, False),
                  'latent_dim': (2, 1024, 1)}


class Model(nn.Module):
    """
    Temporal Convolutional Network
    Paper link: https://arxiv.org/abs/1803.01271
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
                 output_activation: str = HP_defaults_dict['output_activation'],
                 residual_connect: bool = HP_defaults_dict['residual_connect'],
                 latent_dim: int = HP_defaults_dict['latent_dim']
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
        self.__check_and_add_arg('residual_connect', residual_connect, bool)
        self.__check_and_add_arg('latent_dim', latent_dim, int)
        if output_activation is not None:
            self.__check_and_add_arg('output_activation', output_activation, str)
        else:
            self.output_activation = None
        if normalization is not None:
            self.__check_and_add_arg('normalization', normalization, str)
        else:
            self.normalization = None

        self.num_model_in_time_steps = input_dim[0]
        self.num_model_in_channels = input_dim[1]
        self.num_model_out_time_steps = output_dim[0]
        self.num_model_out_channels = output_dim[1]

        # build model arch
        self.network = self.__build_fcn()

    def __check_and_add_arg(self, name, val, expected):
        if not isinstance(val, expected):
            raise TypeError(name + " must be of type " + str(expected))
        else:
            self.__setattr__(name, val)

    def __build_fcn(self):
        # import math
        # dilation_base = 2
        # num_layers = math.ceil(
        #     math.log(
        #         (self.num_model_in_time_steps - 1)
        #         * (dilation_base - 1)
        #         / (self.kernel_size - 1)
        #         / 2
        #         + 1,
        #         dilation_base,
        #     )
        # )
        #
        # effective_depth = np.ceil(self.num_model_in_time_steps/(self.kernel_size - 1))

        layers = []
        for i in range(self.depth):
            layers += [ConvBlock(n_inputs=self.num_model_in_channels if i == 0 else self.num_filters,
                                 n_outputs=self.num_model_out_channels if i == self.depth - 1 else self.num_filters,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 dilation=2 ** i,
                                 padding=(self.kernel_size-1) * (2 ** i),
                                 dropout=self.dropout,
                                 normalization=self.normalization,
                                 activations=self.activations,
                                 residual_connect=True)]

        layers += [FinalBlock(self.num_model_in_time_steps, self.num_model_out_channels,
                              self.num_model_out_time_steps, self.output_activation, self.task_name)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FinalBlock(nn.Module):
    def __init__(self, num_model_in_time_steps, num_model_out_channels,
                 num_model_out_time_steps, output_activation, task):

        super(FinalBlock, self).__init__()

        self.expected_in_t = num_model_in_time_steps
        self.expected_in_c = num_model_out_channels
        self.desired_out_t = num_model_out_time_steps
        self.desired_out_c = num_model_out_channels
        self.task = task

        self.output_activation = output_activation
        if output_activation:
            if output_activation == 'Softmax':
                self.act = getattr(nn, self.output_activation)(dim=1)
            else:
                self.act = getattr(nn, self.output_activation)

        if task == 'classification':
            self.trans = nn.Linear(self.expected_in_t * self.expected_in_c, self.desired_out_c)
            self.init_mod(self.trans)
        elif task == 'regression':
            self.trans = nn.Linear(self.expected_in_t * self.expected_in_c,
                                   self.desired_out_c * self.desired_out_t)
            self.init_mod(self.trans)

    @staticmethod
    def init_mod(mod):
        for name, parameters in mod.named_parameters():
            if 'weight' in name:
                nn.init.normal_(parameters)
            elif 'bias' in name:
                nn.init.zeros_(parameters)

    def classify(self, x):
        x = x.reshape(x.shape[0], self.expected_in_t * self.expected_in_c)
        out = self.trans(x)
        if self.output_activation:
            out = self.act(out)
        return out

    def regress(self, x):
        x = x.reshape(x.shape[0], self.expected_in_t * self.expected_in_c)
        out = self.trans(x)
        if self.output_activation:
            out = self.act(out)
        out = out.reshape(out.shape[0], self.desired_out_t, self.desired_out_c)
        return out

    def forward(self, x):

        if self.task == 'classification':
            return self.classify(x)
        elif self.task == 'regression':
            return self.regress(x)
        elif self.task == 'forecasting':
            return x[:, -self.desired_out_t:, :]
        else:
            raise TypeError(self.task + " not a valid task type!")


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout, normalization,
                 activations, residual_connect):

        super(ConvBlock, self).__init__()

        conv_layer = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        if normalization == 'weight':
            self.conv = weight_norm(conv_layer)
        elif normalization == 'batch':
            self.conv = nn.Sequential(conv_layer,
                                      nn.BatchNorm1d(n_outputs))
        else:
            self.conv = conv_layer

        self.net = nn.Sequential(self.conv,Chomp1d(padding),
                                 getattr(nn, activations)(),
                                 nn.Dropout(p=dropout))
        self.init_mod(self.net)

        self.res_connect = None
        if residual_connect:
            if n_inputs != n_outputs:
                self.res_connect = nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
                self.init_mod(self.res_connect)
            else:
                self.res_connect = nn.Identity()

    @staticmethod
    def init_mod(mod):
        for name, parameters in mod.named_parameters():
            if 'weight' in name:
                nn.init.normal_(parameters)
            elif 'bias' in name:
                nn.init.zeros_(parameters)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x)
        if self.res_connect:
            res = self.res_connect(x)
            out = out + res
        out = out.transpose(1, 2)
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()