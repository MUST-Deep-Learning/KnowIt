"""
This is a 1D CNN architecture; it is similar to the TCN but non-causal by removing the padding and clipping.

The main stage is fully convolutional and performs 1D convolutions over the time domain.

The overall architecture consists of multiple ``ConvBlock`` modules followed by a ``FinalBlock`` module.

---------
ConvBlock
---------

This block consists of 4 to 5 layers.
* is optional

[convolution] -> [normalization*] -> [activation] -> [dropout] -> [residual connection*]

    -   [convolution]
            -   nn.Conv1d(num_input_components, num_filters)    ... if at input
            -   nn.Conv1d(num_filters, num_filters)             ... if in between
            -   nn.Conv1d(num_filters, num_output_components)   ... if at the end
            -   This layer performs 1D convolution over the time steps, with the input
                components as channels.
            -   It outputs a tensor of (batch_size, num_time_steps, num_filters)
    -   [normalization]
            -   depends on the normalization hyperparameter
                    -   nn.utils.weight_norm if normalization='weight'
                    -   nn.BatchNorm1d if normalization='batch'
                    -   skipped if normalization=None
    -   [activation]
            -   Depends on the activation function.
            -   Set by getattr(nn, activation)().
            -   See https://pytorch.org/docs/stable/nn.html for details.
    -   [dropout] = nn.Dropout
    -   [residual connection]
            -   The input to the block is added to the output.
            -   A 1x1 conv is used to resize the input if the input size != output size.
            
----------
FinalBlock
----------

After the CNN stage we have a tensor T(batch_size, num_input_time_steps, num_output_components).

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
    - The CNN is capable of handling regression, classification, and forecasting(WIP) tasks.
    - All conv layers have bias parameters.
    - All non-bias weights are initialized with nn.init.kaiming_uniform_(parameters) if dimension allow, otherwise nn.init.normal_(parameters) is used.
    - All bias weights are initialized with nn.init.zeros_(parameters).
    - Can also run in `variable length` mode (i.e. task='vl_regression'), where the number of timesteps in the input and output are equal.

""" # noqa: INP001, D415, D400, D212, D205

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Example of a 1D Convolutional Neural Network (CNN).'

import torch.nn as nn
import numpy as np
# from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm

from helpers.logger import get_logger
logger = get_logger()

available_tasks = ('regression', 'classification', 'forecasting', 'vl_regression')

# The ranges for each hyperparameter (used later for Knowit Tuner module)
HP_ranges_dict = {'depth': range(-1, 21, 1),
                  'num_filters': range(3, 1025, 1),
                  'kernel_size': range(3, 31, 1),
                  'normalization': ('batch', 'weight', None),
                  'dropout': np.arange(0, 1.1, 0.1),
                  'activations': ('ReLU', 'LeakyReLU', 'Tanh', 'GLU'),  # see Pytorch docs for more options
                  'output_activation': (None, 'Sigmoid', 'Softmax'),
                  'residual_connect': (True, False),
                  'dilation_base': (2, 30, 1)}


class Model(nn.Module):
    """Defines a 1D Convolutional Neural Network (CNN) architecture with a task-specific final block.

    This model supports various tasks,
    including classification, regression, and forecasting.

    Parameters
    ----------
    input_dim : list[int], shape=[in_chunk, in_components]
        The shape of the input data. The "time axis" is along the first dimension.
        Here, `in_chunk` represents the number of time steps, and `in_components` indicates
        the number of input features or channels.
    output_dim : list[int], shape=[out_chunk, out_components]
        The shape of the output data. The "time axis" is along the first dimension.
        `out_chunk` corresponds to the number of output time steps, and `out_components` refers to
        the number of output features or channels.
    task_name : str
        The type of task (classification, regression, or vl_regression).
    depth : int, default=-1
        The desired number of convolutional blocks to include in the CNN stage.
        If depth=-1, the minimum depth to ensure that the receptive field is larger than
        the input sequence 'in_chunk' is automatically calculated.
    num_filters : int, default=64
        The desired number of filters (also referred to as channels) per hidden convolutional block.
    kernel_size : int, default=3
        The desired kernel size for all filters. This parameter affects the size of the local
        receptive field of the convolutions.
    normalization : str | None, default='batch'
        The type of normalization to apply. Options are ('batch', 'weight', None).
    dropout : float | None, default=0.5
        Sets the dropout probability. If None, no dropout is applied, which may lead to overfitting.
    activations : str, default='ReLU'
        Sets the activation type for the convolutional layers. Refer to
        https://pytorch.org/docs/stable/nn.html for details on available activations.
    output_activation : None | str, default=None
        Sets an output activation. If None, no output activation is applied. See
        https://pytorch.org/docs/stable/nn.html for details.
    residual_connect : bool, default=True
        Whether to add a residual connection to each convolutional block to help in learning.
    dilation_base : int, default=2
        The base dilation factor for convolutional blocks, impacting the receptive field.

    Attributes
    ----------
    task_name : str
        The type of task being performed by the model.
    depth : int
        The number of convolutional blocks in the CNN stage.
    num_filters : int
        The number of filters (also channels) per hidden convolutional block.
    kernel_size : int
        The kernel size for all filters.
    normalization : str | None
        The type of normalization applied.
    dropout : float | None
        The dropout probability. If None, no dropout is applied.
    activations : str
        The activation type for the convolutional layers.
    output_activation : None | str
        The output activation applied, if any.
    residual_connect : bool
        Indicates whether a residual connection is added to each convolutional block.
    dilation_base : int
        The base dilation factor for convolutional blocks.
    num_model_in_time_steps : int
        The number of input time steps. Equal to the length of in_chunk.
    num_model_in_channels : int
        The number of input components. Equal to num_in_components.
    num_model_out_time_steps : int
        The number of output time steps. Equal to the length of out_chunk.
    num_model_out_channels : int
        The number of output components. Equal to num_out_components.
    network : nn.Module
        The network architecture built from convolutional blocks and the final task-specific block.
    """
    task_name = None
    depth = -1
    num_filters = 64
    kernel_size = 3
    normalization = 'batch'
    dropout = 0.3
    activations = 'ReLU'
    output_activation = None
    residual_connect = True
    dilation_base = 2
    num_model_in_time_steps = None
    num_model_in_channels = None
    num_model_out_time_steps = None
    num_model_out_channels = None

    def __init__(self,
                 input_dim: list,
                 output_dim: list,
                 task_name: str,
                 *,
                 depth: int = -1,
                 num_filters: int = 64,
                 kernel_size: int = 3,
                 normalization: bool | None = 'batch',
                 dropout: float | None = 0.3,
                 activations: str = 'ReLU',
                 output_activation: str | None = None,
                 residual_connect: bool = True,
                 dilation_base: int = 2
                 ) -> None:

        super(Model, self).__init__()

        if task_name not in available_tasks:
            logger.error(task_name + " must be one of " + str(available_tasks))
            exit(101)

        # Input data type check
        self._check_and_add_att('task_name', task_name, str)
        self._check_and_add_att('depth', depth, int)
        self._check_and_add_att('num_filters', num_filters, int)
        self._check_and_add_att('kernel_size', kernel_size, int)
        self._check_and_add_att('dropout', dropout, (float, type(None)))
        self._check_and_add_att('activations', activations, str)
        self._check_and_add_att('residual_connect', residual_connect, bool)
        self._check_and_add_att('dilation_base', dilation_base, int)
        self._check_and_add_att('output_activation', output_activation, (str, type(None)))
        self._check_and_add_att('normalization', normalization, (str, type(None)))

        if self.kernel_size < self.dilation_base:
            logger.warning('Kernel size %s < dilation base %s. There will be holes in the receptive field of CNN.',
                           str(self.kernel_size), str(self.dilation_base))

        if self.depth == -1 and self.task_name in ('vl_regression', ):
            logger.error('CNN depth=-1 and task_name in (vl_regression, ). '
                         'Depth cannot be automatically set for variable length inputs.')
            exit(101)

        self.num_model_in_time_steps = input_dim[0]
        self.num_model_in_channels = input_dim[1]
        self.num_model_out_time_steps = output_dim[0]
        self.num_model_out_channels = output_dim[1]

        min_depth = self._calc_min_depth(dilation_base, self.num_model_in_time_steps, kernel_size)
        if self.depth != -1 and self.depth < min_depth:
            logger.warning('CNN receptive field below input sequence length.')

        # build model arch
        self.network = self._build_fcn()

    def _check_and_add_att(self, name, val, expected) -> None:
        """Checks the variable (name) type (expected) before setting it as an attribute."""
        if not isinstance(val, expected):
            logger.error(name + " must be of type " + str(expected))
            exit(101)
        else:
            self.__setattr__(name, val)

    def _build_fcn(self, dilation_base=2):
        """Builds the fully convolutional stage of the CNN. """
        depth = self.depth
        if depth == -1:
            depth = int(np.ceil(np.emath.logn(dilation_base,
                                          (((self.num_model_in_time_steps - 1) *
                                            (dilation_base - 1)) /
                                           (self.kernel_size - 1)) + 1)))
            depth = max(depth, 1)
            logger.info('Using minimum CNN depth %s.', str(depth))
        self.depth = depth

        layers = []
        for i in range(depth):
            layers += [ConvBlock(n_inputs=self.num_model_in_channels if i == 0 else self.num_filters,
                                 n_outputs=self.num_model_out_channels if i == depth - 1 else self.num_filters,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 dilation=self.dilation_base ** i,
                                 padding=self.dilation_base ** i,
                                 dropout=self.dropout,
                                 normalization=self.normalization,
                                 activations=self.activations,
                                 residual_connect=self.residual_connect)]

        layers += [FinalBlock(self.num_model_in_time_steps, self.num_model_out_channels,
                              self.num_model_out_time_steps, self.output_activation, self.task_name)]

        return nn.Sequential(*layers)

    @staticmethod
    def _calc_min_depth(dilation_base, num_model_in_time_steps, kernel_size):
        """Calculate the minimum depth to ensure full coverage of the receptive field. """
        depth = int(np.ceil(np.emath.logn(dilation_base,
                                          (((num_model_in_time_steps - 1) *
                                            (dilation_base - 1)) /
                                           (kernel_size - 1)) + 1)))
        return depth

    def forward(self, x):
        """Return model output for an input batch.

        Parameters
        ----------
        x : Tensor, shape=[batch_size, in_chunk, in_components]
            An input tensor. See below for shape exception.

        Returns
        -------
        Tensor, shape=[batch_size, out_chunk, out_components] or [batch_size, num_classes]
            Model output.

        Notes
        -----
        - For 'vl_regression', the input tensor x will have the shape [batch_size, *, in_components], where * is variable length.

        """
        return self.network(x)


class FinalBlock(nn.Module):
    """
    Final processing block for CNN-based models for classification, regression, or forecasting tasks.

    This class applies final transformations to the CNN model output to prepare it for a specific task,
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
                 num_model_out_time_steps, output_activation, task):

        super(FinalBlock, self).__init__()

        self.expected_in_t = num_model_in_time_steps
        self.expected_in_c = num_model_out_channels
        self.desired_out_t = num_model_out_time_steps
        self.desired_out_c = num_model_out_channels
        self.task = task

        self.act = None
        if output_activation is not None:
            if output_activation == 'Softmax':
                self.act = getattr(nn, output_activation)(dim=1)
            else:
                self.act = getattr(nn, output_activation)

        if task == 'classification':
            self.trans = nn.Linear(self.expected_in_t * self.expected_in_c, self.desired_out_c, bias=False)
            init_mod(self.trans)
        elif task == 'regression':
            self.trans = nn.Linear(self.expected_in_t * self.expected_in_c,
                                   self.desired_out_c * self.desired_out_t, bias=False)
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
        x = x.reshape(x.shape[0], self.expected_in_t * self.expected_in_c)
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
        x = x.reshape(x.shape[0], self.expected_in_t * self.expected_in_c)
        out = self.trans(x)
        if self.act is not None:
            out = self.act(out)
        out = out.reshape(out.shape[0], self.desired_out_t, self.desired_out_c)
        return out

    def vl_regress(self, x):
        return x

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
        elif self.task == 'vl_regression':
            return self.vl_regress(x)
        else:
            logger.error(self.task + " not a valid task type!")
            raise ValueError(f"Invalid task type: {self.task}")


class ConvBlock(nn.Module):
    """
    A fully convolutional block for Temporal Convolutional Networks (CNNs).

    This block performs a 1D convolution, followed by optional normalization, activation,
    and dropout. It supports residual connections, where the input is added to the output
    if the number of input and output channels match or if an additional 1x1 convolution layer
    is applied for channel alignment.

    Parameters
    ----------
    n_inputs : int
        The number of input channels for the convolutional layer.
    n_outputs : int
        The number of output channels for the convolutional layer.
    kernel_size : int
        The size of the convolution kernel.
    stride : int
        The stride of the convolution.
    dilation : int
        The dilation factor for the convolution.
    padding : int
        The padding size for the convolution.
    dropout : float
        Dropout probability, applied after the activation.
    normalization : str | None
        Type of normalization to apply; 'weight' for weight normalization, 'batch' for batch
        normalization, or None for no normalization.
    activations : str
        Activation function to use (e.g., 'ReLU', 'LeakyReLU').
    residual_connect : bool
        If True, adds a residual connection from input to output to improve stability
        in deeper networks.

    Attributes
    ----------
    block : nn.Sequential
        Sequential container for the convolutional layer, padding, normalization, activation,
        and dropout layers.
    res_connect : nn.Module or None
        Identity layer or 1x1 convolution for residual connection alignment, if residual
        connections are enabled.

    Methods
    -------
    forward(x)
        Forward pass of the ConvBlock. Applies the convolutional block to input `x`
        and adds a residual connection if specified.

    """
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout, normalization,
                 activations, residual_connect):

        super(ConvBlock, self).__init__()

        layers = [nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                            padding=padding, dilation=dilation)]
        if normalization == 'weight':
            layers[0] = weight_norm(layers[0])
        elif normalization == 'batch':
            layers.append(nn.BatchNorm1d(n_outputs))
        elif normalization is None:
            pass
        else:
            logger.error('Unknown normalization type %s.', normalization)
            exit(101)
        layers.append(getattr(nn, activations)())
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

        init_mod(self.block)

        self.res_connect = None
        if residual_connect:
            if n_inputs != n_outputs:
                self.res_connect = nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
                init_mod(self.res_connect)
            else:
                self.res_connect = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the ConvBlock.

        This method applies the convolutional block to the input tensor `x`. If residual connections are enabled,
        it adds the residual (shortcut) connection to the output, then transposes it back to the original format.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, n_inputs) where `n_inputs`
            is the number of input channels.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, n_outputs), where `n_outputs`
            is the number of output channels.
        """
        x = x.transpose(1, 2)
        out = self.block(x)
        if self.res_connect:
            res = self.res_connect(x)
            out = out + res
        out = out.transpose(1, 2)
        return out


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