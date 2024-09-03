"""This is a set of fully connected feed-forward blocks, 
with optional batch normalization and dropout, 
followed by a final linear layer, 
with optional output activation.

-----------
HiddenBlock
-----------

This block consists of 2 to 4 layers.
* is optional

[linear] -> [batch-norm*] -> [activation] -> [dropout*]

    -   [linear] = nn.Linear
    -   [batch-norm*] = nn.BatchNorm1d
    -   [activation] = Depends on the activation function. Set by getattr(nn, activation)(). See https://pytorch.org/docs/stable/nn.html for details.
    -   [dropout*] = nn.Dropout

Notes
-----
    - The MLP is capable of handling regression or classification tasks.
    - All HiddenBlocks have bias parameters.
    - All hidden layers have the same number of hidden units, defined by the ``width`` parameter.
    - This architecture flattens the input Tensor, and does not assume any temporal dynamics internally.
    - In practice. The output layer is just a ``HiddenBlock`` with no batch normalization, dropout or bias, and a different activation function.

"""  # noqa: INP001, D415, D400, D212, D205

from __future__ import annotations

__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com"
__description__ = "Example of MLP model architecture."

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

import numpy as np
from torch import nn

from helpers.logger import get_logger
logger = get_logger()

available_tasks = ("regression", "classification")

# The ranges for each hyperparameter (used later for Knowit Tuner module)
HP_ranges_dict = {
    "depth": range(1, 21, 1),
    "width": range(2, 1025, 1),
    "batchnorm": (True, False),
    "dropout": np.arange(0, 1.1, 0.1),
    "activations": (
        "ReLU",
        "LeakyReLU",
        "Tanh",
        "GLU",
    ),  # see Pytorch docs for more options
    "output_activation": (None, "Sigmoid", "Softmax"),
}


class Model(nn.Module):
    """Defines an MLP architecture.

    The multilayer perceptron (MLP) is a fully connected feedforward neural
    network with nonlinear activation functions.

    For more information on this architecture, see for example:

    [1] "Deep Learning" by I. Goodfellow, Y. Bengio, and A. Courville
    Link: https://www.deeplearningbook.org/

    [2] "Understanding Deep Learning" by S.J.D Prince
    Link: https://udlbook.github.io/udlbook/

    Parameters
    ----------
    input_dim : list[int], shape=[in_chunk, in_components]
        The shape of the input data. The "time axis" is along the first dimension.
    output_dim : list[int], shape=[out_chunk, out_components]
        The shape of the output data. The "time axis" is along the first dimension.
    task_name : str
        The type of task (classification or regression).
    depth : int, default=3
        The desired number of hidden layers.
    width : int, default=256
        The desired width (number of nodes) of each hidden layer.
    dropout : float | None, default=0.5
        Sets the dropout probability. If None, no dropout is applied.
    activations : str, default='ReLU'
        Sets the activation type for the hidden units. See https://pytorch.org/docs/stable/nn.html for details.
    output_activation : None | str, default=None
        Sets an output activation. See https://pytorch.org/docs/stable/nn.html for details.
        If None, no output activation.
    batchnorm : bool, default=True
        Whether to add batchnorm to hidden layers.

    Attributes
    ----------
    task_name : str | None
        The type of task (classification or regression).
    model_in_dim : int | None
        Number of model input features. Equal to in_chunk * in_component.
    model_out_dim : int | None
        Number of model output features. Equal to out_chunk * out_component.
    final_out_dim : list[int], shape=[out_chunk, out_components]
        The shape of the output data. The "time axis" is along the first dimension.
    model : nn.Sequential
        The entire model architecture.
    """
    task_name = None
    model_in_dim = None
    model_out_dim = None
    final_out_dim = None

    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        *,
        depth: int = 3,
        width: int = 256,
        dropout: float | None = 0.5,
        activations: str = 'ReLU',
        output_activation: None | str = None,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]

        if task_name not in available_tasks:
            logger.error('Only %s tasks are available for MLP. %s given.', str(available_tasks), task_name)
            exit(101)

        self.task_name = task_name
        self.model_in_dim = int(np.prod(input_dim))
        self.model_out_dim = int(np.prod(output_dim))
        self.final_out_shape = output_dim

        layers = [HiddenBlock(self.model_in_dim, width, batchnorm=batchnorm,
                              activation=activations, dropout=dropout, bias=True)]
        for _ in range(depth - 1):
            layers.append(HiddenBlock(width, width, batchnorm=batchnorm,
                                      activation=activations, dropout=dropout, bias=True))
        layers.append(HiddenBlock(width, self.model_out_dim, batchnorm=False,
                                  activation=output_activation, dropout=None, bias=False))
        self.model = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        """Return model output for an input batch.

        Parameters
        ----------
        x : Tensor, shape=[batch_size, in_chunk, in_components]
            An input tensor.

        Returns
        -------
        Tensor, shape=[batch_size, out_chunk, out_components] or [batch_size, num_classes]
            Model output.

        """
        x = x.view(x.shape[0], self.model_in_dim)
        out = self.model(x)
        if self.task_name == "regression":
            out_reshaped = out.view(x.shape[0], self.final_out_shape[0], self.final_out_shape[1])
            return out_reshaped
        else:
            return out


class HiddenBlock(nn.Module):

    """A hidden block of an MLP.

    This block consists of a linear layer, followed by an activation function.
    An optional batchnorm layer can be placed before the activation layer,
    and an optional dropout layer can be placed after.

    Parameters
    ----------
    in_dim : int
        The number of input features.
    out_dim : int
        The number of output features.
    batchnorm : bool
        Whether to include a batchnorm layer before the activation function.
    activation : str | None
        The activation function to be used. See https://pytorch.org/docs/stable/nn.html for details.
    dropout : float | None
        The dropout probability. If None, no dropout will be applied.
    bias : bool
        Whether the linear layer should have a bias vector.

    Attributes
    ----------
    block : nn.Module
        The hidden block.
    """
    def __init__(self, in_dim: int, out_dim: int, *, batchnorm: bool,
                 activation: str | None, dropout: float | None, bias: bool) -> None:

        super(HiddenBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim, bias)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        if activation is not None:
            if activation == "Softmax":
                layers.append(getattr(nn, activation)(dim=1))
            else:
                layers.append(getattr(nn, activation)())
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the hidden block to the input tensor."""
        return self.block(x)


