"""
---------------------
Multilayer Perceptron
---------------------

As an example, we define a multilayer perceptron (MLP) architecture using the
well-known Pytorch library.

The example shows one way of constructing the MLP which can either be shallow
or deep. The MLP is capable of handling regression or classification tasks.
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
    """Define MLP architecture.

    The multilayer perceptron (MLP) is a fully connected feedforward neural
    network with nonlinear activation functions.

    For more information on this architecture, see for example:
    [1] "Deep Learning" by I. Goodfellow, Y. Bengio, and A. Courville
    Link: https://www.deeplearningbook.org/

    [2] "Understanding Deep Learning" by S.J.D Prince
    Link: https://udlbook.github.io/udlbook/
    """

    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        depth: int = 3,
        width: int = 256,
        dropout: float = 0.5,
        activations: str = "ReLU",
        output_activation: None | str = None,
        *,
        batchnorm: bool = True,
    ) -> None:
        """Model constructor.

        Args:
        ----
            input_dim (list[int]):  The model's input features of shape
                                    [in_chunk, in_components].

            output_dim (list[int]): The model's output features of shape
                                    [out_chunk, out_components].

            task_name (str):        The type of task (classification or
                                    regression).

            depth (int):            The number of hidden layers. Default: 3.

            width (int):            The width of each hidden layer.
                                    Default: 256.

            dropout (float):
                                    Sets the dropout value. Default: 0.5.

            activations (str):
                                    Sets the activation type for the hidden
                                    units. Defaults: "ReLU".

            output_activation (None | str):
                                    Sets an output activation (needed for
                                    classification tasks). Default: None.

            batchnorm (bool):       Adds batchnorm to layers. Default: True.

        """
        super().__init__()  # type: ignore[reportUnknownMemberType]

        # Hyperparameters
        self.task_name = task_name
        self.hidden_depth = depth
        self.hidden_width = width
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activations = activations
        self.output_activation = output_activation

        self.model_in_dim = int(np.prod(input_dim))
        self.model_out_dim = int(np.prod(output_dim))
        self.final_out_dim = output_dim

        # Input layer
        layers: list[Module] = []
        layers.append(
            nn.Linear(
                self.model_in_dim,
                self.hidden_width,
                bias=True,
            ),
        )
        if self.batchnorm:
            layers.append(nn.BatchNorm1d(self.hidden_width))
        layers.append(getattr(nn, self.activations)())
        layers.append(nn.Dropout(p=self.dropout))

        # Construct MLP hidden layers
        layers = layers + self._build_hidden_layers()

        # Set output layer according to task type
        if task_name not in available_tasks:
            emsg = "Task must be either: 'regression' or 'classification'."
            raise NameError(emsg)
        self.task_name = task_name

        if self.task_name == "regression":
            output_layer = nn.Linear(
                self.hidden_width,
                self.model_out_dim,
                bias=False,
            )
            layers.append(output_layer)
        if self.task_name == "classification":
            output_layer = nn.Linear(
                self.hidden_width,
                self.model_out_dim,
                bias=False,
            )
            if self.output_activation == "Softmax":
                output_layer_activation = getattr(nn,
                                                  self.output_activation)(dim=1)
            elif self.output_activation is None:
                output_layer_activation = nn.Identity()
            else:
                output_layer_activation = getattr(nn, self.output_activation)()
            layers.append(output_layer)
            layers.append(output_layer_activation)

        # Merge layers together in Sequential
        self.model = nn.Sequential(*layers)


    def _build_hidden_layers(self) -> list[Module]:
        hidden_blocks: list[Module] = []
        for _ in range(self.hidden_depth - 1):
            hidden_blocks.append(
                nn.Linear(self.hidden_width, self.hidden_width, bias=True),
            )
            if self.batchnorm:
                hidden_blocks.append(nn.BatchNorm1d(self.hidden_width))
            hidden_blocks.append(getattr(nn, self.activations)())
            hidden_blocks.append(nn.Dropout(p=self.dropout))

        return hidden_blocks


    def _regression(self, x: Tensor) -> Tensor:
        """Return model output for an input batch for a regression task.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk * in_components)

        Returns:
        -------
            (Tensor):       Model output of shape
                            (batch_size, out_chunk, out_components)

        """
        out = self.model(x)

        return out.view(
            out.shape[0],
            self.final_out_dim[0],
            self.final_out_dim[1],
        )


    def _classification(self, x: Tensor) -> Tensor:
        """Return model output for an input batch for a classification task.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk * in_components)

        Returns:
        -------
            (Tensor):       Model output of shape
                            (batch_size, num_classes)

        """
        return self.model(x)


    def forward(self, x: Tensor) -> Tensor:
        """Return model output for an input batch.

        Args:
        ----
            x (Tensor):     An input tensor of shape
                            (batch_size, in_chunk, in_components).

        Returns:
        -------
            (Tensor):       Model output of shape
                            (batch_size, out_chunk, out_components) if regress-
                            ion.

                            Model output of shape
                            (batch_size, num_classes) if classification.

        """
        x = x.view(x.shape[0], self.model_in_dim)

        if self.task_name == "regression":
            return self._regression(x)

        return self._classification(x)
