__author__ = 'randlerabe@gmail.com, tiantheunissen@gmail.com'
__description__ = 'Contains an example template for a custom model. This is an MLP.'

# Utils
from typing import Union
import warnings

# Machine Learning Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

model_name = "MLP"

available_tasks = ('regression', 'classification')

HP_dict = {'depth': 3,
           'width': 256,
           'batchnorm': True,
           'dropout': 0.5,
           'activations': 'ReLU',
           'output_activation': None}

class Model(nn.Module):
    """
    Model Name
    Paper link: ???
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 task_name: str, 
                 depth: int = HP_dict['depth'], 
                 width: int = HP_dict['width'], 
                 batchnorm: bool = HP_dict['batchnorm'], 
                 dropout: float = HP_dict['dropout'], 
                 activations: str = HP_dict['activations'],
                 output_activation: Union[str, None] = HP_dict['output_activation']
                 ):
        
        super(Model, self).__init__()
        
        # Input data type check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be of type int")
        if not isinstance(task_name, str):
            raise TypeError("task_name must be of type str")
        if not isinstance(depth, int):
            raise TypeError("depth must be of type int")
        if not isinstance(width, int):
            raise TypeError("width must be of type int")
        if not isinstance(batchnorm, bool):
            raise TypeError("batchnorm must be of type bool")
        if not isinstance(dropout, float):
            raise TypeError("dropout must be of type float")
        if not isinstance(activations, str):
            raise TypeError("activations must be of type str")
        if not isinstance(output_activation, Union[str, None]):
            raise TypeError("output_activation must be of type str or None")
        
        # Warnings
        if output_activation == 'Sigmoid' and output_dim > 1:
            warnings.warn("Sigmoid gives incorrect results for output_dim > 1. Softmax is recommended in this case.")
        
        # Hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = depth
        self.hidden_width = width
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activations = activations
        self.output_activation = output_activation
        
        # Input layer
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_width, bias=True))
        if self.batchnorm:
            layers.append(nn.BatchNorm1d(self.hidden_width))
        layers.append(getattr(nn, self.activations)()) 
        layers.append(nn.Dropout(p=self.dropout))
        
        # Construct MLP hidden layers
        layers = layers + self._build_hidden_layers()
        
        # Set output layer according to task type
        assert task_name in available_tasks, "Error: Task must be either: \'regression\' or \'classification\'"
        self.task_name = task_name
        
        if self.task_name == 'regression':
            output_layer = nn.Linear(self.hidden_width, self.output_dim, bias=False)
            layers.append(output_layer)
        if self.task_name == 'classification':
            output_layer = nn.Linear(self.hidden_width, self.output_dim, bias=False)
            if self.output_activation == 'Softmax':
                output_layer_activation = getattr(nn, self.output_activation)(dim=1)
            else:
                output_layer_activation = getattr(nn, self.output_activation)()
            layers.append(output_layer)
            layers.append(output_layer_activation)
            
        # Merge layers together in Sequential
        self.model = nn.Sequential(*layers)
            
    def _build_hidden_layers(self):
        """Construct the hidden layers of the MLP using the user-specified parameters."""
        
        hidden_blocks = []
        for _ in range(self.hidden_depth - 1):
            hidden_blocks.append(nn.Linear(self.hidden_width, self.hidden_width, bias=True))
            if self.batchnorm:
                hidden_blocks.append(nn.BatchNorm1d(self.hidden_width))
            hidden_blocks.append(getattr(nn, self.activations)()) 
            hidden_blocks.append(nn.Dropout(p=self.dropout))
        
        return hidden_blocks
            

    def regression(self, input):
        """Performs a forward pass over the regression MLP.
        
        Returns a tensor of shape (batch_size, output_dim)"""
        
        return self.model(input)

    def classification(self, input):
        """Performs a forward pass over the classifier MLP.
        
        Returns a tensor of shape (batch_size, output_dim)"""
          
        return self.model(input)

    def forward(self, input):
        """Based on the user task, passes the input to the relevant forward function.
        
        Returns an output tensor of shape (batch_size, output_dim)"""
        
        if self.task_name == 'regression':
            output = self.regression(input)
            return output
        if self.task_name == 'classification':
            output = self.classification(input)
            return output