__author__ = 'randlerabe@gmail.com, tiantheunissen@gmail.com'
__description__ = 'This is an MLP.'

from typing import Union
import torch.nn as nn
import numpy as np

model_name = "MLP"

available_tasks = ('regression', 'classification')

# Default hyperparameter values
HP_defaults_dict = {'depth': 3,
           'width': 256,
           'batchnorm': True,
           'dropout': 0.5,
           'activations': 'ReLU',
           'output_activation': None}

# The ranges for each hyperparameter (used later for Knowit Tuner module)
HP_ranges_dict = {'depth': range(1, 21, 1),
           'width': range(2, 1025, 1),
           'batchnorm': (True, False),
           'dropout': np.arange(0, 1.1, 0.1),
           'activations': ('ReLU', 'LeakyReLU', 'Tanh', 'GLU'), # see Pytorch docs for more options
           'output_activation': (None, 'Sigmoid', 'Softmax')}

class Model(nn.Module):
    """
    Model Name
    Paper link: ???
    """

    def __init__(self, 
                 input_dim: tuple,
                 output_dim: tuple,
                 task_name: str, 
                 depth: int = HP_defaults_dict['depth'], 
                 width: int = HP_defaults_dict['width'], 
                 batchnorm: bool = HP_defaults_dict['batchnorm'], 
                 dropout: float = HP_defaults_dict['dropout'], 
                 activations: str = HP_defaults_dict['activations'],
                 output_activation: Union[str, None] = HP_defaults_dict['output_activation']
                 ):
        
        super(Model, self).__init__()
        
        # Input data type check
        if not isinstance(input_dim, tuple):
            raise TypeError("input_dim must be of type int")
        if not isinstance(output_dim, tuple):
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
        if not isinstance(output_activation, str) and output_activation is not None:
            raise TypeError("output_activation must be of type str or None")
        
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
        layers = []
        layers.append(nn.Linear(self.model_in_dim, self.hidden_width, bias=True))
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
            output_layer = nn.Linear(self.hidden_width, self.model_out_dim, bias=False)
            layers.append(output_layer)
        if self.task_name == 'classification':
            output_layer = nn.Linear(self.hidden_width, self.model_out_dim, bias=False)
            if self.output_activation == 'Softmax':
                output_layer_activation = getattr(nn, self.output_activation)(dim=1)
            elif self.output_activation is None:
                output_layer_activation = nn.Identity()
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

        out = self.model(input)
        out = out.view(out.shape[0], self.final_out_dim[0], self.final_out_dim[1])
        
        return out

    def classification(self, input):
        """Performs a forward pass over the classifier MLP.
        
        Returns a tensor of shape (batch_size, output_dim)"""

        out = self.model(input)
          
        return out

    def forward(self, input):
        """Based on the user task, passes the input to the relevant forward function.
        
        Returns an output tensor of shape (batch_size, output_dim)"""

        input = input.view(input.shape[0], self.model_in_dim)
        
        if self.task_name == 'regression':
            output = self.regression(input)
            return output
        if self.task_name == 'classification':
            output = self.classification(input)
            return output