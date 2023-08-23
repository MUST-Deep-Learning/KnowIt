__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains an example template for a custom model. This is an MLP.'

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
           'activations': 'ReLU'}

# Todo:
# 1) Many activation functions: need to ensure correct spelling with current solution   
# 2) For classification: what type of output ie sigmoid, softmax, etc
# 3) At setting the task type, both seem to be doing the same thing

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
                 activations: str = HP_dict['activations']):
        
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
        
        # Hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = depth
        self.hidden_width = width
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activations = activations
        
        # Construct MLP hidden layers
        self._build_hidden_layers()
        self.hidden_stack = nn.Sequential(*self.hidden_block)
        
        # Set output layer according to task type
        assert task_name in available_tasks, "Error: Task must be either: \'regression\' or \'classification\'"
        self.task_name = task_name
        
        if self.task_name == 'regression':
            self.output_layer = nn.Linear(self.hidden_width, self.output_dim, bias=True)
        if self.task_name == 'classification':
            self.output_layer = nn.Linear(self.hidden_width, self.output_dim, bias=True)    
        
            
    def _build_hidden_layers(self):
        """Construct the hidden layers of the MLP using the user-specified parameters"""
        
        self.hidden_block = []
        for _ in range(self.hidden_depth):
            self.hidden_block.append(nn.Linear(self.hidden_width, self.hidden_width, bias=True))
            if self.batchnorm:
                self.hidden_block.append(nn.BatchNorm1d(self.hidden_width))
            self.hidden_block.append(getattr(nn, self.activations)()) 
            self.hidden_block.append(nn.Dropout(p=self.dropout))
            

    def regression(self, input):
        """Todo"""
        
        # test
        output = self.output_layer(self.hidden_stack(input))
        
        return output

    def classification(self, input):
        """Todo"""
        
        output = self.output_layer(self.hidden_stack(input))
          
        return nn.Softmax()(output)

    def forward(self, input):
        """Todo"""
        
        if self.task_name == 'regression':
            output = self.regression(input)
            return output
        if self.task_name == 'classification':
            output = self.classification(input)
            return output
        return None