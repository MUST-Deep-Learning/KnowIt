__author__ = 'randlerabe@gmail.com, tiantheunissen@gmail.com'
__description__ = 'Contains the ki_trainer module.'

#   todo:
#   > Build a class that accepts data structure from data module
#       1.1) User inputs: optimizer, loss, max_epochs. From the /model folder, object instance of model is passed as parameter to Trainer class
#       1.2) Use PL to build trainer.
#   > From data module classes RegressionDataset or ClassificationDataset, will receive
#   train and val dataloaders.
#   > To think about: later want to add tuner; PL has learning rate tuner.
#   > Logger (tensorboard, wandb, etc)?
#   > Save model weights (checkpoint)?
#   > matmul_precision: I think pl is sensitive towards this?
#   > devices: gpu, cpu, etc
#   > test model using test dataloader from data module?
#   > print info about models using pl or torch's built in methods/attr
#   > callbacks?

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl

class Trainer(pl.LightningModule):
    
    def __init__(self, 
                 max_epochs: int, 
                 loss: torch.nn.functional, 
                 optimizer: torch.optim,
                 model: object):
        super().__init__()
        
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        
    def training_step(self):
        pass
    
    def validation_step(self):
        pass
    
    def test_step(self):
        pass
    
    def configure_optimizers(self):
        pass