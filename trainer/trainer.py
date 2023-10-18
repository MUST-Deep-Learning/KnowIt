__author__ = 'randlerabe@gmail.com, tiantheunissen@gmail.com'
__description__ = 'Contains the trainer module.'

#   todo:
#   > Build a class that accepts data structure from data module
#       1.1) User inputs: optimizer, loss, max_epochs. From the /model folder, object instance of model is passed as parameter to Trainer class
#       1.2) Use PL to build trainer.
#   > From data module classes RegressionDataset or ClassificationDataset, will receive
#   train and val dataloaders.
#   > Logger (tensorboard, wandb, etc)?
#   > Save model weights (checkpoint) - > see mustnet. In env, write own def to save model output in own subdir. After every epoch, save in memory model and performance (dict?).
#       Depending on early stopping, save best at the end on hdd.
#   > matmul_precision: I think pl is sensitive towards this?
#   > devices: gpu, cpu, etc
#   > test model using test dataloader from data module?
#   > callbacks?
#   > change imports to from

import os
from env import env_user

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class PLModel(pl.LightningModule):
    
    def __init__(self, 
                 loss: str,
                 learning_rate: float, 
                 optimizer: str,
                 model: object):
        super().__init__()
        
        self.loss = loss
        self.lr = learning_rate
        self.optimizer = optimizer
        self.model = model
        
    def _loss_function(self, y_pred, y):
        # todo: what if user needs to provide more info when calling loss?
        return getattr(F, self.loss)(y_pred, y)
        
    def training_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self._loss_function(y_pred, y)
        
        self.log("Train_loss", loss, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self._loss_function(y_pred, y)
        
        self.log("Val_loss", loss, on_epoch=True, on_step=False)
        
        return loss    
    
    
    def test_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self._loss_function(y_pred, y)
        
        self.log("Test_loss", loss, on_epoch=True, on_step=False)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.model.parameters(), lr=self.lr)
        
        return optimizer


class KITrainer():
    
    def __init__(self,
                 train_device: str,
                 loss_fn: str,
                 optim: str,
                 max_epochs: int,
                 early_stopping: bool,
                 learning_rate: float,
                 learning_rate_schedule: str,
                 loaders: tuple,
                 model: object):
        
        """"Args:
        - train_device: (device),
        - loss_fn: (str) -> CE, MSE, W_CE
        - optim: (str) -> Adam, SGD, RAdam
        - max_epochs: (int) -> 100
        - early_stopping: (bool) -> True
        - learning_rate: (float) -> 1e-03
        - learning_rate_schedule: (str) -> ..."""
        
        # device to use
        self.train_device = train_device
        
        # model hyperparameters
        self.loss_fn = loss_fn
        self.optim = optim
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        
        # instance of model class
        self.model = model
        
        # dataloaders from data module
        self.train_dataloader = loaders[0] # this expects an ordering -> can I do this without assuming that
        self.val_dataloader = loaders[1]
        
        # todo: perform checks on above vals
        
        # initialize PL object
        self.lit_model = PLModel(loss=self.loss_fn,
                                 optimizer=self.optim, 
                                 model=self.model, 
                                 learning_rate=self.learning_rate)
        
    def _fit_model(self):
        # todo: make choice of logger optional?
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=env_user.project_dir)
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             accelerator=self.train_device, 
                             logger=tb_logger,
                             devices='auto', # what other options here?
                             callbacks=[EarlyStopping(monitor="Val_loss", mode="min") if self.early_stopping==True else None][0]) 
        
        trainer.fit(model=self.lit_model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        
    
    def _test_model(self):
        # todo: load model from checkpoint or continue using pl.Trainer above?
        #model = 
        pass
        
    def _save_model_state(self):
        pass
    
        
    




