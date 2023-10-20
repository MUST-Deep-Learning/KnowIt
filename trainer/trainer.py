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
#   > optimizer: momentum arg?
#   > average loss per epoch, not per batch
#   > hardcode performence metrics based on task: regression (same as your loss), classification (accuracy: multiclass, binary: f1) -> also per epoch like previous point

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
                 model: object,
                 learning_rate_scheduler: dict):
        super().__init__()
        
        self.loss = loss
        self.lr = learning_rate
        if learning_rate_scheduler:
            self.lr_scheduler = learning_rate_scheduler.pop('lr_scheduler') # save choice & remove from dict 
            if 'monitor' in learning_rate_scheduler.keys():
                self.monitor = learning_rate_scheduler.pop('monitor') # save choice & remove from dict
            else:
                self.monitor = None
            self.lr_scheduler_kwargs = learning_rate_scheduler # only kwargs remain
        else:
            self.lr_scheduler = None
        self.optimizer = optimizer
        self.model = model
        
    def __get_loss_function(self):
        # todo: what if user needs to provide more info when calling loss?
        return getattr(F, self.loss)
    
    def __get_optim(self):
        # todo: use kwargs for variable user parameters
        return getattr(torch.optim, self.optimizer)
    
    def __get_lr_scheduler(self):
        return getattr(torch.optim.lr_scheduler, self.lr_scheduler)
        
    def training_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self.__get_loss_function()(y_pred, y)
        
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self.__get_loss_function()(y_pred, y)
        
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        
        return loss    
    
    
    def test_step(self, batch, batch_idx):
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        loss = self.__get_loss_function()(y_pred, y)
        
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        
        return loss
        
    def configure_optimizers(self):
        # get user's optimizer
        optimizer = self.__get_optim()(self.model.parameters(), lr=self.lr)
        
        # get user's lr scheduler
        if self.lr_scheduler:
            scheduler = self.__get_lr_scheduler()(optimizer, **self.lr_scheduler_kwargs) # unpack any kwargs needed for choice of optimizer
            if self.monitor: # todo: does monitor only get used here?
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor}}
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}        


class KITrainer():
    
    def __init__(self,
                 train_device: str,
                 loss_fn: str,
                 optim: str,
                 max_epochs: int,
                 early_stopping: bool,
                 learning_rate: float,
                 loaders: tuple,
                 model: object,
                 learning_rate_scheduler: dict = {}):
        
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
        self.learning_rate_scheduler = learning_rate_scheduler # dict: choice of lr_scheduler and kwargs
        
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
                                 learning_rate=self.learning_rate,
                                 learning_rate_scheduler=self.learning_rate_scheduler)
        
    def fit_model(self):
        # todo: make choice of logger optional?
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=env_user.project_dir)
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             accelerator=self.train_device, 
                             logger=tb_logger,
                             devices='auto', # what other options here?
                             callbacks=[EarlyStopping(monitor="val_loss", mode="min") if self.early_stopping==True else None][0]) 
        
        trainer.fit(model=self.lit_model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        
    
    def test_model(self):
        # todo: load model from checkpoint or continue using pl.Trainer above?
        #model = 
        pass
        
    def __save_model_state(self):
        pass
    
        
    




