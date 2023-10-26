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
#   > hardcode performence metrics based on task: regression (same as your loss), 
#       classification (accuracy: multiclass, binary: f1) -> also per epoch like previous point
#   > save logging results as csv
#   > refactor some of the code

import os
from typing import Union
from env import env_user

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class PLModel(pl.LightningModule):
    
    def __init__(self, 
                 loss: Union[str, dict],
                 learning_rate: float, 
                 optimizer: Union[str, dict],
                 model: object,
                 learning_rate_scheduler: dict,
                 performance_metrics: Union[None, dict]):
        super().__init__()
        
        self.loss = loss
        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.model = model
        self.performance_metrics = performance_metrics
        
    def __get_loss_function(self, loss):
        """Helper method to retrieve user's choice of loss function."""
        # todo: what if user needs to provide more info when calling loss?
        return getattr(F, loss)
    
    def __get_optim(self, optimizer):
        """Helper method to retrieve user's choice of optimizer."""
        # todo: use kwargs for variable user parameters
        return getattr(torch.optim, optimizer)
    
    def __get_lr_scheduler(self, scheduler):
        """Helper method to retrieve user's choice of learning rate scheduler."""
        return getattr(torch.optim.lr_scheduler, scheduler)
    
    def __get_performance_metric(self, metric):
        """Helper method to retrieve user's choice of performance metric."""
        return getattr(torchmetrics.functional, metric)
        
    def training_step(self, batch, batch_idx):
        
        # to add:
        # > performence metrics also logged in the same way.
        
        metrics = {} # metrics to be logged
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        
        # compute loss; depends on whether user gave kwargs
        if self.loss:
            if isinstance(self.loss, dict):
                for loss_metric in self.loss.keys():
                    loss_kwargs = self.loss[loss_metric]
                    loss = self.__get_loss_function(loss_metric)(y_pred, y, **loss_kwargs)
                    metrics['train_loss_' + loss_metric] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['train_loss_' + self.loss] = loss
        
        if self.performance_metrics:
            if isinstance(self.performance_metrics, dict):
                for p_metric in self.performance_metrics.keys():
                    perf_kwargs = self.performance_metrics[p_metric]
                    metrics['train_perf_' + p_metric] = self.__get_performance_metric(p_metric)(y_pred, y, **perf_kwargs)
            elif isinstance(self.performance_metrics, str): # only a metric (string) is given, no kwargs
                metrics['train_perf_' + self.performance_metrics] = self.__get_performance_metric(self.performance_metrics)(y_pred, y)
        
        # logs every epoch: the loss and performance is accumulated and averaged over the epoch
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        metrics = {}
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        
        # compute loss; depends on whether user gave kwargs
        if self.loss:
            if isinstance(self.loss, dict):
                for loss_metric in self.loss.keys():
                    loss_kwargs = self.loss[loss_metric]
                    loss = self.__get_loss_function(loss_metric)(y_pred, y, **loss_kwargs)
                    metrics['val_loss_' + loss_metric] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['val_loss_' + self.loss] = loss
        
        if self.performance_metrics:
            if isinstance(self.performance_metrics, dict):
                for p_metric in self.performance_metrics.keys():
                    perf_kwargs = self.performance_metrics[p_metric]
                    metrics['val_perf_' + p_metric] = self.__get_performance_metric(p_metric)(y_pred, y, **perf_kwargs)
            elif isinstance(self.performance_metrics, str): # only a metric (string) is given, no kwargs
                metrics['val_perf_' + self.performance_metrics] = self.__get_performance_metric(self.performance_metrics)(y_pred, y)
        
        # logs every epoch: the loss and performance is accumulated and averaged over the epoch
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
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
        if self.optimizer:
            if isinstance(self.optimizer, dict):
                # optimizer has kwargs
                for optim in self.optimizer.keys():
                    opt_kwargs = self.optimizer[optim]
                    optimizer = self.__get_optim(optim)(self.model.parameters(), lr=self.lr, **opt_kwargs)
            elif isinstance(self.optimizer, str):
                # optimizer has no kwargs
                optimizer = self.__get_optim(self.optimizer)(self.model.parameters(), lr=self.lr)
        
        # get user's learning rate scheduler
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, dict):
                # lr schedular has kwargs
                lr_dict = {}
                for sched in self.lr_scheduler.keys():
                        sched_kwargs = self.lr_scheduler[sched]
                        if 'monitor' in sched_kwargs:
                            monitor = sched_kwargs.pop('monitor')
                            lr_dict['monitor'] = monitor
                        scheduler = self.__get_lr_scheduler(sched)(optimizer, **sched_kwargs)                    
                        lr_dict['scheduler'] = scheduler
                return {"optimizer": optimizer, "lr_scheduler": lr_dict}
            elif isinstance(self.lr_scheduler, str):
                # lr scheduler has no kwargs
                scheduler = self.__get_lr_scheduler(self.lr_scheduler)(optimizer)
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            # no scheduler
            return {"optimizer": optimizer}
                    

class KITrainer():
    
    def __init__(self,
                 train_device: str,
                 loss_fn: Union[str, dict],
                 optim: Union[str, dict],
                 max_epochs: int,
                 early_stopping: bool,
                 learning_rate: float,
                 loaders: tuple,
                 model: object,
                 learning_rate_scheduler: dict = {},
                 performance_metrics: Union[None, dict] = None):
        
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
        self.performance_metrics = performance_metrics # dict: choice of metric and any needed kwargs
        
        # instance of model class
        self.model = model
        
        # dataloaders from data module
        self.train_dataloader = loaders[0] # this expects an ordering -> can I do this without assuming that
        self.val_dataloader = loaders[1]
        
        # todo: perform checks on above vals
        
        # initialize PL object
        # todos: move model initialization to seperate method? Should all these be global vars if it stays in __init__?
        self.lit_model = PLModel(loss=self.loss_fn,
                                 optimizer=self.optim, 
                                 model=self.model, 
                                 learning_rate=self.learning_rate,
                                 learning_rate_scheduler=self.learning_rate_scheduler,
                                 performance_metrics=self.performance_metrics)
        
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
    
        
    




