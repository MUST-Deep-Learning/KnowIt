__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the trainer module.'

#   todo:
#   > Build a class that accepts data structure from data module
#       1.1) User inputs: optimizer, loss, max_epochs. From the /model folder, object instance of model is passed as parameter to Trainer class
#       1.2) Use PL to build trainer.
#   > Logger (tensorboard, wandb, etc)?
#   > matmul_precision: I think pl is sensitive towards this?
#   > change imports to from
#   > save logging results as csv
#   > refactor some of the code (see the regular blocks of code that unpacks user kwargs)
#   > currently using metrics from torch.nn.functional and torchmetrics. Should pick one.
#       >> torchmetrics does not seem to have cross entropy.
#   > add performance method to run on test dataloader
#   > run eval from current training session or from checkpoint (complete classmethod)

import os
from typing import Union
from datetime import datetime

from env import env_user
from helpers.logger import get_logger

logger = get_logger()

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

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
        
        self.save_hyperparameters()
        
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
                    metrics['train_loss'] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['train_loss'] = loss
        
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
                    metrics['val_loss'] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['val_loss'] = loss
        
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
                    metrics['test_loss'] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['test_loss'] = loss
                
        if self.performance_metrics:
            if isinstance(self.performance_metrics, dict):
                for p_metric in self.performance_metrics.keys():
                    perf_kwargs = self.performance_metrics[p_metric]
                    metrics['test_perf_' + p_metric] = self.__get_performance_metric(p_metric)(y_pred, y, **perf_kwargs)
            elif isinstance(self.performance_metrics, str): # only a metric (string) is given, no kwargs
                metrics['test_perf_' + self.performance_metrics] = self.__get_performance_metric(self.performance_metrics)(y_pred, y)

            self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
            
        return metrics
        
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
                 learning_rate: float,
                 loaders: tuple,
                 model: object,
                 learning_rate_scheduler: dict = {},
                 performance_metrics: Union[None, dict] = None,
                 early_stopping: Union[bool, dict] = False):
        
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
        
        # construct trainer
        self.trainer = self._build_PL_trainer()
        
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
        
    @classmethod
    def construct_from_ckpt(cls, path_to_checkpoint):
        pass
        
        
    def fit_model(self):
        
        # fit trainer object to data
        self.trainer.fit(model=self.lit_model,
                    train_dataloaders=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        
        
    def test_model(self, test_dataloader, from_checkpoint=None):
        # todo: choice to either load from checkpoint or from current completed training loop.
        
        try:
            logger.info("Testing model on the current training run's best checkpoint.")
            self.trainer.test(ckpt_path='best', dataloaders=test_dataloader)
        except:
            logger.info("Initialising model from checkpoint.")
            model = PLModel.load_from_checkpoint(from_checkpoint)
            logger.info("Testing model on checkpoint.")
            trainer = pl.Trainer(model=model)
            trainer.test(ckpt_path=from_checkpoint, dataloaders=test_dataloader)
        
        
    def _build_PL_trainer(self):
        
        # training logger
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=env_user.project_dir)
        #csv_logger = pl_loggers.CSVLogger(save_dir=env_user.project_dir)
        
        # save best model state
        ckpt_callback = self.__save_model_state()
        
        # Early stopping
        try:
            early_stopping = EarlyStopping(**self.early_stopping[True])
            logger.info('Early stopping is enabled.')
        except:
            logger.info('Early stopping is not enabled. If Early Stopping should be enabled, it must be passed as a dict with kwargs.')
            early_stopping = None
        
        callbacks = [c for c in [ckpt_callback, early_stopping] if c != None]
        
        # Pytorch Lightning trainer object
        trainer = pl.Trainer(max_epochs=self.max_epochs,
                             accelerator=self.train_device, 
                             logger=tb_logger,
                             devices='auto', # what other options here?
                             callbacks=callbacks,
                             detect_anomaly=True,
                             default_root_dir=env_user.project_dir
                             )
        
        return trainer
        
    def __save_model_state(self):
        
        project_path = env_user.checkpoints_dir
        
        # best models are saved to a folder named as a datetime string
        file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = os.path.join(project_path, 'Checkpoint_' + file_name)
        
        try:
            os.mkdir(path)
        except OSError as error:
            print(error) # folder already exists
        
        return ModelCheckpoint(dirpath=path,
                               monitor='val_loss',
                               filename='bestmodel-{epoch}-{val_loss:.2f} ' + file_name)
    
        
    




