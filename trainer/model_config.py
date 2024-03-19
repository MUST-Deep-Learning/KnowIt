__author__ = 'randlerabe@gmail.com'
__description__ = 'Constructs a Pytorch Lightning model class.'

from typing import Union, Type, Tuple, Optional

import pytorch_lightning as pl
from torch import tensor

from helpers.logger import get_logger
from helpers.fetch_torch_mods import (get_loss_function,
                                      get_lr_scheduler,
                                      get_optim,
                                      get_performance_metric)

logger = get_logger()


class PLModel(pl.LightningModule):
    
    """A Pytorch Lightning model that defines the training, validation, and test steps over a batch. 
    The optimizer configuration is also set inside this class. This is required for Pytorch Lightning's 
    Trainer.

    Args:
        loss (str, dict)                :   Loss function as given in torch.nn.functional
        learning_rate (float)           :   Learning rate
        optimizer (str, dict)           :   The optimizer to be used for training as given in torch.optim. Additional kwargs can 
                                            be provided as a dict.
        learning_rate_scheduler (dict)  :   The choice of learning rate scheduler as given in torch.optim.lr_scheduler. Additional 
                                            kwargs can be provided as a dict.
        performance_metric (dict)       :   The choice of performance metrics as given in torchmetrics.functional.
        model (class)                   :   A Pytorch model architecture defined in ./archs. Note that this is a class, not an object.
        model_params (dict)             :   The parameters needed to instantiate the above Pytorch model.
        
    """
    
    def __init__(self, 
                 loss: Union[str, dict],
                 learning_rate: float, 
                 optimizer: Union[str, dict],
                 learning_rate_scheduler: dict,
                 performance_metrics: Optional[dict],
                 model: type,
                 model_params: dict) -> None:
        super().__init__()
        
        self.loss = loss
        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics
        
        self.model = self._build_model(model, model_params)
        
        self.save_hyperparameters()
        
        
    def _build_model(self, model: Type, model_params: dict) -> Type:
        
        """Instantiates a Pytorch model with the given model parameters

        Args:
            model (class)           : A Pytorch model that provides architecture and forward method.
            model_params (dict)     : A dictionary containing the parameters used to init the model.

        Returns:
            (type)                  : Pytorch model 
        """
        
        return model(**model_params)
        
    
    def _compute_loss(self, 
                      log_metrics: dict, 
                      y: Union[int, float, tensor], 
                      y_pred: Union[int, float, tensor], 
                      loss_label: str) -> Tuple[Union[int, float, tensor], dict]:
        
        """
        Given a target y and the model's prediction y_pred, computes the loss between y_pred and y.

        Args:
            log_metrics (dict)                  : The dictionary that logs the metrics.
            y (Union[int, float, tensor])       : The target from a set of training pairs.
            y_pred (Union[int, float, tensor])  : The model's prediction.
            loss_label (str)                    : Name to be used for labeling purposes.

        Returns:
            (Tuple)                             : The computed loss between y and y_pred and the 
                                                    dictionary that logs the loss.

        """
        
        if isinstance(self.loss, dict):
            for loss_metric in self.loss.keys():
                loss_kwargs = self.loss[loss_metric]
                loss = get_loss_function(loss_metric)(y_pred, y, **loss_kwargs)
                log_metrics[loss_label] = loss
        elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
            loss = get_loss_function(self.loss)(y_pred, y)
            log_metrics[loss_label] = loss
            
        return loss, log_metrics
    
    
    def _compute_performance(self, 
                             log_metrics: dict, 
                             y: Union[int, float, tensor], 
                             y_pred: Union[int, float, tensor], 
                             perf_label: str):
        
        """
          Given a target y and the model's prediction y_pred, computes the performance score between y_pred and y.

        Args:
            log_metrics (dict)                  : The dictionary that logs the metrics.
            y (Union[int, float, tensor])       : The target from a set of training pairs.
            y_pred (Union[int, float, tensor])  : The model's prediction.
            perf_label (str)                    : Name to be used for labeling purposes.

        Returns:
            (Tuple)                             : The computed score between y and y_pred and the dictionary 
                                                  that logs the performance score.
        """
        
        if isinstance(self.performance_metrics, dict):
            for p_metric in self.performance_metrics.keys():
                perf_kwargs = self.performance_metrics[p_metric]
                log_metrics[perf_label + p_metric] = get_performance_metric(p_metric)(y_pred, y, **perf_kwargs)
        elif isinstance(self.performance_metrics, str): # only a metric (string) is given, no kwargs
            log_metrics[perf_label + self.performance_metrics] = get_performance_metric(self.performance_metrics)(y_pred, y)
            
        return log_metrics
    
        
    def training_step(self, batch, batch_idx):
        
        metrics = {} # metrics to be logged
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        
        # compute loss; depends on whether user gave kwargs
        loss, metrics = self._compute_loss(log_metrics=metrics, y=y, y_pred=y_pred, loss_label='train_loss')
        # compute performance; depends on whether user gave kwargs
        metrics = self._compute_performance(log_metrics=metrics, y=y, y_pred=y_pred, perf_label='train_perf_')
        
        
        # logs every epoch: the loss and performance is accumulated and averaged over the epoch
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        metrics = {} # metrics to be logged
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        
        # compute loss; depends on whether user gave kwargs
        loss, metrics = self._compute_loss(log_metrics=metrics, y=y, y_pred=y_pred, loss_label='valid_loss')
        # compute performance; depends on whether user gave kwargs
        metrics = self._compute_performance(log_metrics=metrics, y=y, y_pred=y_pred, perf_label='valid_perf_')
        
        # logs every epoch: the loss and performance is accumulated and averaged over the epoch
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss    
    
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        
        metrics = {} # metrics to be logged
        
        if dataloader_idx == 0:
            current_loader = "result_train_"
        elif dataloader_idx == 1:
            current_loader = "result_valid_"
        elif dataloader_idx == 2:
            current_loader = "result_eval_"
        
        x = batch['x']
        y = batch['y']
        
        y_pred = self.model.forward(x)
        
        # compute loss; depends on whether user gave kwargs
        _, metrics = self._compute_loss(log_metrics=metrics, y=y, y_pred=y_pred, loss_label=current_loader + 'loss')
        # compute performance; depends on whether user gave kwargs
        metrics = self._compute_performance(log_metrics=metrics, y=y, y_pred=y_pred, perf_label=current_loader + 'perf_')

        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True, add_dataloader_idx=False)
            
        return metrics
    
        
    def configure_optimizers(self):
        
        # get user's optimizer
        if self.optimizer:
            if isinstance(self.optimizer, dict): # optimizer has kwargs
                for optim in self.optimizer.keys():
                    opt_kwargs = self.optimizer[optim]
                    optimizer = get_optim(optim)(self.model.parameters(), lr=self.lr, **opt_kwargs)
            elif isinstance(self.optimizer, str): # optimizer has no kwargs
                print(self.model.parameters().__doc__)
                optimizer = get_optim(self.optimizer)(self.model.parameters(), lr=self.lr)
        
        # get user's learning rate scheduler
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, dict): # lr schedular has kwargs
                lr_dict = {}
                for sched in self.lr_scheduler.keys():
                        sched_kwargs = self.lr_scheduler[sched]
                        if 'monitor' in sched_kwargs:
                            monitor = sched_kwargs.pop('monitor')
                            lr_dict['monitor'] = monitor
                        scheduler = get_lr_scheduler(sched)(optimizer, **sched_kwargs)                    
                        lr_dict['scheduler'] = scheduler
                return {"optimizer": optimizer, "lr_scheduler": lr_dict}
            elif isinstance(self.lr_scheduler, str): # lr scheduler has no kwargs
                scheduler = get_lr_scheduler(self.lr_scheduler)(optimizer)
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            # no scheduler
            return {"optimizer": optimizer}