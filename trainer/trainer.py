__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the trainer module.'

"""
---------------
KITrainer
---------------

The ``KITrainer'' is a wrapper class that, given a user's parameters, trains a model (defined in 
the ./archs folder) using Pytorch Lightning's Trainer module.

To use Pytorch Lightning's Trainer, a Pytorch Lightning model needs to be provided. A Pytorch Lightning 
model is given in the class ``PLModel'' with several required parameters from the user. The model parameter 
is a Pytorch model class (not an object) and is instantiated within ``PLModel''.

``KITrainer'' will pass the user's parameters appropriately to ``PLModel'' and Pytorch Lightning's Trainer.

To instantiate ``KITrainer'', there are three possible states:
    - State 1: Train a new model from scratch.
    - State 2: Continue training an existing model from checkpoint.
    - State 3: Load a trained model and evaluate it on a eval set.

State 1: Instantiate KITrainer.
>>>> Compulsory User Parameters <<<<
 - experiment_name: str.                                The name of the experiment.
 - train_device: str.                                   The choice of training device ('cpu', 'gpu', etc).
 - loss_fn: str or dict.                                The choice of loss function (see Pytorch's torch.nn.functional documentation)
 - optim: str or dict:                                  The choice of optimizer (see Pytorch's torch.nn.optim documentation)
 - max_epochs: int.                                     The number of epochs that the model should train on.
 - learning_rate: float.                                The learning rate that the chosen optimizer should use.
 - model: Class.                                        The Pytorch model architecture define by the user in Knowits ./archs subdir.
 - model_params: dict.                                  A dictionary with values needed to init the above Pytorch model.
 
>>>> Optional User Parameters <<<<
 - learning_rate_scheduler: dict. Default: {}.          A dictionary that specifies the learning rate scheduler 
                                                            and any needed kwargs.
 - performance_metrics: None or dict. Default: None.    Specifies any performance metrics on the validation 
                                                            set during training.
 - early_stopping: bool or dict. Default: False.        Specifies early stopping conditions.
 - gradient_clip_val: float: Default: 0.0.              Clips exploding gradients according to the chosen 
    gradient_clip_algorithm.
 - gradient_clip_algorithm: str. Default: 'norm'.       Specifies how the gradient_clip_val should be applied.
 - set_seed: int or bool. Default: False.               A global seed applied by Pytorch Lightning for reproducibility.
 - deterministic: bool, str, or None. Default: None.    Pytorch Lightning attempts to further reduce randomness 
                                                            during training. This may incur a performance hit.
 - safe_mode: bool. Default: False.                     If set to True, aborts the model training if the experiment name already 
                                                            exists in the user's project output folder.

NOTE!
The loss functions needs to be used exactly as in Pytorch's torch.nn.functional library. The performance metrics needs to be used 
exactly as in Pytorchmetrics torchmetrics.functional. Note that both methods use the functional libraries (not the modular analogs).


To train the model, the user calls the ".fit_model" method. The method must be provided a tuple consisting of the train data 
loader and the validation data loader (in this order).

State 2: Instantiate KITrainer using "resume_from_ckpt" method.
>>>> Compulsory User Parameters <<<<
 - experiment_name: str.                                The name of the experiment.
 - path_to_checkpoint: str.                             The path to the pretrained model's checkpoint.
 - loss_fn: str or dict.                                The choice of loss function (see Pytorch's torch.nn.functional documentation)
 - optim: str or dict:                                  The choice of optimizer (see Pytorch's torch.nn.optim documentation)
 - max_epochs: int.                                     The number of epochs that the model should train on.
 - learning_rate: float.                                The learning rate that the chosen optimizer should use.
 - model: Class.                                        The Pytorch model architecture define by the user in Knowits ./archs subdir.
 - model_params: dict.                                  A dictionary with values needed to init the above Pytorch model.
 
>>>> Optional User Parameters <<<<
 - set_seed: int. Default: None.                        The global seed set by Pytorch Lightning. The seed should be the same as used to train the 
                                                            checkpoint model.

State 3: Instantiate Trainer using "eval_from_ckpt" method.
>>>> Compulsory User Parameters <<<<
 - experiment_name: str.                                The name of the experiment.
 - path_to_checkpoint: str.                             The path to the pretrained model's checkpoint.

Checkpointing: Once a model has been trained, the best model checkpoint is saved to the user's project output folder under the
name of the experiment.

Testing: The model can be tested on an eval set using the "evaluate_model" method on an appropriately instantiated KITrainer.


------------
PLModel
------------

``PLModel'' is a class required by Pytorch Lightning's Trainer. For more information, see Pytorch Lightning's documentation.

The following parameters needs to be provided by the user:
 - loss: str or dict.                                   The loss function to be used for training.
 - learning_rate: float.                                The learning rate that is to be used for training.
 - optimizer: str or dict.                              The choice of optimizer that Pytorch Lightning needs to use.
 - learning_rate_scheduler: dict.                       The choice of learning rate scheduler that Pytorch Lightning needs to use.
 - performance_metrics: None or dict.                   The performance metrics to be computed on the train and validation sets.
 - model: Class.                                        A Pytorch model class (not an object). This is defined in ./archs
 - model: dict.                                         The parameters needed to instantiate the Pytorch model.
 
"""

#   todo:
#   > matmul_precision: I think pl is sensitive towards this?
#   > change imports to from
#   > refactor some of the code (see the regular blocks of code that unpacks user kwargs)
#   > (fix) currently, checkpoint dir is being created even if training loops are not completed, resulting in empty folders

# Notes:
# > for eval log, want corresponding best model's results and epoch
# > add option to mute logging during tuner
# > Output folder is experiment specific: Trainer will be given experiment subfolder. Ckpts and logs for that experiment goes in here.
# > Metadata? What settings did trainer use to create that ckpt that was saved.
# 	>> Also the model performance


import os
from typing import Any, Dict, Union, Literal
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
from lightning.pytorch import seed_everything

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
                 performance_metrics: Union[None, dict],
                 model: type,
                 model_params: dict):
        super().__init__()
        
        self.loss = loss
        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics
        
        self.model = self._build_model(model, model_params)
        
        #self.save_hyperparameters(ignore=["model"], logger=False)
        self.save_hyperparameters()
        
    def _build_model(self, model, model_params):
        """Instantiates a Pytorch model with the given model parameters

        Args:
            model (class): _description_
            model_params (dict): _description_

        Returns:
            object: Pytorch model 
        """
        return model(**model_params)
        
    def __get_loss_function(self, loss):
        """A helper method to retrieve the user's choice of loss function.

        Args:
            loss (str): The loss function as specified in torch.nn.functional.

        Returns:
            object: Pytorch loss function.
            
        """
        
        return getattr(F, loss)
    
    def __get_optim(self, optimizer):
        """A helper method to retrieve the user's choice of optimizer.

        Args:
            optimizer (str): The loss function as specified in torch.optim.

        Returns:
            object: Pytorch optimizer function.
            
        """
        
        return getattr(torch.optim, optimizer)
    
    def __get_lr_scheduler(self, scheduler):
        """A helper method to retrieve the user's choice of learning rate scheduler.

        Args:
            scheduler (str): The loss function as specified in torch.nn.functional.

        Returns:
            object: Pytorch learning scheduler.
            
        """
        
        return getattr(torch.optim.lr_scheduler, scheduler)
    
    def __get_performance_metric(self, metric):
        """A helper method to retrieve the user's choice of performance metric.

        Args:
            metric (str): The metric function as specified in torchmetrics.functional.

        Returns:
            object: Torchmetrics function.
            
        """
        
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
                    metrics['eval_loss'] = loss
            elif isinstance(self.loss, str): # only a metric (string) is given, no kwargs
                loss = self.__get_loss_function(self.loss)(y_pred, y)
                metrics['eval_loss'] = loss
                
        if self.performance_metrics:
            if isinstance(self.performance_metrics, dict):
                for p_metric in self.performance_metrics.keys():
                    perf_kwargs = self.performance_metrics[p_metric]
                    metrics['eval_perf_' + p_metric] = self.__get_performance_metric(p_metric)(y_pred, y, **perf_kwargs)
            elif isinstance(self.performance_metrics, str): # only a metric (string) is given, no kwargs
                metrics['eval_perf_' + self.performance_metrics] = self.__get_performance_metric(self.performance_metrics)(y_pred, y)

            self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            
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
           
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        print(checkpoint.keys())

class KITrainer():
    
    """ A wrapper class that handles user parameters
    
    Args (compulsary)
    - experiment_name: str.                                The name of the experiment.
    - train_device: str.                                   The choice of training device ('cpu', 'gpu', etc).
    - loss_fn: str or dict.                                The choice of loss function (see Pytorch's torch.nn.functional documentation)
    - optim: str or dict:                                  The choice of optimizer (see Pytorch's torch.nn.optim documentation)
    - max_epochs: int.                                     The number of epochs that the model should train on.
    - learning_rate: float.                                The learning rate that the chosen optimizer should use.
    - model: Class.                                        The Pytorch model architecture define by the user in Knowits ./archs subdir.
    - model_params: dict.                                  A dictionary with values needed to init the above Pytorch model.
 
    Args (optional)
    - learning_rate_scheduler: dict. Default: {}.          A dictionary that specifies the learning rate scheduler 
                                                            and any needed kwargs.
    - performance_metrics: None or dict. Default: None.    Specifies any performance metrics on the validation 
                                                            set during training.
    - early_stopping: bool or dict. Default: False.        Specifies early stopping conditions.
    - gradient_clip_val: float: Default: 0.0.              Clips exploding gradients according to the chosen 
                                                            gradient_clip_algorithm.
    - gradient_clip_algorithm: str. Default: 'norm'.       Specifies how the gradient_clip_val should be applied.
    - set_seed: int or bool. Default: False.               A global seed applied by Pytorch Lightning for reproducibility.
    - deterministic: bool, str, or None. Default: None.    Pytorch Lightning attempts to further reduce randomness 
                                                            during training. This may incur a performance hit.
    - safe_mode: bool. Default: False.                     If set to True, aborts the model training if the experiment name already 
                                                            exists in the user's project output folder.
    """
    
    def __init__(self,
                 experiment_name: str,
                 train_device: str,
                 loss_fn: Union[str, dict],
                 optim: Union[str, dict],
                 max_epochs: int,
                 learning_rate: float,
                 model: type,
                 model_params: dict,
                 learning_rate_scheduler: dict = {},
                 performance_metrics: Union[None, dict] = None,
                 early_stopping: Union[bool, dict] = False,
                 gradient_clip_val: float=0.0,
                 gradient_clip_algorithm: str='norm',
                 train_flag: bool = True,
                 from_ckpt_flag: bool = False,
                 set_seed: Union[int, bool] = False,
                 deterministic: Union[bool, Literal['warn'], None] = None,
                 path_to_checkpoint: Union[str, None] = None,
                 safe_mode: bool = False):
        
        # create an experiment directory in the user's project folder
        self.experiment_dir = self.__make_experiment_dir(name=experiment_name, safe_mode=safe_mode)
        
        # internal flags used by class to determine which state the user is instantiating KITrainer
        self.train_flag = train_flag
        self.from_ckpt_flag = from_ckpt_flag
        
        # save global seed
        self.set_seed = set_seed
        
        # device to use
        self.train_device = train_device
        self.deterministic = deterministic
        
        # Pytorch model class and parameters
        self.model = model
        self.model_params = model_params
        
        # model hyperparameters
        self.loss_fn = loss_fn
        self.optim = optim
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler 
        self.performance_metrics = performance_metrics 
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        
        # user is training from scratch
        if train_flag == True and from_ckpt_flag == False:
            
            # construct trainer
            self.trainer = self.__build_PL_trainer()
            
            # build a PL model from arguments (untrained)
            self.lit_model = PLModel(loss=self.loss_fn,
                                 optimizer=self.optim, 
                                 model=self.model,
                                 model_params=self.model_params, 
                                 learning_rate=self.learning_rate,
                                 learning_rate_scheduler=self.learning_rate_scheduler,
                                 performance_metrics=self.performance_metrics)
        
        # user is continuing model training from saved checkpoint
        elif train_flag == True and from_ckpt_flag == True:
            
            self.path_to_checkpoint = path_to_checkpoint
            
            # construct trainer
            self.lit_model = PLModel.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
            self.trainer = self.__build_PL_trainer()
            
        # user is evaluating model from saved checkpoint
        elif train_flag == False and from_ckpt_flag == True:
            self.lit_model = PLModel.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
            self.trainer = pl.Trainer()
        
    @classmethod
    def eval_from_ckpt(cls, experiment_name, path_to_checkpoint):        
        """A constructor initializing KITrainer in state 3 (model evaluation only).

        Args:
            experiment_name (str): Experiment name
            path_to_checkpoint (str): The path to the checkpoint file.
            
        """
        kitrainer = cls(
            experiment_name=experiment_name,
            train_device=None,
            loss_fn=None,
            optim=True,
            max_epochs=None,
            learning_rate=None,
            model=None,
            model_params=None,
            train_flag=False,
            from_ckpt_flag=True,
            path_to_checkpoint=path_to_checkpoint
        )
        
        return kitrainer
    
    @classmethod
    def resume_from_ckpt(cls, experiment_name, max_epochs, path_to_checkpoint, set_seed=None):
        """A constructor initializing KITrainer in state 2 (resume model training from checkpoint).

        Args:
            experiment_name (str)       : Experiment name
            path_to_checkpoint (str)    : The path to the checkpoint file.
            max_epochs (int)            : The number of further epochs to train the model. If the pretrained model
                                            was trained for x epochs and the user wants to train for a further y epochs,
                                            then this should be set to max_epochs = x+y.
            set_seed   (None or int)    : The seed value that was used for the pretrained model.
            
        """
        
        kitrainer = cls(
            experiment_name=experiment_name,
            train_device=None,
            loss_fn=None,
            optim=True,
            max_epochs=max_epochs,
            learning_rate=None,
            model=None,
            model_params=None,
            train_flag=True,
            from_ckpt_flag=True,
            path_to_checkpoint=path_to_checkpoint,
            set_seed=set_seed
        )
        
        return kitrainer
        
        
    def fit_model(self, dataloaders):
        """User Pytorch Lightning to fit the model to the train data

        Args:
            dataloaders (tuple): The train dataloader and validation dataloader. The ordering of the tuple is (train, val).
            
        """
        
        train_dataloader = dataloaders[0]
        val_dataloader = dataloaders[1]
        
        # fit trainer object to data
        if self.train_flag == True and self.from_ckpt_flag == False:
            self.trainer.fit(model=self.lit_model,
                            train_dataloaders=train_dataloader,
                            val_dataloaders=val_dataloader)
        elif self.train_flag == True and self.from_ckpt_flag == True:
            logger.info("Resuming model training from checkpoint.")
            self.trainer.fit(model=self.lit_model,
                            train_dataloaders=train_dataloader,
                            val_dataloaders=val_dataloader,
                            ckpt_path=self.path_to_checkpoint)
            
    def evaluate_model(self, eval_dataloader):
        """Evaluates the model's performance on a evaluation set.

        Args:
            eval_dataloader (Pytorch dataloader)    : The evaluation dataloader. 
        """
        
        if self.train_flag:
            logger.info("Testing model on the current training run's best checkpoint.")
            self.trainer.test(ckpt_path='best', dataloaders=eval_dataloader)
        else:
            logger.info("Testing on model loaded from checkpoint.")
            self.trainer.test(model=self.lit_model, dataloaders=eval_dataloader)
            
        
        
    def __build_PL_trainer(self):
        """Calls Pytorch Lightning's trainer using the user's parameters.
        
        """
        
        # save best model state
        ckpt_path, ckpt_callback = self.__save_model_state()
        
        # training logger - save results in current model's folder
        #csv_logger = pl_loggers.CSVLogger(save_dir=self.experiment_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=ckpt_path)
        
        # Early stopping
        try:
            early_stopping = EarlyStopping(**self.early_stopping[True])
            logger.info('Early stopping is enabled.')
        except:
            logger.info('Early stopping is not enabled. If Early Stopping should be enabled, it must be passed as a dict with kwargs.')
            early_stopping = None
        
        callbacks = [c for c in [ckpt_callback, early_stopping] if c != None]
        
        # set seed
        if self.set_seed:
            seed_everything(self.set_seed, workers=True)            
        
        # Pytorch Lightning trainer object
        if self.train_flag == True and self.from_ckpt_flag == False:
            trainer = pl.Trainer(max_epochs=self.max_epochs,
                             accelerator=self.train_device, 
                             logger=csv_logger,
                             devices='auto', # what other options here?
                             callbacks=callbacks,
                             detect_anomaly=True,
                             #default_root_dir=self.experiment_dir,
                             default_root_dir=ckpt_path,
                             deterministic=self.deterministic,
                             gradient_clip_val=self.gradient_clip_val,
                             gradient_clip_algorithm=self.gradient_clip_algorithm
                             )
        elif self.train_flag == True and self.from_ckpt_flag == True:
            trainer = pl.Trainer(max_epochs=self.max_epochs,
                             #default_root_dir=self.experiment_dir,
                             default_root_dir=ckpt_path,
                             callbacks=callbacks,
                             detect_anomaly=True,
                             logger=csv_logger
                             )
        
        return trainer
        
    def __save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.
        
        """
        
        model_dir = self.experiment_dir + '/models'
        
        # best models are saved to a folder named as a datetime string
        file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ckpt_path = os.path.join(model_dir, 'Model_' + file_name)
        
        return ckpt_path, ModelCheckpoint(dirpath=ckpt_path,
                                        monitor='val_loss',
                                        filename='bestmodel-{epoch}-{val_loss:.2f} ' + file_name)
        
        
    def __make_experiment_dir(self, name, safe_mode):
        """Given a user's name for the experiment, creates a directory in the user's project output directory.

        Args:
            name (str): The name of the experiment.
            safe_mode (bool): If the experiment name already exists in the directory and safe_mode = True, abort 
                                the experiment.

        Returns:
            str: The path to the experiment's directory.
        """
        if name in os.listdir(env_user.project_dir):
            if safe_mode == False:
                logger.warning("A folder with the same experiment name already exists. Safe mode is set to False.")
            else:
                logger.info("A folder with the same experiment name already exists. Safe mode is set to True.")
                logger.info("Aborting...")
                exit()
        
        experiment_dir = os.path.join(env_user.project_dir, name)
        os.path.join(experiment_dir, 'models')
                    
        return experiment_dir
        
    def display_results(self, path_to_ckpt):
        ckpt = torch.load(path_to_ckpt)
        
        best_epoch = ckpt['Epoch']
        best_val_score = ckpt['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_score'].item()
        best_model_dir = ckpt['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]['best_model_path']
        
        print("Best Model Location: ", best_model_dir)
        print(f"Best Model at Epoch: ", best_epoch)
        print(f"Best Model's Validation Score: ", best_val_score)
        
        
        
        
        
        
        
        
    
        
    




