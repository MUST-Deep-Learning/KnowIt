__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the trainer module.'

"""
---------------
Trainer
---------------

The ``Trainer'' is a wrapper class that, given a user's parameters, trains a model (defined in 
the ./archs folder) using Pytorch Lightning's Trainer module.

To use Pytorch Lightning's Trainer, a Pytorch Lightning model needs to be provided. A Pytorch Lightning 
model is given in the class ``PLModel'' with several required parameters from the user. The model parameter 
is a Pytorch model class (not an object) and is instantiated within ``PLModel''.

``Trainer'' will pass the user's parameters appropriately to ``PLModel'' and Pytorch Lightning's Trainer.

To instantiate ``Trainer'', there are three possible states:
    - State 1: Train a new model from scratch.
    - State 2: Continue training an existing model from checkpoint.
    - State 3: Load a trained model and evaluate it on a eval set.

State 1: Instantiate Trainer.
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
 - train_precision: str: Default: '32-true'             Sets the precision to be 
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

State 2: Instantiate Trainer using "resume_from_ckpt" method.
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

Testing: The model can be tested on an eval set using the "evaluate_model" method on an appropriately instantiated Trainer.


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
#   > To check: after training, testing on all three dataloaders gives slight discrepancy between logged train vals
#   > in torch/nn/modules/module.py, there is a method called register_parameters. If I open 
#       the "param" variable in debugger, it says that cuda is False and CPU is True. Is it a bug
#       or is it being set to CPU in Tian's script? Check.
#   > added new parameter: 'train_precision'. Needs to be updated in Knowit




from typing import Union, Literal, Optional

from trainer.model_config import PLModel

from helpers.logger import get_logger

logger = get_logger()

import torch

from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything


class Trainer:
    
    """ A wrapper class that handles user parameters and directs the instructions to Pytorch Lightning.
    
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
                 out_dir: str,
                 train_device: str,
                 loss_fn: Union[str, dict],
                 optim: Union[str, dict],
                 max_epochs: int,
                 learning_rate: float,
                 model: type,
                 model_params: dict,
                 learning_rate_scheduler: dict = {},
                 performance_metrics: Optional[dict] = None,
                 early_stopping: Union[bool, dict] = False,
                 gradient_clip_val: float = 0.0,
                 gradient_clip_algorithm: str = 'norm',
                 train_precision: str = '32-true',
                 train_flag: bool = True,
                 from_ckpt_flag: bool = False,
                 set_seed: Union[int, bool] = False,
                 deterministic: Union[bool, Literal['warn'], None] = None,
                 path_to_checkpoint: Optional[str] = None,
                 return_final: bool = False,
                 mute_logger: bool = False,
                 model_selection_mode: str = 'min'):
        
        # internal flags used by class to determine which state the user is instantiating Trainer
        self.train_flag = train_flag
        self.from_ckpt_flag = from_ckpt_flag
        self.out_dir = out_dir
        
        # turn off logger during hp tuning
        self.mute_logger = mute_logger
        
        # save global seed
        self.set_seed = set_seed
        
        # device to use
        self.train_device = train_device
        if train_device == 'gpu':
            # TODO: Check the effect of this later.
            try:
                torch.set_float32_matmul_precision('high')
            except:
                logger.warning('Tried to utilize Tensor Cores, but failed.')
        #torch.backends.cuda.matmul.allow_tf32 = True
        self.deterministic = deterministic
        self.precision = train_precision
        
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
        
        # misc
        self.return_final = return_final
        self.model_selection_mode = model_selection_mode
        
        # State 1: user is training from scratch
        if train_flag == True and from_ckpt_flag == False:
            
            # construct trainer
            self.trainer = self._build_PL_trainer()
            
            # build a PL model from arguments (untrained)
            self.lit_model = PLModel(loss=self.loss_fn,
                                 optimizer=self.optim, 
                                 model=model,
                                 model_params=model_params, 
                                 learning_rate=self.learning_rate,
                                 learning_rate_scheduler=self.learning_rate_scheduler,
                                 performance_metrics=self.performance_metrics)
        
        # State 2: user is continuing model training from saved checkpoint
        elif train_flag == True and from_ckpt_flag == True:
            
            self.path_to_checkpoint = path_to_checkpoint
            
            # construct trainer
            self.lit_model = PLModel.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
            self.trainer = self._build_PL_trainer()
            
        # State 3: user is evaluating model from saved checkpoint
        elif train_flag == False and from_ckpt_flag == True:
            self.lit_model = PLModel.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
            self.trainer = PLTrainer()
        
    @classmethod
    def eval_from_ckpt(cls, experiment_name, path_to_checkpoint):        
        """A constructor initializing Trainer in state 3 (model evaluation only).

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
    def resume_from_ckpt(cls, experiment_name, max_epochs, path_to_checkpoint, set_seed=None, safe_mode=False):
        """A constructor initializing Trainer in state 2 (resume model training from checkpoint).

        Args:
            experiment_name (str)       : Experiment name
            path_to_checkpoint (str)    : The path to the checkpoint file.
            max_epochs (int)            : The number of further epochs to train the model. If the pretrained model
                                            was trained for x epochs and the user wants to train for a further y epochs,
                                            then this should be set to max_epochs = x+y.
            set_seed (None or int)      : The seed value that was used for the pretrained model.
            safe_mode (bool)            : If set to True, aborts the model training if the experiment name already 
                                            exists in the user's project output folder.
            
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
            set_seed=set_seed,
            safe_mode=safe_mode
        )
        
        return kitrainer
        
        
    def fit_model(self, dataloaders):
        """Uses Pytorch Lightning to fit the model to the train data

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
        
        # the path to the best model ckpt or the last model ckpt
        if self.return_final:
            set_ckpt_path = self.out_dir + '/last.ckpt'
        else:
            set_ckpt_path = 'best'
        
        if self.train_flag:
            logger.info("Testing model on the current training run's best checkpoint.")
            self.trainer.test(ckpt_path=set_ckpt_path, dataloaders=eval_dataloader)
        else:
            logger.info("Testing on model loaded from checkpoint.")
            self.trainer.test(model=self.lit_model, dataloaders=eval_dataloader)
            

    def _build_PL_trainer(self):
        """Calls Pytorch Lightning's trainer using the user's parameters.
        
        """
        
        # save best model state
        #ckpt_path, ckpt_callback = self.__save_model_state()
        
        # training logger - save results in current model's folder
        if self.mute_logger:
            csv_logger = None
            ckpt_path, ckpt_callback = None, None
        else:
            ckpt_path, ckpt_callback = self._save_model_state()
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
            trainer = PLTrainer(max_epochs=self.max_epochs,
                             accelerator=self.train_device, 
                             logger=csv_logger,
                             devices='auto', # what other options here?
                             callbacks=callbacks,
                             detect_anomaly=True,
                             precision=self.precision,
                             default_root_dir=ckpt_path,
                             deterministic=self.deterministic,
                             gradient_clip_val=self.gradient_clip_val,
                             gradient_clip_algorithm=self.gradient_clip_algorithm
                             )
        elif self.train_flag == True and self.from_ckpt_flag == True:
            trainer = PLTrainer(max_epochs=self.max_epochs,
                             #default_root_dir=self.experiment_dir,
                             default_root_dir=ckpt_path,
                             callbacks=callbacks,
                             detect_anomaly=True,
                             logger=csv_logger
                             )
        
        return trainer


    def _save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.
        
        """
        
        # model_dir = os.path.join(self.experiment_dir, 'models')
        #
        # # best models are saved to a folder named as a datetime string
        # file_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # ckpt_path = os.path.join(model_dir, self.model_name + '_' + file_name)
        
        # return ckpt_path, ModelCheckpoint(dirpath=ckpt_path,
        #                                 monitor='valid_loss',
        #                                 filename='bestmodel-{epoch}-{valid_loss:.2f} ' + file_name)

        # determine if last model or best model needs to be returned
        if self.return_final:
            set_top_k = 0
            set_save_last = True
        else:
            set_top_k = 1
            set_save_last = False

        to_monitor = 'valid_loss'
        if self.performance_metrics:
            try:
                met = list(self.performance_metrics.keys())
                met = met[0]
                to_monitor = 'valid_perf_' + met
            except:
                to_monitor = 'valid_perf_' + self.performance_metrics


        
        return self.out_dir, ModelCheckpoint(dirpath=self.out_dir,
                                             monitor=to_monitor,
                                             filename='bestmodel-{epoch}-{' + to_monitor + ':.2f}',
                                             save_top_k=set_top_k,
                                             save_last=set_save_last,
                                             mode=self.model_selection_mode)

        
        
        
        
        
        
        
        
    
        
    




