__author__ = "randlerabe@gmail.com"
__description__ = "Contains the trainer module."

from typing import Optional

from helpers.logger import get_logger
from trainer.trainer_states import TrainNew, ContinueTraining, EvaluateOnly

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
 - seed: int or bool. Default: False.               A global seed applied by Pytorch Lightning for reproducibility.
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
 - seed: int. Default: None.                        The global seed set by Pytorch Lightning. The seed should be the same as used to train the 
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

#   TODO:
#   > To check: after training, testing on all three dataloaders gives slight discrepancy between logged train vals
#   > in torch/nn/modules/module.py, there is a method called register_parameters. If I open
#       the "param" variable in debugger, it says that cuda is False and CPU is True. Is it a bug
#       or is it being set to CPU in Tian's script? Check.
#   > added new parameter: 'train_precision', 'num_devices'. Needs to be updated in Knowit

logger = get_logger()


class KITrainer:
    """A wrapper class that handles user parameters and directs the instructions to Pytorch Lightning.

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
    - seed: int or bool. Default: False.               A global seed applied by Pytorch Lightning for reproducibility.
    - deterministic: bool, str, or None. Default: None.    Pytorch Lightning attempts to further reduce randomness 
                                                            during training. This may incur a performance hit.
    - safe_mode: bool. Default: False.                     If set to True, aborts the model training if the experiment name already
                                                            exists in the user's project output folder.
    """

    _state = None
    
    def __init__(
        self,
        state,
        ckpt_file=None,
        **kwargs,
    ) -> None:
        self._set_state(state=state, base_trainer_kwargs=kwargs, ckpt_file=ckpt_file)
        
    def _set_state(self, state, base_trainer_kwargs, ckpt_file) -> None:
        if ckpt_file:
            self._state = state(**base_trainer_kwargs, ckpt_file=ckpt_file)
            self._state.context = self
        else:
            self._state = state(**base_trainer_kwargs)
            self._state.context = self

    def fit_and_eval(self, dataloaders):
        self._state.fit_model(dataloaders=dataloaders)
        self._state.evaluate_model(dataloaders=dataloaders)
        
    def eval(self, dataloaders):
        self._state.evaluate_model(dataloaders=dataloaders)
