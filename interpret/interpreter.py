from __future__ import annotations

__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the Knowit interpreter module.'

"""
---------------
KIInterpreter
---------------

The ``KIInterpreter'' class is the parent (root) class for the interpreter module.

``KIInterpreter'' is provided the following arguments:
    - model (class)             : the Pytorch model architecture defined in ./archs
    - model_params (dict)       : the dictionary that contains the models init parameters
    - path_to_ckpt (str)        : the path to a trained model's checkpoint file.
    - datamodule (Knowit obj)   : the Knowit datamodule for the experiment
    
The function of the ``KIInterpreter'' class is to store the datamodule and initialize the Pytorch model 
for use by its descendant classes. As such, it is a direct link to Knowit's other modules. It is agnostic 
to the user's choice of interpretability method.
 
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import archs

import torch

from helpers.logger import get_logger

logger = get_logger()

class KIInterpreter():
    
    def __init__(self,
                 model: archs.UserModel.Model,
                 model_params: dict,
                 datamodule: object,
                 path_to_checkpoint: str) -> None:
        
        self.model = self._load_model_from_ckpt(model=model, 
                                                model_params=model_params, 
                                                ckpt_path=path_to_checkpoint)    
        self.datamodule = datamodule
        
    def _load_model_from_ckpt(self, 
                              model: archs.UserModel.Model, 
                              model_params: dict, 
                              ckpt_path: str) -> archs.UserModel.Model:
        
        """
        Initializes a Pytorch model from its state dictionary.
        
        Args:
        model: Class                        A class that specifies the model architecture. Note, 
                                            this is not a class instance.
        model_params: dict                  Any hyperparameters required to initialize the model (such 
                                            as input dimension, layer width, etc) and the type of task 
                                            (regression or classification).
        ckpt_path: str                      The path to the checkpoint file (path should include the 
                                            checkpoint file name). 

        Returns:
            Pytorch model                   A Pytorch model initialized with the weights from a checkpoint
                                            file.
        """
        
        logger.info("Initializing Pytorch model using checkpoint file.")
        
        # init Pytorch model with user params
        model = model(**model_params)
        
        # obtain model weights from ckpt
        ckpt = torch.load(f=ckpt_path)
        model_weights = ckpt['state_dict']
        
        # For each key, PL saves it as "model.model." instead of "model." as expected by Pytorch.
        # The below method is implemented in PL's own example, see:
        # https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        for key in list(model_weights):
            model_weights[key.replace("model.", "", 1)] = model_weights.pop(key)

        # load model weights and set to eval mode
        model.load_state_dict(model_weights)
        model.eval()
        
        return model
    
        
        
        
        
        