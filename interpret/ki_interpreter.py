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

from typing import Type

import torch

class KIInterpreter():
    """
    To fill
    """
    
    def __init__(self,
                 model: Type,
                 model_params: dict,
                 datamodule: object,
                 path_to_checkpoint: str):
        
        self.model = self.__load_model_from_ckpt(model=model, model_params=model_params, ckpt_path=path_to_checkpoint)    
        self.datamodule = datamodule
        
    def __load_model_from_ckpt(self, model, model_params, ckpt_path):
        
        # init Pytorch model with user params
        pt_model = model(**model_params)
        
        # obtain state of model from ckpt
        ckpt = torch.load(f=ckpt_path)
        state_dict = ckpt['state_dict']

        # todo: PL saves keys as "model.model..."; Pytorch expects "model....". Fix.
        # Approach below is temp solution
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]

        # load model and set to eval mode
        pt_model.load_state_dict(state_dict)
        pt_model.eval()
        
        return pt_model
    
        
        
        
        
        