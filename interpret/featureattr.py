__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for performing feature attribution.'

"""
---------------
FeatureAttribution
---------------

The ``FeatureAttribution'' class is a child class which inherits from ``KIInterpreter''.

``KIInterpreter'' is provided the following arguments:
    - model (class)             : the Pytorch model architecture defined in ./archs
    - model_params (dict)       : the dictionary that contains the models init parameters
    - path_to_ckpt (str)        : the path to a trained model's checkpoint file.
    - datamodule (Knowit obj)   : the Knowit datamodule for the experiment.
    - i_data (str)              : the user's choice of dataset to perform feature attribution. Choices: 'train', 'valid', 'eval'
    
The function of the ``FeatureAttribution'' class is to serve the user's choice of feature attribution method (a child class) by extracting 
the necessary information from Knowit's datamodule and returning it in the correct form.
 
"""

from typing import Type

from interpret.interpreter import KIInterpreter

import torch

from helpers.logger import get_logger

logger = get_logger()

class FeatureAttribution(KIInterpreter):
    """
    To fill
    """
    
    def __init__(self, 
                 model: Type, 
                 model_params: dict, 
                 path_to_ckpt: str, 
                 datamodule: object,
                 i_data: str
                 ):
        super().__init__(model=model, 
                         model_params=model_params, 
                         path_to_checkpoint=path_to_ckpt,
                         datamodule=datamodule
                         )
    
    # Todo: for the moment, i_mode is a single point. Needs to handle range
        self.i_data = i_data
    
    def _pred_points_from_datamodule(self, prediction_point_ids, custom_i_data=None):
        
        # Method handles the case where a range of points is chosen by the user.

        if not custom_i_data:
            data_loader = self.datamodule.get_dataloader(self.i_data)
        else:
            data_loader = self.datamodule.get_dataloader(custom_i_data)
        
        if isinstance(prediction_point_ids, tuple):
            ids = [x for x in range(prediction_point_ids[0], prediction_point_ids[1])]
            tensor = data_loader.dataset.__getitem__(idx=ids)['x']
            return tensor # shape: (len(ids), in_chunk, in_components)
        elif isinstance(prediction_point_ids, list):
            ids = prediction_point_ids
            tensor = data_loader.dataset.__getitem__(idx=ids)['x']
            return tensor # shape: (len(ids), in_chunk, in_components)
        elif isinstance(prediction_point_ids, int):
            ids = prediction_point_ids
            tensor = data_loader.dataset.__getitem__(idx=ids)['x']
            tensor = torch.unsqueeze(tensor, 0)
            return tensor # shape: (1, in_chunk, in_components)
            
        
            
        
        
    
    
    
    
    
    
    
    
    