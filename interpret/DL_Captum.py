__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for DeepLift. Uses the Captum library'

"""
---------------
DeepL
---------------

The ``DeepL'' class implements the DeepLift feature attribution method using the Captum library. It inherits from the 
parent class``FeatureAttribution''.

``DeepL'' is provided the following arguments:
    - model (class)             : the Pytorch model architecture defined in ./archs
    - model_params (dict)       : the dictionary that contains the models init parameters
    - path_to_ckpt (str)        : the path to a trained model's checkpoint file.
    - datamodule (Knowit obj)   : the Knowit datamodule for the experiment.
    - i_data (str)              : the user's choice of dataset to perform feature attribution. Choices: 'train', 'valid', 'eval'
    
The ``DeepL'' class will take a user's choice of prediction point ids, generate a baselines, and return a dict of attribution 
matrices using Captum. The attribution matrices is specific to the task - regression or classification. 
 
"""


from interpret.featureattr import FeatureAttribution

import torch
import numpy as np

from captum.attr import DeepLift

from typing import Any, Dict, Union, Literal

from helpers.logger import get_logger

logger = get_logger()

class DeepL(FeatureAttribution):
    
    def __init__(self,
                 i_data: str,
                 model: type,
                 model_params: dict,
                 datamodule: object,
                 path_to_ckpt: str,
                 multiply_by_inputs: bool = True):
        super().__init__(model=model,
                        model_params=model_params,
                        datamodule=datamodule,
                        path_to_ckpt=path_to_ckpt,
                        i_data=i_data
                        )
        
        self.dl = DeepLift(self.model, multiply_by_inputs=multiply_by_inputs)
        
    def generate_baseline_from_data(self, num_baselines, pred_point_id):
        
        logger.info("Generating baselines.")
        
        # if self.i_data == 'train':
        #     num_samples = self.datamodule.train_set_size
        # elif self.i_data == 'valid':
        #     num_samples = self.datamodule.valid_set_size
        # else:
        #     num_samples = self.datamodule.eval_set_size
        #
        # # pred_point_id is either int or tuple
        # # baselines can't be the same as points from pred_point_id?
        # if isinstance(pred_point_id, tuple):
        #     invalid_ids = [id for id in range(pred_point_id[0], pred_point_id[1])]
        # else:
        #     invalid_ids = [pred_point_id]
        #
        # baselines = []
        # while len(baselines) < num_baselines:
        #     id = np.random.randint(0, num_samples)
        #     if id in invalid_ids:
        #         continue
        #     baselines.append(id)
        #     invalid_ids.append(id)
        #
        #
        # baselines = self._pred_points_from_datamodule(baselines)

        num_samples = self.datamodule.train_set_size
        if num_samples < num_baselines:
            logger.warning('Not enough prediction points for %s baselines. Using %s.',
                           str(num_samples), str(num_baselines))
            num_baselines = num_samples

        baselines = np.random.choice(np.arange(num_samples), size=num_baselines, replace=False)
        baselines = [b for b in baselines]

        baselines = self._pred_points_from_datamodule(baselines, custom_i_data='train')
        
        return baselines
        
    def interpret(self, pred_point_id: Union[int, tuple], num_baselines: int = 1000):
        
        # Input_tensor below will have shape: (number_of_pred_points, in_chunk, in_components)
        
        input_tensor = super()._pred_points_from_datamodule(pred_point_id)
        input_tensor.requires_grad = True
        
        # generate baselines from the same distribution as the user's chosen dataset
        # baselines will have shape: (number_of_baselines, in_chunk, in_components)
        baselines = self.generate_baseline_from_data(num_baselines=num_baselines, pred_point_id=pred_point_id)
        
        avg_baseline = torch.mean(baselines, 0, keepdim=True)

        # obtain attribution matrices using Captum
        if hasattr(self.datamodule, 'class_set'):
            logger.info("Preparing attribution matrices for classification task.")
            is_classification = True
        else:
            logger.info("Preparing attribution matrices for regression task.")
            is_classification = False
            out_shape = self.datamodule.out_shape
            
        if isinstance(pred_point_id, tuple):
            counter = pred_point_id[0]
        else:
            counter = pred_point_id
                
        attrib = {}
        for example in range(input_tensor.shape[0]):
            cur_input_tensor = input_tensor[example]
            cur_input_tensor = torch.unsqueeze(cur_input_tensor, 0) # Captum requires extra dimension
                
            results = {}
            if is_classification:
                for key in self.datamodule.class_set.keys():
                    target = self.datamodule.class_set[key]
                    attributions, delta = self.dl.attribute(inputs=cur_input_tensor, baselines=avg_baseline, target=target, return_convergence_delta=True)
                    attributions = torch.squeeze(attributions, 0)
                
                    results[target] = {
                        "attributions": attributions,
                        "delta": delta
                    }
            else:
                for out_chunk in range(out_shape[0]):
                    for out_component in range(out_shape[1]):
                        target = (out_chunk, out_component)
                        attributions, delta = self.dl.attribute(inputs=cur_input_tensor, baselines=avg_baseline, target=target, return_convergence_delta=True)
                        attributions = torch.squeeze(attributions, 0)
                
                        results[target] = {
                            "attributions": attributions,
                            "delta": delta
                        }
                
            attrib[counter] = results
            counter += 1
                
        return attrib
                    
            
            
            
            
            
        










