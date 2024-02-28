from __future__ import annotations

__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for IntegratedGrad. Uses the Captum library'

"""
---------------
IntegratedGrad
---------------

The ``IntegratedGrad'' class implements the integrated gradients feature attribution 
method using the Captum library. It inherits from the parent class``FeatureAttribution''.

``IntegratedGrad'' is provided the following arguments:
    - model (class)             : the Pytorch model architecture defined in ./archs
    - model_params (dict)       : the dictionary that contains the models init parameters
    - path_to_ckpt (str)        : the path to a trained model's checkpoint file.
    - datamodule (Knowit obj)   : the Knowit datamodule for the experiment.
    - i_data (str)              : the user's choice of dataset to perform feature attribution. 
                                    Choices: 'train', 'valid', 'eval'
    
The ``IntegratedGrad'' class will take a user's choice of prediction point ids, generate an 
average baseline, and return a dict of attribution matrices using Captum. The attribution matrices 
is specific to the task type (regression or classification). 
 
"""

from typing import TYPE_CHECKING, Dict, Union, Tuple

if TYPE_CHECKING:
    import archs

from interpret.featureattr import FeatureAttribution

import torch
import numpy as np

from captum.attr import IntegratedGradients

from helpers.logger import get_logger

logger = get_logger()

class IntegratedGrad(FeatureAttribution):
    
    def __init__(self,
                 i_data: str,
                 model: type,
                 model_params: dict,
                 datamodule: object,
                 path_to_ckpt: str,
                 multiply_by_inputs: bool = True) -> None:
        super().__init__(model=model,
                        model_params=model_params,
                        datamodule=datamodule,
                        path_to_ckpt=path_to_ckpt,
                        i_data=i_data
                        )
        
        self.ig = IntegratedGradients(self.model, multiply_by_inputs=multiply_by_inputs)
        
    def generate_baseline_from_data(self, 
                                    num_baselines: int) -> torch.tensor:
        
        """
        Returns an average baseline. The average is computed over a random sample of size num_baselines from 
        the training data.
        
        Args:
            num_baselines: int                      The total number of baselines to sample.

        Returns:
            torch.tensor                            A torch tensor of shape (1, in_chunk, in_components)
        """
        
        logger.info("Generating baselines.")
        
        num_samples = self.datamodule.train_set_size
        if num_samples < num_baselines:
            logger.warning('Not enough prediction points for %s baselines. Using %s.',
                           str(num_samples), str(num_baselines))
            num_baselines = num_samples

        baselines = np.random.choice(np.arange(num_samples), size=num_baselines, replace=False)
        baselines = [b for b in baselines]
        
        baselines = self._fetch_points_from_datamodule(baselines, is_baseline=True)
        avg_baseline = torch.mean(baselines, 0, keepdim=True)
        
        return avg_baseline
        
    def interpret(self, 
                  pred_point_id: Union[int, tuple], 
                  num_baselines: int = 1000) -> Dict[Union[int, Tuple], Dict[str, torch.tensor]]:
        
        """
        Generates attribution matrices for a single prediction point or a range of prediction points 
        (also referred to as explicands).
        
        Note: The output stores the information from a tensor of size
        (out_chunk, out_components, number_of_prediction_points, in_chunk, in_components) 
        in a dictionary data structure. Especially for time series data, this can grow rapidly.
        
        Args:
            pred_point_id: Union[int, tuple]        The prediction point or range of prediction points that will 
                                                    be used to generate attribution matrices.
            num_baselines: int                      Specifies the size of the baseline distribution.

        Returns:
            results: Dict                           For a regression model with output shape (out_chunk, out_components), 
                                                    returns a dictionary as follows:
                                                        * Dict Key: a tuple (a, b) with a in range(out_chunk) and 
                                                                    b in range(out_features)
                                                        * Dict Element: a torch tensor with shape:
                                                                            > (number_of_prediction_points, in_chunk, in_components)
                                                                            if prediction_point_id is a tuple
                                                                            > (in_chunk, in_components) if prediction_point_id is int.
                                                                            
                                                    For a classification model with output shape (classes,) returns a dictionary 
                                                    as follows:
                                                        * Dict Key: a class value from classes
                                                        * Dict Element: a torch tensor with shape:
                                                                            > (number_of_prediction_points, in_chunk, in_components)
                                                                            if prediction_point_id is a tuple
                                                                            > (in_chunk, in_components) if prediction_point_id is int. 
        """
        
        # extract explicands using ids
        input_tensor = super()._fetch_points_from_datamodule(pred_point_id)
        input_tensor.requires_grad = True # required by Captum
        
        if not isinstance(pred_point_id, tuple):
            input_tensor = torch.unsqueeze(input_tensor, 0) # Captum requires batch dimension = number of explicands
        
        # generate a baseline (baseline is computed as an average over a distribution)
        baselines = self.generate_baseline_from_data(num_baselines=num_baselines)

        # determine model output type
        if hasattr(self.datamodule, 'class_set'):
            logger.info("Preparing attribution matrices for classification task.")
            is_classification = True
        else:
            logger.info("Preparing attribution matrices for regression task.")
            is_classification = False
            out_shape = self.datamodule.out_shape            

        # compute attribution matrices for each output component and each input explicand
        results = {}
        if is_classification:
            for key in self.datamodule.class_set.keys():
                target = self.datamodule.class_set[key]
                attributions, delta = self.ig.attribute(inputs=input_tensor, 
                                                        baselines=baselines, 
                                                        target=target, 
                                                        return_convergence_delta=True)
                attributions = torch.squeeze(attributions, 0)
            
                results[target] = {
                    "attributions": attributions,
                    "delta": delta
                }
        else:
            for out_chunk in range(out_shape[0]):
                for out_component in range(out_shape[1]):
                    target = (out_chunk, out_component)
                    attributions, delta = self.ig.attribute(inputs=input_tensor, 
                                                            baselines=baselines, 
                                                            target=target, 
                                                            return_convergence_delta=True)
                    attributions = torch.squeeze(attributions, 0)
            
                    results[target] = {
                        "attributions": attributions,
                        "delta": delta
                    }
                
        
        return results
                    
            
            
            
            
            
        










