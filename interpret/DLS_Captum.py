__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for DeepLiftShap. Uses the Captum library'

from interpret.featureattr import FeatureAttribution

from captum.attr import DeepLiftShap

import torch

from typing import Any, Dict, Union, Literal

from helpers.logger import get_logger

logger = get_logger()

class DLS(FeatureAttribution):
    
    def __init__(self,
                 i_data: str,
                 model: type,
                 model_params: dict,
                 datamodule: object,
                 path_to_ckpt: str):
        super().__init__(model=model,
                        model_params=model_params,
                        datamodule=datamodule,
                        path_to_ckpt=path_to_ckpt,
                        i_data=i_data
                        )
        
        self.dls = DeepLiftShap(self.model)
        
    def generate_baseline(self, baseline_mode, input_tensor_shape):
        # needs to have the same shape as input tensor
        
        if baseline_mode == 'random':
            return torch.rand(2, *input_tensor_shape)
        elif baseline_mode == 'mean': ##### Todo: what do we mean by 'mean'? Ie mean across instances, or slices?
            return 
        elif baseline_mode == 'zeros':
            return torch.zeros(2, *input_tensor_shape)
        elif baseline_mode == 'ones':
            return torch.ones(2, *input_tensor_shape)
        else:
            logger.error("baseline_mode currently not supported. Currently supported: 'random', 'mean', 'zeros', 'ones'")
            exit(101) 
        
    def interpret(self, pred_point_id: Union[int, tuple], baseline_mode: str):
        
        # go into dataloader and find the user's choice of points
        if isinstance(pred_point_id, int):
            input_tensor = super()._pred_point_from_datamodule(pred_point_id)
        
        input_tensor = torch.unsqueeze(input_tensor, 0)
        input_tensor.requires_grad = True
        
        input_tensor_shape = tuple(input_tensor.shape[1:])
        baselines = self.generate_baseline(baseline_mode=baseline_mode, input_tensor_shape=input_tensor_shape)
        baselines.requires_grad = True
        
        # todo: return a feat,delta pair for each point inside out_chunk
        results = {}
        if hasattr(self.datamodule, 'class_set'):
            for key in self.datamodule.class_set.keys():
                target = self.datamodule.class_set[key]
                attributions, delta = self.dls.attribute(inputs=input_tensor, baselines=baselines, target=target, return_convergence_delta=True)
                attributions = torch.squeeze(attributions, 0)
                
                results[target] = {
                    "attributions": attributions,
                    "delta": delta
                }
                
            return results
        else:
            # datamodule output y has shape (out_chunk, out_components)
            out_shape = self.datamodule.out_shape
            
            for out_chunk in range(out_shape[0]):
                for out_component in range(out_shape[1]):
                    target = (out_chunk, out_component)
                    attributions, delta = self.dls.attribute(inputs=input_tensor, baselines=baselines, target=target, return_convergence_delta=True)
                    attributions = torch.squeeze(attributions, 0)
                
                    results[target] = {
                        "attributions": attributions,
                        "delta": delta
                    }
                    
            return results
                    
            
            
            
            
            
        










