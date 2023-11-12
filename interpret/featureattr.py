__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for performing feature attribution.'

from interpret.ki_interpreter import KIInterpreter
from captum.attr import DeepLiftShap

import torch

class FeatureAttribution(KIInterpreter):
    """
    To fill
    """
    
    def __init__(self, 
                 model, 
                 model_params, 
                 path_to_ckpt, 
                 datamodule,
                 i_mode,
                 i_data):
        super().__init__(model=model, 
                         model_params=model_params, 
                         path_to_checkpoint=path_to_ckpt,
                         datamodule=datamodule,
                         i_mode=i_mode,
                         i_data=i_data)
    # Needs to handle the cases described on whiteboard and call Captum's DLS 
    # how should I import the datamodule: here or in parent class?
    
    def deepliftshap(self):
        
        input = torch.rand(8, 51, 3)
        baseline = torch.zeros(8, 51, 3)
        
        dls = DeepLiftShap(self.model)
        
        return dls.attribute(inputs=input, baselines=baseline, target=0, return_convergence_delta=True)
    
    
    
    
    
    
    
    
    