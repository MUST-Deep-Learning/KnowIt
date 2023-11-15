__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for performing feature attribution.'

from interpret.ki_interpreter import KIInterpreter

import torch

from helpers.logger import get_logger

logger = get_logger()

class FeatureAttribution(KIInterpreter):
    """
    To fill
    """
    
    def __init__(self, 
                 model, 
                 model_params, 
                 path_to_ckpt, 
                 datamodule,
                 i_data
                 ):
        super().__init__(model=model, 
                         model_params=model_params, 
                         path_to_checkpoint=path_to_ckpt,
                         datamodule=datamodule
                         )
    
    # Todo: for the moment, i_mode is a single point. Needs to handle range
        self.i_data = i_data
    
    def _pred_point_from_datamodule(self, prediction_point_id):
        
        # call the relevant dataloader from the datamodule
        # A point or range is chosen by the user.
        # We have to use the idx from the dataloader and work backwards to Knowit's data structure and extract the datapoints. This 
        # gives us the slice, instances, and components
        
        # start with a single prediction point
        
        # retrieve chosen dataloader inside datamodule
        data_loader = self.datamodule.get_dataloader(self.i_data)
        x = data_loader.dataset.__getitem__(idx=prediction_point_id)['x']
        
        # extract the first data point
        # it is a dictionary containing 'x' (shape=(batch_size, in_chunk, in_components)), 'y' (shape=(batch_size, out_chunk)), 
        # 's_id' (shape = [batch_size])      
        
        return x
    
    def _pred_range_from_datamodule(self):
        # returns tuple of tensors
        pass
        
    
    
    
    
    
    
    
    
    