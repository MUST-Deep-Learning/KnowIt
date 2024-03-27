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
    - i_data (str)              : the user's choice of dataset to perform feature attribution. 
                                    Choices: 'train', 'valid', 'eval'
    
The function of the ``FeatureAttribution'' class is to serve the user's choice of feature attribution 
method (a child class) by extracting the necessary information from Knowit's datamodule and returning 
it in the correct form.
 
"""

from typing import Union

from interpret.interpreter import KIInterpreter

from torch import tensor

from helpers.logger import get_logger

logger = get_logger()

class FeatureAttribution(KIInterpreter):
    
    def __init__(self, 
                 model: type, 
                 model_params: dict, 
                 path_to_ckpt: str, 
                 datamodule: object,
                 i_data: str
                 ) -> None:
        super().__init__(model=model, 
                         model_params=model_params, 
                         path_to_checkpoint=path_to_ckpt,
                         datamodule=datamodule
                         )
    
        self.i_data = i_data
    
    def _fetch_points_from_datamodule(self, 
                                      point_ids: Union[int, tuple], 
                                      is_baseline: bool=False) -> tensor:
        
        """
        Using a singe id or a range of ids, extracts the corresponding data points from a Pytorch dataloader.
        
        Args:
            prediction_points_ids: Union[int, tuple]                A single prediction point or a range specified by
                                                                    a tuple.
            is_baseline: bool                                       A flag to determine if user is sampling points for 
                                                                    generating baselines.                  

        Returns:
            torch.tensor                                            A tensor of shape (number_of_points_ids, in_chunk, in_components)
        """

        if is_baseline:
            data_loader = self.datamodule.get_dataloader("train") # only sample baselines from training set
        else:
            data_loader = self.datamodule.get_dataloader(self.i_data) 
        
        if isinstance(point_ids, tuple):
            ids = [x for x in range(point_ids[0], point_ids[1])]
        else:
            ids = point_ids

        
        try:
            tensor = data_loader.dataset.__getitem__(idx=ids)['x']
        except:
            logger.error("Invalid: ids %s not in choice \"%s\" (which has range %s)", 
                             str(point_ids), str(self.i_data), str(getattr(self.datamodule, self.i_data + "_set_size")))
            exit(101)
            
        return tensor
        
            
        
        
    
    
    
    
    
    
    
    
    