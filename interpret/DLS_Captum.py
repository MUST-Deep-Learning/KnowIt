__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the class for DeepLiftShap. Uses the Captum library'

from interpret.featureattr import FeatureAttribution

import torch
import numpy as np

from captum.attr import DeepLiftShap

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
        
    def generate_baseline_from_data(self, num_baselines, pred_point_id):
        
        if self.i_data == 'train':
            num_samples = self.datamodule.train_set_size
        elif self.i_data == 'valid':
            num_samples = self.datamodule.valid_set_size
        else: 
            num_samples = self.datamodule.eval_set_size
            
        # pred_point_id is either int or tuple
        # baselines can't be the same as points from pred_point_id?
        if isinstance(pred_point_id, tuple):
            invalid_ids = [id for id in range(pred_point_id[0], pred_point_id[1])]
        else:
            invalid_ids = [pred_point_id]
            
        baselines = []
        while len(baselines) < num_baselines:
            id = np.random.randint(0, num_samples)
            if id in invalid_ids:
                continue
            
            invalid_ids.append(id)
            baseline = self._pred_point_from_datamodule(id)
            # baseline.requires_grad = True
            baselines.append(baseline)
        
        baselines = torch.stack(baselines, dim=0)
         
        return baselines
        
    def interpret(self, pred_point_id: Union[int, tuple], num_baselines: int):
        
        """
        Example:

        RegressionDataset with: 
            - in_chunk = [-5, 5] -> 11
            - in_components = ['x1', 'x2', 'x3', 'x4'] -> 4
            - out_chunk = [0, 0] -> 1
            - out_components = ['y1', 'y2'] -> 2
            
        Interpretability: User chooses:
            - 'train'
            - pred_point = (500, 505) -> 5 pred points
            - num_baselines = 100
            
        Then: 
            - input_tensor.shape = (num_pred_points, in_chunk, in_components) = (5, 11, 4)
            - baselines.shape = (num_baselines, in_chunk, in_components) = (100, 11, 4)
            - target = (num_pred_points, out_chunk, out_components) = (5, 1, 2)
         
        Output:
        attribution: a nested dict data structure that contains the attribution matrices for each output component and for each prediction point:
            - attribution[pred_point_id][output_element_index_tuple]["attributions" or "delta"]
            
        For example, to access the attribution matrix for point 503 in the pred_point and for output element index (0, 1),
            - attribution[503][(0,1)]["attributions"]
        """
        
        # Todo: code in this section is clunky, fix
        
        # extract the prediction point from dataset in chosen dataloader.
        # input_tensor will have shape: (number_of_pred_points, in_chunk, in_components)
        if isinstance(pred_point_id, int):
            input_tensor = super()._pred_point_from_datamodule(pred_point_id)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            input_tensor.requires_grad = True
        elif isinstance(pred_point_id, tuple):
            input_tensor_list = super()._pred_range_from_datamodule(pred_point_id)
            # input_tensor_list = [torch.unsqueeze(input_tensor, 0) for input_tensor in input_tensor_list]
            for t in input_tensor_list:
                t.requires_grad = True
            input_tensor = torch.stack(input_tensor_list, dim=0)
        
        # generate baselines from the same distribution as the user's chosen dataset
        # baselines will have shape: (number_of_baselines, in_chunk, in_components)
        baselines = self.generate_baseline_from_data(num_baselines=num_baselines, pred_point_id=pred_point_id)

        # todo: return a feat,delta pair for each point inside out_chunk
        if hasattr(self.datamodule, 'class_set'):
            results = {}
            
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
            
            attrib = {}
            if isinstance(pred_point_id, tuple):
                counter = pred_point_id[0]
            else:
                counter = pred_point_id
                
            for i in range(input_tensor.shape[0]):
                cur_input_tensor = input_tensor[i, :, :]
                cur_input_tensor = torch.unsqueeze(cur_input_tensor, 0)
                results = {}
                for out_chunk in range(out_shape[0]):
                    for out_component in range(out_shape[1]):
                        target = (out_chunk, out_component)
                        attributions, delta = self.dls.attribute(inputs=cur_input_tensor, baselines=baselines, target=target, return_convergence_delta=True)
                        attributions = torch.squeeze(attributions, 0)
                
                        results[target] = {
                            "attributions": attributions,
                            "delta": delta
                        }
                attrib[counter] = results
                counter += 1
            return attrib
                    
            
            
            
            
            
        










