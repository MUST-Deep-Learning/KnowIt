__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains examples that demonstrates the Interpreter Module of Knowit.'


"""
Regression Example:

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
    - target: a tuple of index values from the examples corresponding model output
         
The attribution matrices are computed for each (1, 11, 4) example in input_tensor (ie for each example i in input_tensor, we will have a collection 
of atrribution matrices that correspond to the model output tensors index values - see below).
        
attribution: a nested dict data structure that contains the attribution matrices for each output component and for each prediction point:
    - attribution[pred_point_id][output_element_index_tuple]["attributions" or "delta"]
            
For example, to access the attribution matrix for point 503 in the pred_point and for output element index (0, 1),
    - attribution[503][(0,1)]["attributions"]
         
###############################
           
Classification is identical to the above except, for a given input tensor example, the targets provided to Captum is the class value (ie 0 and 1 for binary case) 
instead of the output tensor's index value.
        
"""

from interpret.DLS_Captum import DLS
from archs.MLP import Model

from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset

import torch

from helpers.logger import get_logger

logger = get_logger()
logger.setLevel(20)

# Links:
#   https://proceedings.neurips.cc/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf
#   https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9895252
#   https://www.mdpi.com/2076-3417/12/3/1427
#   https://github.com/pytorch/captum/issues/958
#   https://gist.github.com/smaeland/f132ae58db4aa709d92d49bf2bf19d58
#   https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html

#   https://www.kaggle.com/code/s903124/visualize-nfl-run-play-with-captum
#   https://captum.ai/docs/attribution_algorithms

def main():
    path_to_ckpt = "/home/randle/projects/KnowIt/RegressionTestMLP/models/Model_2023-11-13 10:54:39/bestmodel-epoch=4-val_loss=0.09 2023-11-13 10:54:39.ckpt"
    
    data_option = 'dummy_zero'
    datamodule = RegressionDataset(name=data_option,
                                           in_components=['x1', 'x2', 'x3', 'x4'],
                                           out_components=['y1', 'y2'], in_chunk=[-5, 5], out_chunk=[0, 0],
                                           split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=None,
                                           min_slice=10, scaling_method='z-norm', scaling_tag='full',
                                           split_method='slice-random')
    
    model = Model
    model_params = {
        "input_dim": datamodule.in_shape,
        "output_dim": datamodule.out_shape,
        "task_name": 'regression'
    }
    
    dls = DLS(model=model,
            model_params=model_params, 
            path_to_ckpt=path_to_ckpt, 
            datamodule=datamodule,
            i_data='train') # other option: val, eval
    
    # a single pred_point_id can be provided or a range that is specified by a tuple (start, end) (end not included in range)
    attributions = dls.interpret(pred_point_id=(10000, 10100), num_baselines=1000)
    
    # Example - attribution matrix for a pred point in above and output element index (0,1)
    print(attributions[10033][(0,1)]['attributions'].shape)
    print(attributions[10033][(0,1)]['attributions'])
    
    ###################################################################################
    
    # path_to_ckpt = "/home/randle/projects/KnowIt/ClassificationTesMLP/models/Model_2023-11-09 14:09:07/bestmodel-epoch=0-val_loss=0.01 2023-11-09 14:09:07.ckpt"
    
    # data_option = "penguin_pce_full"
    # datamodule = ClassificationDataset(name=data_option,
    #                  in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
    #                  in_chunk=[-25, 25], out_chunk=[0, 0], 
    #                  split_portions=(0.6, 0.2, 0.2), 
    #                  seed=333, batch_size=32, limit=10000, 
    #                  min_slice=10, scaling_method='z-norm', 
    #                  scaling_tag='in_only', split_method='chronological')
    
    # model = Model    
    # model_params = {
    #     "input_dim": datamodule.in_shape,
    #     "output_dim": datamodule.out_shape,
    #     "task_name": 'classification'
    # }
    
    # dls = DLS(model=model,
    #         model_params=model_params, 
    #         path_to_ckpt=path_to_ckpt, 
    #         datamodule=datamodule,
    #         i_data='valid') # options: train, valid, eval
    
    # # a single pred_point_id can be provided or a range that is specified by a tuple (start, end) (end not included in range)
    # attributions = dls.interpret(pred_point_id=(200, 210), num_baselines=1000)
    
    # # Example - attribution matrices for a pred point in above range and for both class 0 and class 1
    # print(attributions[204][0]['attributions'].shape)
    # print(attributions[204][1]['attributions'].shape)
    # print(attributions[204][0]['attributions'])
    

if __name__ == "__main__":
    main()
    
    
    