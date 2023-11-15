__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains examples that demonstrates the Interpreter Module of Knowit.'

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
            i_data='train') # other option: valid, eval
    
    results = dls.interpret(pred_point_id=200, baseline_mode='ones')
    print(results.keys())
    print(results[(0,0)]['attributions'].shape, results[(0,0)]['delta'].shape)
    print(results[(0,1)]['attributions'].shape, results[(0,1)]['delta'].shape)
    print(results[(0,0)]['attributions'])
    
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
    #         i_data='train') # other option: valid, eval
    
    # results = dls.interpret(pred_point_id=200, baseline_mode='random')
    # print(results.keys())
    # print(results[0]['attributions'].shape, results[0]['delta'].shape)
    # print(results[1]['attributions'].shape, results[0]['delta'].shape)

if __name__ == "__main__":
    main()
    
    
    