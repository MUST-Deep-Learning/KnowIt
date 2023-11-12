__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains examples that demonstrates the Interpreter Module of Knowit.'

from interpret.featureattr import FeatureAttribution
from archs.MLP import Model

from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset

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
    path_to_ckpt = "/home/randle/projects/KnowIt/ClassificationTesMLP/models/Model_2023-11-09 14:09:07/bestmodel-epoch=0-val_loss=0.01 2023-11-09 14:09:07.ckpt"
    
    data_option = "penguin_pce_full"
    datamodule = ClassificationDataset(name=data_option,
                     in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
                     in_chunk=[-25, 25], out_chunk=[0, 0], 
                     split_portions=(0.6, 0.2, 0.2), 
                     seed=333, batch_size=32, limit=10000, 
                     min_slice=10, scaling_method='z-norm', 
                     scaling_tag='in_only', split_method='chronological')
    
    # trainer_loader = datamodule.get_dataloader('train')
    # val_loader = datamodule.get_dataloader('valid')
    # eval_loader = datamodule.get_dataloader('eval')
    # model = Model
    # model_params = {
    #     "input_dim": datamodule.in_shape,
    #     "output_dim": datamodule.out_shape,
    #     "task_name": 'classification'
    # }
    
    # fa = FeatureAttribution(model=model,
    #                         model_params=model_params, 
    #                         path_to_ckpt=path_to_ckpt, 
    #                         datamodule=datamodule,
    #                         i_mode = 'point', # other option: range
    #                         i_data='train') # other option: valid, eval
    
    # attributions, delta = fa.deepliftshap()
    # print(attributions.shape, delta.shape)
    
    print(datamodule.__dict__.keys())
    print(next(iter(datamodule.get_dataloader('train'))))
    

if __name__ == "__main__":
    main()