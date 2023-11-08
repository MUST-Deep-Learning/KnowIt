__author__ = 'randlerabe@gmail.com'
__description__ = 'This is a scratchpad for debugging the trainer framework.'

# Instructions:
#
# There are three states for Knowit's Trainer module:
# State 1: Train a new model from scratch.
# State 2: Continue training an existing model from checkpoint.
# State 3: Load a trained model and evaluate it on a eval set.
#
# For each case, the user needs to provide the following parameters:
    
# State 1: instantiate KITrainer
#   Compulsory User Parameters:
#           > experiment_name: str.         The name of the experiment.
#           > train_device: str.            The choice of training device ('cpu', 'gpu', etc).
#           > loss_fn: str or dict.         The choice of loss function (see Pytorch's torch.nn.functional documentation)
#           > optim: str or dict:           The choice of optimizer (see Pytorch's torch.nn.optim documentation)
#           > max_epochs: int.              The number of epochs that the model should train on.
#           > learning_rate: float.         The learning rate that the chosen optimizer should use.
#           > model: Class.                 The Pytorch model architecture define by the user in Knowits ./archs subdir.
#           > model_params: dict.           A dictionary with values needed to init the above Pytorch model.
#   Optional User Parameters:
#           > learning_rate_scheduler: dict. Default: {}.       A dictionary that specifies the learning rate scheduler 
#               and any needed kwargs.
#           > performance_metrics: None or dict. Default: None. Specifies any performance metrics on the validation 
#               set during training.
#           > early_stopping: bool or dict. Default: False.     Specifies early stopping conditions.
#           > gradient_clip_val: float: Default: 0.0.           Clips exploding gradients according to the chosen 
#               gradient_clip_algorithm.
#           > gradient_clip_algorithm: str. Default: 'norm'.    Specifies how the gradient_clip_val should be applied.
#           > set_seed: int or bool. Default: False.            A global seed applied by Pytorch Lightning for reproducibility.
#           > deterministic: bool, str, or None. Default: None. Pytorch Lightning attempts to further reduce randomness 
#               during training. This may incur a performance hit.
#           > safe_mode: bool. Default: False.                  If set to True, aborts the model training if the experiment name already 
#               exists in the user's project output folder.

#   To train the model, the user calls the ".fit_model" method. The method must be provided a tuple consisting of the train 
#   data loader and the validation data loader (in this order).
#
# State 2: Instantiate KITrainer using "resume_from_ckpt" method.
#   Compulsory User Parameters:
#           > experiment_name: str. The name of the experiment.
#           > path_to_checkpoint: str. The path to the pretrained model's checkpoint.
#           > max_epochs: int. The additional number of epochs the user would like to train the model on. If the model is 
#               trained for 10 epochs, say, and the user would like to train for an additional 5 epochs, then this value needs 
#               to be set to 15 epochs (not 5 epochs).
#   Optional User parameters:
#           > set_seed: int. Default: None. The global seed set by Pytorch Lightning. The seed should be the same as used to train the 
#               checkpoint model.
#
# State 3: Instantiate Trainer using "eval_from_ckpt" method.
#   Compulsory User Parameters:
#           > experiment_name: str. The name of the experiment.
#           > path_to_checkpoint: str. The path to the pretrained model's checkpoint 
#
# Checkpointing: Once a model has been trained, the best model checkpoint is saved to the user's project output folder under the
# name of the experiment.
#
# Testing: The model can be tested on an eval set using the "evaluate_model" method on an appropriately instantiated KITrainer (see 
# below examples).



from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from data.base_dataset import BaseDataset
from trainer import KITrainer

import torch

#from archs.MLP import Model
from archs.TCN import Model
#from archs.CNN import Model

from helpers.logger import get_logger

logger = get_logger()
logger.setLevel(20)

def main():
    
    # ----------------------------------------------------------------------------------------------------------------------
    # For a classification dataset
    # ----------------------------------------------------------------------------------------------------------------------
    
    
    data_option = "penguin_pce_full"
    datamodule = ClassificationDataset(name=data_option,
                     in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
                     in_chunk=[-25, 25], out_chunk=[0, 0], 
                     split_portions=(0.6, 0.2, 0.2), 
                     seed=333, batch_size=32, limit=10000, 
                     min_slice=10, scaling_method='z-norm', 
                     scaling_tag='in_only', split_method='chronological')
    
    trainer_loader = datamodule.get_dataloader('train')
    val_loader = datamodule.get_dataloader('valid')
    eval_loader = datamodule.get_dataloader('eval')
    
    model = Model    
    model_params = {
        "input_dim": datamodule.in_shape,
        "output_dim": datamodule.out_shape,
        "task_name": 'classification'
    }
    
    pm = {
        'f1_score': {
            'num_classes': 2,
            'task': 'binary'
        },
        'accuracy': {
            'num_classes': 2,
            'task': 'binary'
        }
    }
    
    loss_fn = {
        'cross_entropy':{
            'weight': torch.Tensor([0.00972, 1.0]).to('cuda')
        }
    }
    
    lr_sched = {
        'ExponentialLR':{
            'gamma': 0.9
        }
    }
    
    optim ={
        'Adam':{
            'weight_decay': 0.5
        }
    }
    
    early_stopping = {
        True:
            {
                'monitor': 'val_loss',
                'mode': 'min'
            }
    }
    
    #################### State 1: Train from scratch and test ####################
    
    # trainer = KITrainer(experiment_name="ClassificationTestTCN",
    #                     train_device='gpu',
    #                     loss_fn=loss_fn,
    #                     performance_metrics=pm, 
    #                     optim=optim,
    #                     max_epochs=15,
    #                     #early_stopping=early_stopping,
    #                     learning_rate=1e-03,
    #                     learning_rate_scheduler=lr_sched,
    #                     model=model,
    #                     model_params=model_params,
    #                     gradient_clip_val=0.5, # TCN
    #                     gradient_clip_algorithm='norm', # TCN
    #                     set_seed=13)
    
    # trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
    
    #################### State 2: Resume training from ckpt and test ####################
    # best_model_path = '/home/randle/projects/KnowIt/ClassificationTestTCN/models/Model_2023-11-07 10:46:17/bestmodel-epoch=1-val_loss=0.01 2023-11-07 10:46:17.ckpt'
    # trainer = KITrainer.resume_from_ckpt(experiment_name="ClassificationTestTCN", max_epochs=20, path_to_checkpoint=best_model_path, set_seed=13, safe_mode=False)
    # trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
    
    #################### State 3: Load from ckpt and test ####################
    # best_model_path = '/home/randle/projects/KnowIt/ClassificationTestTCN/models/Model_2023-11-08 12:04:15/bestmodel-epoch=1-val_loss=0.01 2023-11-08 12:04:15.ckpt'
    # trainer = KITrainer.eval_from_ckpt(experiment_name="ClassificationTestTCN", path_to_checkpoint=best_model_path)
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
        
    
    # ----------------------------------------------------------------------------------------------------------------------
    # For a regression dataset
    # ----------------------------------------------------------------------------------------------------------------------
    
    # data_option = 'dummy_zero'
    # datamodule = RegressionDataset(name=data_option,
    #                                        in_components=['x1', 'x2', 'x3', 'x4'],
    #                                        out_components=['y1', 'y2'], in_chunk=[-5, 5], out_chunk=[0, 0],
    #                                        split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=None,
    #                                        min_slice=10, scaling_method='z-norm', scaling_tag='full',
    #                                        split_method='slice-random')
    
    # trainer_loader = datamodule.get_dataloader('train')
    # val_loader = datamodule.get_dataloader('valid')
    # eval_loader = datamodule.get_dataloader('eval')
    
    # model = Model
    # model_params = {
    #     "input_dim": datamodule.in_shape,
    #     "output_dim": datamodule.out_shape,
    #     "task_name": 'regression'
    # }
    
    # pm = 'mean_squared_error'
    
    # loss_fn = 'mse_loss'
    
    
    # lr = {
    #     'ExponentialLR':{
    #         'gamma': 0.9
    #     }
    # }
    
    # optim ={
    #     'SGD':{
    #         'momentum': 0.9
    #     }
    # }
    
    # early_stopping = {
    #     True:
    #         {
    #             'monitor': 'val_loss',
    #             'mode': 'min'
    #         }
    # }
    
    #################### State 1: Train from scratch and test ####################
    # trainer = KITrainer(experiment_name='RegressionTestMLP',
    #                     train_device='cpu',
    #                     model=model,
    #                     model_params=model_params,
    #                     loss_fn=loss_fn,  
    #                     optim=optim,
    #                     performance_metrics=pm,
    #                     max_epochs=5,
    #                     early_stopping=early_stopping,
    #                     learning_rate=1e-02,
    #                     learning_rate_scheduler=lr,
    #                     gradient_clip_val=0.5, # TCN
    #                     gradient_clip_algorithm='norm', # TCN
    #                     set_seed=42,
    #                     deterministic=False,
    #                     safe_mode=False,
    #                     mute_logger=False)
    
    # trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
    
    #################### State 2: Resume training from ckpt and test ####################
    # best_model_path = '/home/randle/projects/KnowIt/models/Model_2023-11-06 10:52:31/bestmodel-epoch=7-val_loss=0.09 2023-11-06 10:52:31.ckpt'
    # trainer = KITrainer.resume_from_ckpt(experiment_name='RegressionTestMLP', max_epochs=20, path_to_checkpoint=best_model_path, set_seed=42, safe_mode=False)
    # trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
    
    #################### State 3: Load from ckpt and test ####################
    # restore from ckpt and test
    # best_model_path = '/home/randle/projects/KnowIt/RegressionTestMLP/models/Model_2023-11-08 11:49:25/bestmodel-epoch=4-val_loss=0.09 2023-11-08 11:49:25.ckpt'
    # trainer = KITrainer.eval_from_ckpt(experiment_name='RegressionTestMLP', path_to_checkpoint=best_model_path)
    # trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
    
    
    ############################################################################################
    ############################################################################################
    pass

if __name__ == "__main__":
    main()