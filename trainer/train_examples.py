from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from data.base_dataset import BaseDataset
from trainer import KITrainer

from archs.MLP import Model
#from archs.TCN import Model
#from archs.CNN import Model

import torch
from torchmetrics import F1Score, Accuracy

from helpers.logger import get_logger

logger = get_logger()
logger.setLevel(20)

# Todo:
#   > write instructions/documentation on how to use KITrainer below.

def main():
    
    
    # ----------------------------------------------------------------------------------------------------------------------
    # For a classification dataset
    # ----------------------------------------------------------------------------------------------------------------------
    
    # data_option = "penguin_pce_full"
    # datamodule = ClassificationDataset(name=data_option,
    #                  in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
    #                  in_chunk=[-25, 25], out_chunk=[0, 0], 
    #                  split_portions=(0.6, 0.2, 0.2), 
    #                  seed=333, batch_size=32, limit=10000, 
    #                  min_slice=10, scaling_method='z-norm', 
    #                  scaling_tag='in_only', split_method='chronological')
    
    # trainer_loader = datamodule.get_dataloader('train')
    # val_loader = datamodule.get_dataloader('valid')
    # test_loader = datamodule.get_dataloader('eval')
    
    # model = Model(input_dim=datamodule.in_shape, 
    #               output_dim=datamodule.out_shape,
    #               task_name='classification')
    
    # Instructions:
    # 1. To set up the trainer the user needs to provide:
    #   > train_device
    #   > loss_fn
    #   > optim
    #   > max_epochs
    #   > learning_rate
    #   > loaders
    #
    #
    #
    #
    #
    #
    
    # pm = {
    #     'f1_score': {
    #         'num_classes': 2,
    #         'task': 'binary'
    #     },
    #     'accuracy': {
    #         'num_classes': 2,
    #         'task': 'binary'
    #     }
    # }
    
    # loss_fn = {
    #     'cross_entropy':{
    #         'weight': torch.Tensor([0.00972, 1.0]).to('cuda')
    #     }
    # }
    
    # # if 'ReduceLROnPlateau', must use "train_or_val_loss" + "loss_function" as specified elsewhere
    # # lr = {
    # #     'ReduceLROnPlateau':{
    # #         'mode': 'min',
    # #         'patience': 5,
    # #         'monitor': 'train_loss_cross_entropy'
    # #     }
    # # }
    
    # lr = {
    #     'ExponentialLR':{
    #         'gamma': 0.9
    #     }
    # }
    
    # optim ={
    #     'Adam':{
    #         'weight_decay': 0.5
    #     }
    # }
    
    # early_stopping = {
    #     True:
    #         {
    #             'monitor': 'val_loss',
    #             'mode': 'min'
    #         }
    # }
    
    # # early_stopping = {
    # #     True
    # # }
        
    # # trainer = KITrainer(train_device='gpu',
    # #                     loss_fn=loss_fn,
    # #                     performance_metrics=pm, 
    # #                     optim=optim,
    # #                     max_epochs=20,
    # #                     #early_stopping=early_stopping,
    # #                     learning_rate=1e-03,
    # #                     #learning_rate_scheduler=lr,
    # #                     loaders=(trainer_loader, val_loader),
    # #                     model=model)
    
    # # train
    # # trainer.fit_model()
    
    # # restore from ckpt and test
    # best_model_path = '/home/randle/projects/KnowIt/checkpoints/Checkpoint_2023-10-31 17:24:43/bestmodel-epoch=8-val_loss=0.00 2023-10-31 17:24:43.ckpt'
    
    # trainer = KITrainer.build_from_ckpt(best_model_path)
    
    # # test
    # trainer.test_model(test_dataloader=test_loader)
    
    
    
    
    ############################################################################################
    ###############################   REGRESSION   #############################################
    
    data_option = 'dummy_zero'
    datamodule = RegressionDataset(name=data_option,
                                           in_components=['x1', 'x2', 'x3', 'x4'],
                                           out_components=['y1', 'y2'], in_chunk=[-5, 5], out_chunk=[0, 0],
                                           split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=None,
                                           min_slice=10, scaling_method='z-norm', scaling_tag='full',
                                           split_method='slice-random')
    
    trainer_loader = datamodule.get_dataloader('train')
    val_loader = datamodule.get_dataloader('valid')
    eval_loader = datamodule.get_dataloader('eval')
    
    model = Model
    model_params = {
        "input_dim": datamodule.in_shape,
        "output_dim": datamodule.out_shape,
        "task_name": 'regression'
    }
    
    pm = 'mean_squared_error'
    
    loss_fn = 'mse_loss'
    
    
    lr = {
        'ExponentialLR':{
            'gamma': 0.9
        }
    }
    
    optim ={
        'SGD':{
            'momentum': 0.9
        }
    }
    
    early_stopping = {
        True:
            {
                'monitor': 'val_loss',
                'mode': 'min'
            }
    }
    
    #################### Train from scratch and test ####################
    # file = torch.load("/home/randle/projects/KnowIt/models/Model_2023-11-06 13:09:21/bestmodel-epoch=7-val_loss=0.09 2023-11-06 13:09:21.ckpt")
    # print(file.keys())
    # print(file['epoch'])
    # print(file['callbacks'].keys())
    # print(file['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"])
    # print(file['callbacks']["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["best_model_score"].item())
    trainer = KITrainer(experiment_name='RegressionTestMLP',
                        train_device='gpu',
                        model=model,
                        model_params=model_params,
                        loss_fn=loss_fn,  
                        optim=optim,
                        performance_metrics=pm,
                        max_epochs=10,
                        early_stopping=early_stopping,
                        learning_rate=1e-02,
                        learning_rate_scheduler=lr,
                        gradient_clip_val=0.5, # TCN
                        gradient_clip_algorithm='norm', # TCN
                        set_seed=42,
                        deterministic=False,
                        safe_mode=False)
    
    trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    trainer.evaluate_model(eval_dataloader=eval_loader)
    
    #################### Resume training from ckpt and test ####################
    # best_model_path = '/home/randle/projects/KnowIt/models/Model_2023-11-06 10:52:31/bestmodel-epoch=7-val_loss=0.09 2023-11-06 10:52:31.ckpt'
    # trainer = KITrainer.resume_from_ckpt(max_epochs=20, path_to_checkpoint=best_model_path, set_seed=42)
    # trainer.fit_model(dataloaders=(trainer_loader, val_loader))
    # trainer.evaluate_model(eval_dataloader=eval_loader)
    
    #################### Load from ckpt and test ####################
    # restore from ckpt and test
    # best_model_path = '/home/randle/projects/KnowIt/models/Model_2023-11-06 13:09:21/bestmodel-epoch=7-val_loss=0.09 2023-11-06 13:09:21.ckpt'
    # trainer = KITrainer.eval_from_ckpt(path_to_checkpoint=best_model_path)
    # trainer.evaluate_model(eval_dataloader=eval_loader)
    
    
    ############################################################################################
    #####################################  RUN TRAINER  ########################################
    
    
        
main()