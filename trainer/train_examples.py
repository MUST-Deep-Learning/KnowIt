from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from data.base_dataset import BaseDataset
from trainer import KITrainer

from archs.MLP import Model

import torch

import pickle

def test_func(**kwargs):
    for key, value in kwargs.items():
        print("%s==%s" % (key, value))

def main():
    data_option = "penguin_pce_full"
    datamodule = ClassificationDataset(name=data_option,
                     in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
                     in_chunk=[-25, 25], out_chunk=[0, 0], 
                     split_portions=(0.6, 0.2, 0.2), 
                     seed=333, batch_size=32, limit=10000, 
                     min_slice=10, scaling_method='z-norm', 
                     scaling_tag='in_only', split_method='chronological')
    
    # data_option = 'dummy_zero'
    # datamodule = RegressionDataset(name=data_option,
    #                                       in_components=['x1', 'x2', 'x3', 'x4'],
    #                                       out_components=['y1', 'y2'], in_chunk=[-5, 5], out_chunk=[0, 0],
    #                                       split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=None,
    #                                       min_slice=10, scaling_method='zero-one', scaling_tag='full',
    #                                       split_method='slice-random')
    
    trainer_loader = datamodule.get_dataloader('train')
    val_loader = datamodule.get_dataloader('valid')
    
    model = Model(input_dim=datamodule.in_shape, 
                  output_dim=datamodule.out_shape,
                  task_name='regression',
                  output_activation='Softmax')
    
    trainer = KITrainer(train_device='gpu',
                        loss_fn='cross_entropy',
                        optim='Adam',
                        max_epochs=250,
                        early_stopping=True,
                        learning_rate=1e-03,
                        #learning_rate_scheduler={'lr_scheduler': 'ReduceLROnPlateau', 'monitor': 'train_loss', 'mode': 'min', 'patience': 2},
                        learning_rate_scheduler={'lr_scheduler': 'ExponentialLR', 'gamma': 0.9},
                        loaders=(trainer_loader, val_loader),
                        model=model)
    
    # trainer = KITrainer(train_device='gpu',
    #                     loss_fn='mse_loss',
    #                     optim='Adam',
    #                     max_epochs=250,
    #                     early_stopping=True,
    #                     learning_rate=1e-03,
    #                     #learning_rate_scheduler={'lr_scheduler': 'ReduceLROnPlateau', 'monitor': 'train_loss', 'mode': 'min', 'patience': 2},
    #                     learning_rate_scheduler={'lr_scheduler': 'ExponentialLR', 'gamma': 0.9},
    #                     loaders=(trainer_loader, val_loader),
    #                     model=model)
    
    # train
    trainer.fit_model()
    
    # test
    trainer.test_model()
    # test_dic = {"a": 0, "b": 1}
    # #test_func(**test_dic)

    
main()