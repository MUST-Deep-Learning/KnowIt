from data.classification_dataset import ClassificationDataset as clf
from data.base_dataset import BaseDataset
from trainer import KITrainer

from archs.MLP import Model

import torch

import pickle

def main():
    data_option = "penguin_pce_full"
    datamodule = clf(name=data_option,
                     in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
                     in_chunk=[-25, 25], out_chunk=[0, 0], 
                     split_portions=(0.6, 0.2, 0.2), 
                     seed=333, batch_size=32, limit=10000, 
                     min_slice=10, scaling_method='z-norm', 
                     scaling_tag='in_only', split_method='chronological')
    
    trainer_loader = datamodule.get_dataloader('train')
    val_loader = datamodule.get_dataloader('valid')
    print(trainer_loader)
    #test_loader = datamodule.get_dataloader('test')
    #d = next(iter(trainer_loader))
    #print(d.keys())
    # print(d['x'].shape)
    # print(d['y'].shape)
    # print(d['s_id'].shape)
    # print(d['y'])
    #print(d['x'].shape)
    
    model = Model(input_dim=datamodule.in_shape, # 51 in_chunk, 3 in_components
                  output_dim=datamodule.out_shape, # out_chunk=[0, 0] ie window of size 1
                  task_name='classification',
                  output_activation='Softmax')
    
    trainer = KITrainer(train_device='gpu',
                        loss_fn='cross_entropy',
                        optim='Adam',
                        max_epochs=10,
                        early_stopping=True,
                        learning_rate=1e-03,
                        learning_rate_schedule='None', # fix this, does nothing at the moment
                        loaders=(trainer_loader, val_loader),
                        model=model)
    # train
    trainer._fit_model()
    
    # test
    trainer._test_model()
    
main()