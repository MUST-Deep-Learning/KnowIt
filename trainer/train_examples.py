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
    #test_loader = datamodule.get_dataloader('test')
    d = next(iter(trainer_loader))
    print(d.keys())
    # print(d['x'].shape)
    # print(d['y'].shape)
    # print(d['s_id'].shape)
    # print(d['y'])
    print(d['x'].shape)
    print(torch.flatten(d['x'], start_dim=1, end_dim=-1).shape)
    
    model = Model(input_dim=51*3, # 51 in_chunk, 3 in_components
                  output_dim=1, # out_chunk=[0, 0] ie window of size 1
                  task_name='classification',
                  output_activation='Softmax')
    
    trainer = KITrainer(train_device='gpu',
                        loss_fn='cross_entropy',
                        optim='Adam',
                        max_epochs=10,
                        early_stopping=False,
                        learning_rate=1e-03,
                        learning_rate_schedule='None', # fix this, does nothing at the moment
                        loaders=(trainer_loader, val_loader),
                        model=model)
    
    test_batch = zip(torch.flatten(d['x'], start_dim=1, end_dim=-1), d['y'])
    print(set(test_batch)[0])
    
    #print(trainer.lit_model.training_step(batch=test_batch, batch_idx=d['s_id']))

main()