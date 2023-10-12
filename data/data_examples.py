import torch

from helpers.logger import get_logger

logger = get_logger()
logger.setLevel(20)

# ----------------------------------------------------------------------------------------------------------------------
# For a classification dataset
# ----------------------------------------------------------------------------------------------------------------------
from data.classification_dataset import ClassificationDataset

# 1. Generate a data option of your raw dataframe
# from data.base_dataset import BaseDataset
# path_to_my_raw_data = '/home/tian/postdoc_work/penguin_PCE/data_as_received/penguin_pce_full.pkl'
# baseDS = BaseDataset.from_path(path_to_my_raw_data, safe_mode=False, base_nan_filler='split', nan_filled_components=None)

# 2. Generate a classification dataset from stored data option
data_option = 'penguin_pce_full'
classification_DS = ClassificationDataset(name=data_option,
                                          in_components=['accX', 'accY', 'accZ'],
                                          out_components=['PCE'], in_chunk=[-25, 25], out_chunk=[0, 0],
                                          split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=32, limit=500000,
                                          min_slice=10, scaling_method='z-norm', scaling_tag='in_only',
                                          split_method='instance-random')

# 4. Train your model with it (or something)
from torch import argmax
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from archs.MLP import Model as mlp
import numpy as np

train_loader = classification_DS.get_dataloader('train')
in_dim = int(np.prod(classification_DS.in_shape))
out_dim = int(len(classification_DS.class_set))
my_network = mlp(in_dim, out_dim, task_name='classification')
loss_fn = CrossEntropyLoss()
optim = Adam(my_network.parameters(), lr=0.0001)
for epoch in range(100):
    correct = 0
    for batch_id, batch in enumerate(train_loader):
        x = batch['x']
        y = batch['y']
        x = x.view(classification_DS.batch_size, in_dim)
        optim.zero_grad()
        h_n = my_network.forward(x)
        loss = loss_fn(h_n, y)
        loss.backward()
        optim.step()

        pred = argmax(h_n, dim=1)
        tp = torch.count_nonzero(torch.logical_and(pred, y))
        fp = torch.count_nonzero(torch.logical_and(pred, torch.logical_not(y)))
        fn = torch.count_nonzero(torch.logical_and(y, torch.logical_not(pred)))
        ping = 0


    print('last batch loss: ' + str(loss.item()))







ping = 0