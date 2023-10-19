__author__ = 'tiantheunissen@gmail.com'
__description__ = 'This is a scratchpad for debugging the data framework.'

import torch
from helpers.logger import get_logger

logger = get_logger()
logger.setLevel(20)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # if true => very slow
train_device = torch.device('cuda')
torch.seed()

# ----------------------------------------------------------------------------------------------------------------------
# For a classification dataset
# ----------------------------------------------------------------------------------------------------------------------
from data.classification_dataset import ClassificationDataset

# 1. Generate a data option of your raw dataframe
# from data.base_dataset import BaseDataset
# path_to_my_raw_data = '/home/tian/postdoc_work/penguin_PCE/data_as_received/penguin_pce_full.pkl'
# bs = BaseDataset.from_path(path_to_my_raw_data, safe_mode=False,
#                            base_nan_filler='split', nan_filled_components=None)

# 2. Generate a classification dataset from stored data option
data_option = 'penguin_pce_full'
classification_DS = ClassificationDataset(name=data_option,
                                          in_components=['accX', 'accY', 'accZ'],
                                          out_components=['PCE'], in_chunk=[-25, 25], out_chunk=[0, 0],
                                          split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=-1,
                                          min_slice=100, scaling_method='z-norm', scaling_tag='in_only',
                                          split_method='instance-random')

# 4. Train your model with it (or something)
from torch import argmax
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
# from archs.MLP import Model as the_arch
from archs.FCN import Model as the_arch
import numpy as np


def get_model(train_device):
    my_network = the_arch(classification_DS.in_shape,
                          classification_DS.out_shape,
                          task_name='classification')
    my_network.to(train_device)
    return my_network


def get_train_setup(train_device, model):
    loss_fn = CrossEntropyLoss(weight=torch.Tensor([0.00972, 1.0]))
    loss_fn.to(train_device)
    optim = Adam(model.parameters(), lr=0.001)

    return optim, loss_fn


def log_results(tp, fp, fn, e_loss, e_count, loss, h_n, y):
    e_loss += loss
    pred = argmax(h_n, dim=1)
    tp += torch.count_nonzero(torch.logical_and(pred, y)).item()
    fp += torch.count_nonzero(torch.logical_and(pred, torch.logical_not(y))).item()
    fn += torch.count_nonzero(torch.logical_and(y, torch.logical_not(pred))).item()
    e_count += 1
    return tp, fp, fn, e_loss, e_count


def get_pr(tp, fp, fn):
    try:
        p = tp / (tp + fp)
    except:
        p = 0
    try:
        r = tp / (tp + fn)
    except:
        r = 0

    return p, r


def do_train_step(x, y, train_device, my_network, loss_fn, optim):
    x = x.to(train_device)
    y = y.to(train_device)
    optim.zero_grad()
    h_n = my_network.forward(x)
    loss = loss_fn(h_n, y)
    loss.backward()
    optim.step()
    return loss.item(), h_n.detach().cpu()


def do_valid_step(x, y, train_device, my_network, loss_fn):
    my_network.eval()
    x = x.to(train_device)
    y = y.to(train_device)
    h_n = my_network.forward(x)
    loss = loss_fn(h_n, y)
    my_network.train()
    return loss.item(), h_n.detach().cpu()


torch.manual_seed(classification_DS.seed)
train_loader = classification_DS.get_dataloader('train')
valid_loader = classification_DS.get_dataloader('valid')
my_network = get_model(train_device)
optim, loss_fn = get_train_setup(train_device, my_network)

for epoch in range(100):

    train_tp, train_fp, train_fn, train_e_loss, train_e_count = 0., 0., 0., 0., 0.
    for batch_id, batch in enumerate(train_loader):
        x = batch['x']
        y = batch['y']
        loss, h_n = do_train_step(x, y, train_device, my_network, loss_fn, optim)
        train_tp, train_fp, train_fn, train_e_loss, train_e_count = (
            log_results(train_tp, train_fp,
                        train_fn, train_e_loss,
                        train_e_count, loss,
                        h_n, y))
    train_p, train_r = get_pr(train_tp, train_fp, train_fn)

    valid_tp, valid_fp, valid_fn, valid_e_loss, valid_e_count = 0., 0., 0., 0., 0.
    for batch_id, batch in enumerate(valid_loader):
        x = batch['x']
        y = batch['y']
        loss, h_n = do_valid_step(x, y, train_device, my_network, loss_fn)
        valid_tp, valid_fp, valid_fn, valid_e_loss, valid_e_count = (
            log_results(valid_tp, valid_fp,
                        valid_fn, valid_e_loss,
                        valid_e_count, loss,
                        h_n, y))
    valid_p, valid_r = get_pr(valid_tp, valid_fp, valid_fn)

    print(' --- Epoch ' + str(epoch + 1) + ' --- ')
    print('train P-R: ' + str((train_p, train_r)))
    print('valid P-R: ' + str((valid_p, valid_r)))
    print('train loss: ' + str(train_e_loss / train_e_count))
    print('valid loss: ' + str(valid_e_loss / valid_e_count))

# ----------------------------------------------------------------------------------------------------------------------
# For a regression dataset
# ----------------------------------------------------------------------------------------------------------------------
# 1. Generate a data option of your raw dataframe
# from data.base_dataset import BaseDataset
# base_DS = BaseDataset('dummy_zero', mem_light=True)

# from regression_dataset import RegressionDataset
# data_option = 'dummy_zero'
# classification_DS = RegressionDataset(name=data_option,
#                                           in_components=['x1', 'x2', 'x3', 'x4'],
#                                           out_components=['y1', 'y2'], in_chunk=[-5, 5], out_chunk=[0, 0],
#                                           split_portions=(0.6, 0.2, 0.2), seed=666, batch_size=64, limit=None,
#                                           min_slice=10, scaling_method='zero-one', scaling_tag='full',
#                                           split_method='slice-random')
# dl = classification_DS.get_dataloader('train')

# ping = 0
