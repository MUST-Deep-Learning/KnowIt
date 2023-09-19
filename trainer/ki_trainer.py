__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_trainer module.'

import numpy as np

from data.classification_dataset import ClassificationDataset

# ---------------------DEBUGGING----------------------------------------------------------------------------------------

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # if true => very slow
train_device = torch.device('cuda')
torch.seed()

DS = ClassificationDataset(name='penguin_dummy', split_method='slice-random',
                     split_portions=(0.6, 0.2, 0.2),
                     scaling_method='z-norm', sampling_method='random',
                     seed=1, scaling_mode='data_feature',
                     batch_size=32, shuffle_train=True, limit=50000)

torch.manual_seed(DS.seed)

train_loader = DS.get_dataloader('train')
valid_loader = DS.get_dataloader('valid')

from torch.optim import SGD, Adam, RAdam
from torch.nn import MSELoss, CrossEntropyLoss
from model.custom_model_TEMPLATE import Model
in_dim = int(np.prod(DS.in_shape))
out_dim = int(len(DS.class_set))
my_mlp = Model(in_dim, out_dim, 'classification',
               width=256, depth=3, dropout=0.3, batchnorm=False)
my_mlp.to(train_device)
optimizer = Adam(my_mlp.parameters(), lr=0.0001)
loss_fn = CrossEntropyLoss()
max_epoch = 500

epoch_losses = []
for epoch in range(max_epoch):
    train_batch_losses = []
    train_tp = []
    train_fp = []
    train_fn = []

    for batch_id, batch in enumerate(train_loader):
        x = batch['x']
        y = batch['y']
        x = x.view(DS.batch_size, in_dim)
        x = x.to(train_device)
        y = y.to(train_device)
        optimizer.zero_grad()
        h_n = my_mlp.forward(x)
        loss = loss_fn(h_n, y)
        loss.backward()
        optimizer.step()
        train_batch_losses.append(loss.item())

        pred = torch.argmax(h_n, dim=1)
        train_tp.extend(torch.logical_and(pred == 1, y == 1).cpu().detach().numpy())
        train_fp.extend(torch.logical_and(pred == 1, y == 0).cpu().detach().numpy())
        train_fn.extend(torch.logical_and(pred == 0, y == 1).cpu().detach().numpy())

    train_batch_losses = np.array(train_batch_losses)
    train_epoch_loss = (np.mean(train_batch_losses), np.std(train_batch_losses))

    my_mlp.eval()
    valid_batch_losses = []

    train_tp = np.count_nonzero(np.array(train_tp))
    train_fp = np.count_nonzero(np.array(train_fp))
    train_fn = np.count_nonzero(np.array(train_fn))

    try:
        precision = train_tp / (train_tp + train_fp)
    except:
        precision = -666.
    try:
        recall = train_tp / (train_tp + train_fn)
    except:
        recall = -666.

    print('P: ' + str(precision) + ' R: ' + str(recall))

    # valid_tp = []
    # valid_fp = []
    # valid_fn = []


    for batch_id, batch in enumerate(valid_loader):
        x = batch['x']
        y = batch['y']
        x = x.view(x.shape[0], in_dim)
        x = x.to(train_device)
        y = y.to(train_device)
        h_n = my_mlp.forward(x)
        loss = loss_fn(h_n, y)
        valid_batch_losses.append(loss.item())

        # pred = torch.argmax(h_n, dim=1)
        # tp.extend(torch.logical_and(pred == 1, y == 1).cpu().detach().numpy())
        # fp.extend(torch.logical_and(pred == 1, y == 0).cpu().detach().numpy())
        # fn.extend(torch.logical_and(pred == 0, y == 1).cpu().detach().numpy())

    # tp = np.count_nonzero(np.array(tp))
    # fp = np.count_nonzero(np.array(fp))
    # fn = np.count_nonzero(np.array(fn))
    #
    # try:
    #     precision = tp / (tp + fp)
    # except:
    #     precision = -666.
    # try:
    #     recall = tp / (tp + fn)
    # except:
    #     recall = -666.
    #
    # print('P: ' + str(precision) + ' R: ' + str(recall))

    valid_batch_losses = np.array(valid_batch_losses)
    valid_epoch_loss = (np.mean(valid_batch_losses), np.std(valid_batch_losses))
    my_mlp.train()

    # print(str(epoch) + '.   Train: ' + str(train_epoch_loss[0]) + ', Valid: ' + str(valid_epoch_loss[0]))
    epoch_losses.append([train_epoch_loss[0], valid_epoch_loss[0]])

epoch_losses = np.array(epoch_losses)
import matplotlib.pyplot as plt

plt.plot(epoch_losses[:, 0], label='train MSE', color='green', alpha=0.7)
plt.plot(epoch_losses[:, 1], label='valid MSE', color='red', alpha=0.7)
plt.legend()
plt.xlabel('Epoch')
plt.show()
plt.close()


exit(101)

# ---------------------DEBUGGING----------------------------------------------------------------------------------------