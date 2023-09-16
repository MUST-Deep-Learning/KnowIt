__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the prepared dataset class.'

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy

from data.base_dataset import BaseDataset
from data.data_splitter import get_target_splits
from data.data_scaler import get_scaler
from helpers.logger import get_logger
from helpers.read_configs import load_from_path

logger = get_logger()


class PreparedDataset(BaseDataset):

    def __init__(self, name: str,
                 split_method: str,
                 split_portions: tuple,
                 scaling_method: str,
                 sampling_method: str,
                 seed: int,
                 scaling_mode: str,
                 batch_size: int,
                 shuffle_train: bool):

        super().__init__(name, 'option')

        self.split_method = split_method
        self.split_portions = split_portions
        self.scaling_method = scaling_method
        self.scaling_mode = scaling_mode
        self.sampling_method = sampling_method
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        np.random.seed(seed)
        self.prepare()

    def prepare(self):

        # split data
        train_select, valid_select, eval_select = (
            get_target_splits(self.the_data, self.split_method,
                              self.split_portions, self.instances,
                              self.num_target_timesteps, self.seed))
        self.train_set_size = train_select.shape[0]
        self.valid_set_size = valid_select.shape[0]
        self.eval_set_size = eval_select.shape[0]
        self.selection = {'train': train_select, 'valid': valid_select, 'eval': eval_select}

        # get_scaler
        self.x_scaler, self.y_scaler = get_scaler(self.the_data, self.selection,
                                                  self.instances, self.in_chunk,
                                                  self.out_chunk, self.task,
                                                  mode=self.scaling_mode,
                                                  method=self.scaling_method)

        # define io dimensions
        self.in_shape = (self.in_chunk[1] - self.in_chunk[0] + 1, len(self.input_components))
        self.out_shape = (self.out_chunk[1] - self.out_chunk[0] + 1, len(self.target_components))

        delattr(self, 'the_data')

    def get_dataloader(self, set_tag):
        dataset = Customdataset(**self.extract_dataset(set_tag))

        drop_last = False
        if set_tag == 'train':
            drop_last = True

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle_train, drop_last=drop_last)
        return dataloader

    def extract_dataset(self, set_tag):

        the_data = self.get_the_data()

        # [sample][timesteps][features]
        x_vals = np.array([self.extract_sample(s[0], s[1], s[2], 'x', the_data) for s in self.selection[set_tag]])
        y_vals = np.array([self.extract_sample(s[0], s[1], s[2], 'y', the_data) for s in self.selection[set_tag]])

        x_vals = self.x_scaler.transform(x_vals)
        y_vals = self.y_scaler.transform(y_vals)

        return {'x': x_vals, 'y': y_vals}

    def extract_sample(self, i, s, t, io, the_data):
        if io == 'y':
            return the_data[self.instances[i]][s][io][t + self.out_chunk[0]:t + self.out_chunk[1] + 1, :]
        elif io == 'x':
            return the_data[self.instances[i]][s][io][t + self.in_chunk[0]:t + self.in_chunk[1] + 1, :]
        else:
            logger.error('Unknown io type for sample extraction %s.', io)
            exit(101)

    def get_the_data(self):
        try:
            return load_from_path(self.data_path)['the_data']
        except:
            exit(101)


class Customdataset(Dataset):
    """Dataset format for torch."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        input_x = self.x[idx, :, :]
        output_y = self.y[idx, :, :]
        input_x = input_x.astype('float')
        output_y = output_y.astype('float')
        output_y = output_y.squeeze()
        sample = {'x': from_numpy(input_x).float(),
                  'y': from_numpy(output_y).float()}
        return sample

# ---------------------DEBUGGING----------------------------------------------------------------------------------------
# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False  # if true => very slow
# train_device = torch.device('cuda')
# torch.seed()
#
# DS = PreparedDataset(name='dummy_zero', split_method='slice-random',
#                      split_portions=(0.6, 0.2, 0.2),
#                      scaling_method='z-norm', sampling_method='random',
#                      seed=1, scaling_mode='data_feature',
#                      batch_size=32, shuffle_train=True)
# torch.manual_seed(DS.seed)
#
# train_loader = DS.get_dataloader('train')
# from torch.optim import SGD, Adam
# from torch.nn import MSELoss
# from model.custom_model_TEMPLATE import Model
# in_dim = int(np.prod(DS.in_shape))
# out_dim = int(np.prod(DS.out_shape))
# my_mlp = Model(in_dim, out_dim, 'regression', width=512, depth=4, dropout=0.)
# my_mlp.to(train_device)
# optimizer = Adam(my_mlp.parameters(), lr=0.0005)
# loss_fn = MSELoss()
#
# for epoch in range(500):
#
#     losses = []
#     for batch_id, batch in enumerate(train_loader):
#
#         x = batch['x']
#         y = batch['y']
#         x = x.view(DS.batch_size, in_dim)
#
#         x = x.to(train_device)
#         y = y.to(train_device)
#
#         # forward pass
#         optimizer.zero_grad()
#         h_n = my_mlp.forward(x)
#         # compute and store batch loss
#         loss = loss_fn(h_n, y)
#         # compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # update parameters linked to optimizer
#         optimizer.step()
#
#         losses.append(loss.item())
#
#     losses = np.array(losses)
#     print(str(epoch) + '.   ' + str((np.mean(losses), np.std(losses))))
#
# exit(101)


# ---------------------DEBUGGING----------------------------------------------------------------------------------------


