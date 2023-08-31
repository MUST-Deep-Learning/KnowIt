__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the prepared dataset class.'

import numpy as np

from data.base_dataset import BaseDataset
from data.data_splitter import get_target_splits
from helpers.logger import get_logger

logger = get_logger()

class PreparedDataset(BaseDataset):

    def __init__(self, name: str,
                 split_method: str,
                 split_portions: tuple,
                 scaling_method: str,
                 sampling_method: str):

        super().__init__(name, 'option')

        self.split_method = split_method
        self.split_portions = split_portions
        self.scaling_method = scaling_method
        self.sampling_method = sampling_method
        self.prepare()

    def prepare(self):

        # split data
        self.train_select, self.valid_select, self.eval_select = (
            get_target_splits(self.the_data, self.split_method,
                              self.split_portions, self.instances,
                              self.num_target_timesteps))
        self.train_set_size = self.train_select.shape[0]
        self.valid_set_size = self.valid_select.shape[0]
        self.eval_set_size = self.eval_select.shape[0]

        # scale data
        train_set = self.get_minibatch('train', 512)


    def get_minibatch(self, set_tag, mb_size):

        if self.sampling_method == 'random':
            sampling = np.random.choice(np.arange(0, self.train_set_size), 512)
            sampling = self.train_select[sampling, :]
            for s in sampling:
                y = self.the_data[self.instances[s[0]]][s[1]]['y']
                y = self.the_data[self.instances[s[0]]][s[1]]['y'][s[2]-self.out_chunk[0]:s[2]-self.out_chunk[1]+1, :]
                x = self.the_data[self.instances[s[0]]][s[1]]['x'][s[2]-self.in_chunk[0]:s[2]-self.in_chunk[1]+1, :]
                ping = 0

        ping = 0





DS = PreparedDataset(name='dummy_zero', split_method='slice-random',
                     split_portions=(0.6, 0.2, 0.2),
               scaling_method='z', sampling_method='random')
ping = 0

