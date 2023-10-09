__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the prepared dataset class.'

import numpy as np

"""
---------------
PreparedDataset
---------------

The ``PreparedDataset`` represents a BaseDataset that is preprocessed for model training.
It inherits from BaseDataset.

It receives and stores the following variables:
- name (str): The name of the new dataset option.
- split_method (str): The method of splitting data. See below for details.
- split_portions (tuple): The approximate portions of the (train, valid, eval) splits.
    Note these portions are considered in combination with the split_method.
- scaling method (str): What method to use for scaling the data features. 
    See below for details.
- scaling mode (str): In what mode to scale the data.
    See below for details.
- batch_size (int): The mini-batch size for training.
- shuffle_train (bool): Whether the training set after every epoch.
- sampling_method (str): The batch sampling method to use. Currently only supports random.
- seed (int): The seed for reproducibility.
- limit (int): The number of samples to limit the data to. 
    Default: -1 = no limit.

--------------------
Splitting & Limiting
--------------------
The first step in preparing the dataset is to split it into a train-, validation-, 
and evaluation set (train, valid, eval) along with limiting it if applicable. 
This is done in the data_splitting.py script. 
More details can be found there, but we summarize the options here:
- `split_method` = 
    - 'random': Ignore all distinction between instances and slices, 
            and split on time steps randomly.
    - 'chronological': Ignore all distinction between instances and slices, 
            and split on time steps chronologically.
    - 'slice-random': Ignore all distinction between instances, 
            and split on slices randomly.
    - 'slice-chronological': Ignore all distinction between instances, 
            and split on slices chronologically.
    - 'instance-random': Split on instances randomly.
    - 'instance-chronological': Split on instances chronologically.
Note that the data is split ON the relevant level. 
I.e. If you split on instances and there are only 3, then split_portions=(0.6, 0.2, 0.2) 
will be a wild approximation i.t.o actual time steps.
    
Note that the data is limited prior to splitting, ignoring all instances and slices, 
and the data is limited by removing the excess data points from the end of the 
data block without any shuffling.

-------
Scaling
-------
After the data is split and limited, a scaler is fit to the train set data 
which will be applied to all data being extracted for model training.
This is done in the data_scaler.py script. 
More details can be found there, but we summarize the options here:
- `scaling_method` = 
    - 'z-norm': Input features are scaled by subtracting the mean and dividing by the std.
    - 'zero-one': Input features are scaled linearly to be in the range (0, 1).
    - 'none': No scaling occurs.
If the task is regression, then the scaling also applies to the output features.
- `scaling_mode` = 
    - 'data_feature': Features are scaled across time.
    - 'model_feature': Features are scaled per model input feature, not time.

"""

from numpy import (array, random)

from data.base_dataset import BaseDataset
from data.data_splitting import DataSplitter
from data.data_scaling import DataScaler
from helpers.logger import get_logger
from helpers.read_configs import load_from_path

logger = get_logger()


class PreparedDataset(BaseDataset):

    def __init__(self, **args):

        """Initialize a PreparedDataset object with the specified configuration."""

        # inherited from BaseDataset
        self.components = None
        self.data_path = None
        self.instances = None
        self.time_delta = None
        self.base_nan_filler = None
        self.nan_filled_components = None

        # required as arguments
        self.name = None
        self.in_components = None
        self.out_components = None
        self.in_chunk = None
        self.out_chunk = None
        self.split_portions = None
        self.seed = None
        self.batch_size = None

        # optional arguments
        self.split_method = None
        self.scaling_method = None
        self.scaling_tag = None
        self.shuffle_train = None
        self.limit = None
        self.padding_method = None
        self.min_slice = None

        # to be filled
        self.x_map = None
        self.y_map = None
        self.train_set_size = None
        self.valid_set_size = None
        self.eval_set_size = None
        self.selection = None
        self.x_scaler = None
        self.y_scaler = None
        self.in_shape = None
        self.out_shape = None

        for key, value in self.__required_prepared_meta().items():
            if key not in args:
                logger.error('Argument %s not provided for prepared dataset.', key)
                exit(101)
            if isinstance(args[key], value):
                setattr(self, key, args[key])
            else:
                logger.error('Provided %s should be of type %s.', key,
                             str(value))
                exit(101)

        super().__init__(args['name'], mem_light=False)

        # 2. Save default optional arguments
        self.__setattr_or_default(args, 'split_method', 'chronological')
        self.__setattr_or_default(args, 'scaling_method', 'z-norm')
        self.__setattr_or_default(args, 'scaling_tag', None)
        self.__setattr_or_default(args, 'shuffle_train', True)
        self.__setattr_or_default(args, 'limit', -1)
        self.__setattr_or_default(args, 'padding_method', 'zero')
        self.__setattr_or_default(args, 'min_slice', None)

        # 4. Initiate the data preparation
        random.seed(self.seed)
        self.__prepare()

    def get_the_data(self):
        if hasattr(self, 'the_data'):
            return self.the_data
        else:
            return load_from_path(self.data_path)['the_data']

    def __setattr_or_default(self, args, name, default):
        if name in args:
            setattr(self, name, args[name])
        else:
            setattr(self, name, default)

    def __prepare(self):

        """Prepare the underlying BaseDataset by splitting and scaling the data.
        Note that this is not done directly on the data. The splits are defined in the
        'selection' variable in a selection matrix."""

        logger.info('Preparing overhead.')
        self.y_map = array([i for i, e in enumerate(self.components)
                            if e in self.out_components])
        self.x_map = array([i for i, e in enumerate(self.components)
                            if e in self.in_components])
        self.in_shape = (self.in_chunk[1] - self.in_chunk[0] + 1, len(self.in_components))
        self.out_shape = (self.out_chunk[1] - self.out_chunk[0] + 1, len(self.out_components))

        logger.info('Preparing data splits (selection).')
        self.selection = DataSplitter(self.get_the_data(), self.split_method,
                                      self.split_portions, self.instances,
                                      self.limit, self.y_map, self.out_chunk,
                                      self.min_slice).get_selection()
        self.train_set_size = len(self.selection['train'])
        self.valid_set_size = len(self.selection['valid'])
        self.eval_set_size = len(self.selection['eval'])
        logger.info('Data split sizes: ' + str((self.train_set_size, self.valid_set_size, self.eval_set_size)))


        logger.info('Preparing data scalers, if relevant.')
        self.x_scaler, self.y_scaler = DataScaler(self.extract_dataset('train'),
                                                  self.scaling_method,
                                                  self.scaling_tag).get_scalers()

        if hasattr(self, 'the_data'):
            delattr(self, 'the_data')

    def extract_dataset(self, set_tag):

        """Extracts the relevant samples for one of the data splits, scale them, and package them."""

        the_data = self.get_the_data()

        # [sample][time steps][features]
        x_vals, y_vals = self.__fast_extract(the_data, set_tag)

        if self.x_scaler:
            x_vals = self.x_scaler.transform(x_vals)
        if self.y_scaler:
            y_vals = self.y_scaler.transform(y_vals)

        return {'x': x_vals, 'y': y_vals}

    def __fast_extract(self, the_data, set_tag):

        selection = self.selection[set_tag]
        relevant_slice_indx = np.unique(selection[:, :2], axis=0)
        max_pad = max(abs(array(self.in_chunk))) + 1

        padded_slices = {}
        for r in relevant_slice_indx:
            i = self.instances[r[0]]
            s = r[1]
            padded_slices[tuple(r)] = the_data[i][s]['d'][:, self.x_map]
            padded_slices[tuple(r)] = self.__do_padding(padded_slices[tuple(r)],
                                                        'backward',
                                                        self.padding_method,
                                                        len(padded_slices[tuple(r)]) + max_pad)
            padded_slices[tuple(r)] = self.__do_padding(padded_slices[tuple(r)],
                                                        'forward',
                                                        self.padding_method,
                                                        len(padded_slices[tuple(r)]) + max_pad)

        x_vals, y_vals = PreparedDataset.__parr_sample(selection, self.in_chunk, self.out_chunk,
                                                       max_pad, padded_slices, the_data,
                                                       self.instances, self.y_map)

        return x_vals, y_vals

    @staticmethod
    def __parr_sample(selection, in_chunk, out_chunk, max_pad, padded_slices, the_data,
                      instances, y_map):

        def sample_blocks(t, chunk, pad):
            x_blocks = np.arange(chunk[0] + pad, chunk[1] + pad + 1)
            x_blocks = np.expand_dims(x_blocks, axis=0)
            t_selection = t
            x_blocks = np.repeat(x_blocks, len(t_selection), axis=0)
            t_selection = np.expand_dims(t_selection, axis=1)
            t_selection = np.repeat(t_selection, x_blocks.shape[1], axis=1)
            x_blocks += t_selection
            return x_blocks

        x_blocks = sample_blocks(selection[:, 2], in_chunk, max_pad)
        y_blocks = sample_blocks(selection[:, 2], out_chunk, 0)
        x_vals = []
        y_vals = []
        s_indx = 0
        for s in selection:
            x_vals.append(padded_slices[(s[0], s[1])][x_blocks[s_indx]])
            y_vals.append(the_data[instances[s[0]]][s[1]]['d'][y_blocks[s_indx], y_map])
            s_indx += 1
        x_vals = array(x_vals)
        y_vals = array(y_vals)

        return x_vals, y_vals

    @staticmethod
    def __do_padding(vals, direction, method, cap):

        if direction == 'backward':
            pw = ((cap - vals.shape[0], 0), (0, 0))
        elif direction == 'forward':
            pw = ((0, cap - vals.shape[0]), (0, 0))
        else:
            logger.error('Unknown padding direction %s.', direction)
            exit(101)

        pw = np.array(pw)
        if method == 'zero':
            ret_vals = np.pad(vals,
                              pad_width=pw,
                              mode='constant')
        else:
            ret_vals = np.pad(vals,
                              pad_width=pw,
                              mode=method)

        return ret_vals

    @staticmethod
    def __required_prepared_meta():

        return {'name': (str,),
                'in_components': (list,),
                'out_components': (list,),
                'in_chunk': (list, tuple),
                'out_chunk': (list, tuple),
                'split_portions': (list, tuple),
                'seed': (int,),
                'batch_size': (int,)}




