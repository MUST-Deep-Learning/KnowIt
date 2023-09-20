__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the prepared dataset class.'

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
This is done in the data_splitter.py script. 
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

from numpy import array
from numpy import random

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
                 scaling_mode: str,
                 batch_size: int,
                 shuffle_train: bool,
                 sampling_method: str,
                 seed: int,
                 limit: int = -1):

        """Initialize a PreparedDataset object with the specified configuration."""

        # 1. Init super class
        super().__init__(name, 'option')

        # 2. Save given arguments
        self.split_method = split_method
        self.split_portions = split_portions
        self.scaling_method = scaling_method
        self.scaling_mode = scaling_mode
        self.seed = seed
        self.limit = limit

        # TODO: These three are not used in PreparedDataset but in child classes
        # RegressionDataset and ClassificationDataset. Should think if we want to move them.
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.sampling_method = sampling_method

        # 3. Additional variables to be filled dynamically
        self.train_set_size = None
        self.valid_set_size = None
        self.eval_set_size = None
        self.selection = None
        self.x_scaler = None
        self.y_scaler = None
        self.in_shape = None
        self.out_shape = None

        # 4. Initiate the data preparation
        random.seed(seed)
        self.prepare()

    def prepare(self):

        """Prepare the underlying BaseDataset by splitting and scaling the data.
        Note that this is not done directly on the data. The splits are defined in the
        'selection' variable in a selection matrix."""

        # 1. split data
        train_select, valid_select, eval_select = (
            get_target_splits(self.the_data, self.split_method,
                              self.split_portions, self.instances,
                              self.num_target_timesteps, self.seed, self.limit))
        self.train_set_size = train_select.shape[0]
        self.valid_set_size = valid_select.shape[0]
        self.eval_set_size = eval_select.shape[0]
        self.selection = {'train': train_select, 'valid': valid_select, 'eval': eval_select}

        # 2. get_scaler
        self.x_scaler, self.y_scaler = get_scaler(self.the_data, self.selection,
                                                  self.instances, self.in_chunk,
                                                  self.out_chunk, self.task,
                                                  mode=self.scaling_mode,
                                                  method=self.scaling_method)

        # 3. define io dimensions
        self.in_shape = (self.in_chunk[1] - self.in_chunk[0] + 1, len(self.input_components))
        self.out_shape = (self.out_chunk[1] - self.out_chunk[0] + 1, len(self.target_components))

        # 4. Remove the datastructure
        delattr(self, 'the_data')

    def extract_dataset(self, set_tag):

        """Extracts the relevant samples for one of the data splits, scale them, and package them."""

        the_data = self.get_the_data()

        # [sample][time steps][features]
        x_vals = array([self.extract_sample(s[0], s[1], s[2], 'x', the_data) for s in self.selection[set_tag]])
        y_vals = array([self.extract_sample(s[0], s[1], s[2], 'y', the_data) for s in self.selection[set_tag]])

        x_vals = self.x_scaler.transform(x_vals)
        y_vals = self.y_scaler.transform(y_vals)

        return {'x': x_vals, 'y': y_vals}

    def extract_sample(self, i, s, t, io, the_data):

        """Extracts the relevant sample as defined from the selection matrix."""

        if io == 'y':
            return the_data[self.instances[i]][s][io][t + self.out_chunk[0]:t + self.out_chunk[1] + 1, :]
        elif io == 'x':
            return the_data[self.instances[i]][s][io][t + self.in_chunk[0]:t + self.in_chunk[1] + 1, :]
        else:
            logger.error('Unknown io type for sample extraction %s.', io)
            exit(101)

    def get_the_data(self):
        return load_from_path(self.data_path)['the_data']







