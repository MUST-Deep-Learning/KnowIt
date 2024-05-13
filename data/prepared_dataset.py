__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the PreparedDataset class for Knowit.'

"""
------------------
PreparedDataset
------------------

The ``PreparedDataset`` represents a BaseDataset that is preprocessed for model training.
It inherits from BaseDataset. Based on the provided `name' variable, it will populate the 
parent BaseDataset's variables. 

--------------------
Prediction points
--------------------

In order to define a PreparedDataset, the input-output dynamics of the model to be trained 
must defined rather precisely. The PreparedDataset is built on the idea of 'prediction points'.
Each time step in the BaseDataset can be regarded a prediction point (under some assumptions). 
At every prediction point, a model is to predict a specific set of features-over-time from 
a specific other set of features-over-time. The specifics are defined as follows.

    -   in_components (list): A subset of the 'components' variable of the BaseDataset.
            These are the components that will be used as input to the model.
    -   out_components (list): A subset of the 'components' variable of the BaseDataset.
            These are the components that will be used as output to the model.
    -   in_chunk (list): A list of two integers [a, b] for which a <= b. This defines the 
            time steps (of in_components) to be used for prediction at point 
            t as [t + a, t + b].
    -   out_chunk (list): A list of two integers [a, b] for which a <= b. This defines the 
            time steps (of out_components) to be predicted at prediction point 
            t as [t + a, t + b].
            
Note that this might seem cumbersome, but it allows us to easily define many different types of tasks:
    
    -   regression (heartrate from 11 millisecond window given three instantaneous biometrics)
            in_components = [biometric1, biometric2, biometric2]
            out_components = [heart rate]
            in_chunk = [-5, 5]
            out_chunk = [0, 0]
    - autoregressive univariate forcasting (predict a stock's value in 5 days given last 20 days)
            in_components = [stock,]
            out_components = [stock,]
            in_chunk = [-20, 0]
            out_chunk = [5, 5]
    - time series classification (detect a whale call given an audio recording of 5 seconds)
            in_components = [sound,]
            out_components = [positive_call,]
            in_chunk = [-2, 2]
            out_chunk = [0, 0]


--------------------
Input arguments
--------------------

In addition to the dataset name and prediction point dynamics defined above, the following variables 
should also be provided.

- name (str): The name of the new dataset option.
    - split_portions (tuple): The approximate portions of the (train, valid, eval) splits.
        Note these portions are considered in combination with the split_method.
    - seed (int): The seed for reproducibility.
    - batch_size (int): The mini-batch size for training.
    
    - shuffle_train (bool, optional): Whether the training set is shuffled after every epoch.
        Default: True
    - split_method (str, optional): The method of splitting data. See below for details.
        Default: 'chronological'
    - limit (int, optional): The number of instances / slices / time steps to limit 
        the data to. This depends on the split_method.
        Default: None = no limit.
    - scaling_method (str, optional): What method to use for scaling the data features. 
        See below for details. Default: z-norm
    - scaling_tag (str, optional): In what mode to scale the data. (in_only, full, None)
        See below for details. Default: None
    - padding_method (str, optional): What method to pad model inputs will.
        See below for details. Default: zero
    - min_slice (str, optional): The minimum slice size to consider 
        during data splitting / selection. Default: None = Consider all slices


-----------------------
Splitting & Limiting
-----------------------

The first step in preparing the dataset is to split it into a train-, validation-, 
and evaluation set (train, valid, eval) along with limiting it if applicable. 
This is done with the DataSplitter module.  More details can be found there, 
but we summarize the options here:
- `split_method` = 
    - 'random': Ignore all distinction between instances and slices, 
            and split on time steps randomly.
    - 'chronological' (default): Ignore all distinction between instances and slices, 
            and split on time steps chronologically.
    - 'slice-random': Ignore all distinction between instances, 
            and split on slices randomly.
    - 'slice-chronological': Ignore all distinction between instances, 
            and split on slices chronologically.
    - 'instance-random': Split on instances randomly.
    - 'instance-chronological': Split on instances chronologically.
Note that the data is split ON the relevant level (instance, slice, or timesteps). 
I.e. If you split on instances and there are only 3 instances, then split_portions=(0.6, 0.2, 0.2) 
will be a wild approximation i.t.o actual time steps.
    
Note that the data is limited during splitting, and the data is limited by removing 
the excess data points from the end of the data block after shuffling or ordering according to time. 
Also note that if the data is limited too much for a given split_portion to have a single entry,
 an error will occur confirming it.


----------
Scaling
----------

After the data is split and limited, a scaler is fit to the train set data 
which will be applied to all data being extracted for model training.
This is done with the DataScaler module. 
More details can be found there, but we summarize the options here:
- `scaling_method`
    - 'z-norm': Features are scaled by subtracting the mean and dividing by the std.
    - 'zero-one': Input features are scaled linearly to be in the range (0, 1).
    - None: No scaling occurs.
If the task is regression, then the scaling also applies to the output features.
- `scaling_tag`
    - 'in_only': Only the input features will be scaled.
    - 'full': The in and output features will be scaled.
    - None: No features will be scaled.


----------
Padding
----------
At some prediction points the `in_chunk' or 'out_chunk' might exceed the corresponding 
slice range. In these cases we pad the input values. The output values are never padded.
Prediction points that do not have valid output values for an 'out_chunk' are not selected 
during data splitting. The argument `padding_method' is the same as 'mode' in the numpy.pad 
function (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).

"""

# external imports
from numpy import (array, random, unique, arange,
                   expand_dims, repeat, pad)

# internal imports
from data.base_dataset import BaseDataset
from data.data_splitting import DataSplitter
from data.data_scaling import DataScaler
from helpers.logger import get_logger

logger = get_logger()


class PreparedDataset(BaseDataset):

    def __init__(self, **args):

        """ Instantiate a PreparedDataset object with the specified configuration. """

        # inherited from BaseDataset
        self.components = None
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

        # to be filled automatically
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

        # check that all the required variables are given in the right format
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

        logger.info('Initializing PreparedClass for %s', args['name'])

        # initialize the BaseClass variables while keeping the_data in memory
        super().__init__(args['data_path'], mem_light=False)

        # Save default optional arguments
        self.__setattr_or_default(args, 'split_method', 'chronological')
        self.__setattr_or_default(args, 'scaling_method', 'z-norm')
        self.__setattr_or_default(args, 'scaling_tag', None)
        self.__setattr_or_default(args, 'shuffle_train', True)
        self.__setattr_or_default(args, 'limit', None)
        self.__setattr_or_default(args, 'padding_method', 'zero')
        self.__setattr_or_default(args, 'min_slice', None)

        # Initiate the data preparation
        random.seed(self.seed)
        self.__prepare()

    def __setattr_or_default(self, args: dict, name: str, default: object):
        """ Set object attribute with given (if given) or given default. """

        if name in args:
            setattr(self, name, args[name])
        else:
            setattr(self, name, default)

    def __prepare(self):

        """Prepare the dataset by splitting and scaling the data.
        Note that this is not done directly on the data. The splits are defined in the
        'selection' variable in a selection matrix. """

        logger.info('Preparing overhead.')
        missing_in_components = set(self.in_components) - set(self.components)
        if len(missing_in_components) > 0:
            logger.error('Defined in_components %s not in data option.',
                         str(missing_in_components))
            exit(101)
        missing_out_components = set(self.out_components) - set(self.components)
        if len(missing_out_components) > 0:
            logger.error('Defined out_components %s not in data option.',
                         str(missing_out_components))
            exit(101)
        self.y_map = array([i for i, e in enumerate(self.components)
                            if e in self.out_components])
        self.x_map = array([i for i, e in enumerate(self.components)
                            if e in self.in_components])
        # self.in_shape = (self.in_chunk[1] - self.in_chunk[0] + 1, len(self.in_components))
        # self.out_shape = (self.out_chunk[1] - self.out_chunk[0] + 1, len(self.out_components))
        self.in_shape = [self.in_chunk[1] - self.in_chunk[0] + 1, len(self.in_components)]
        self.out_shape = [self.out_chunk[1] - self.out_chunk[0] + 1, len(self.out_components)]

        logger.info('Preparing data splits (selection).')
        self.selection = DataSplitter(self.get_the_data(), self.split_method,
                                      self.split_portions, self.instances,
                                      self.limit, self.y_map, self.out_chunk,
                                      self.min_slice).get_selection()
        self.train_set_size = len(self.selection['train'])
        self.valid_set_size = len(self.selection['valid'])
        self.eval_set_size = len(self.selection['eval'])
        logger.info('Data split sizes: ' + str((self.train_set_size,
                                                self.valid_set_size,
                                                self.eval_set_size)))

        logger.info('Preparing data scalers, if relevant.')
        self.x_scaler, self.y_scaler = DataScaler(self.extract_dataset('train'),
                                                  self.scaling_method,
                                                  self.scaling_tag).get_scalers()

        if hasattr(self, 'the_data'):
            delattr(self, 'the_data')
            logger.info('the_data structure is removed from memory.')

    def get_ist_values(self, set_tag: str):

        the_data = self.get_the_data()

        ist_values = []
        for p in self.selection[set_tag]:
            i = self.instances[p[0]]
            s = p[1]
            t = the_data[i][s]['t'][p[2]]
            ist_values.append((i, s, t))
        return ist_values

    def extract_dataset(self, set_tag: str):

        """Extracts the relevant samples for one of the data splits, scale them, and package them. """

        # TODO: This method is very heavy on memory if the dataset is large.
        #  Will need to find alternative where we do not put the entire split in memory.

        the_data = self.get_the_data()

        # [sample][time steps][features]
        x_vals, y_vals = self.__fast_extract(the_data, set_tag)

        if self.x_scaler:
            x_vals = self.x_scaler.transform(x_vals)
        if self.y_scaler:
            y_vals = self.y_scaler.transform(y_vals)

        return {'x': x_vals, 'y': y_vals}

    def __fast_extract(self, the_data: dict, set_tag: str):

        """ Pad relevant slices. And sample the input values (padded)
        and output values (not padded). """

        # get overhead
        selection = self.selection[set_tag]
        relevant_slice_indx = unique(selection[:, :2], axis=0)
        max_pad = max(abs(array(self.in_chunk))) + 1

        # only pad the relevant slices
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
        # sample input and output values
        x_vals, y_vals = PreparedDataset.__parr_sample(selection, self.in_chunk, self.out_chunk,
                                                       max_pad, padded_slices, the_data,
                                                       self.instances, self.y_map)

        return x_vals, y_vals

    @staticmethod
    def __parr_sample(selection: array, in_chunk: list, out_chunk: list, max_pad: int,
                      padded_slices: dict, the_data: dict,
                      instances: list, y_map: array):

        """
        Sample data blocks for parallel processing based on selection indices.

        This function samples data blocks from the provided inputs according to the selection indices. It samples blocks
        from the input data and the target data for parallel processing.

        Parameters:
            selection (array-like): Selection indices specifying the data blocks to sample.
            in_chunk (tuple): Input data chunk size.
            out_chunk (tuple): Target data chunk size.
            max_pad (int): Maximum padding for input data blocks.
            padded_slices (dict): Dictionary containing padded slices of the input data.
            the_data (list): List containing the input data instances.
            instances (list): List of instances related to the input data.
            y_map (array): Mapping for target data components.

        Returns:
            x_vals (array-like): Sampled input data blocks.
            y_vals (array-like): Sampled target data blocks.

        Internal Function:
        sample_blocks(t, chunk, pad):
            Helper function for sampling blocks of data along a given axis.

        The function samples data blocks from the specified inputs based on the selection indices, and returns the
        sampled input and target data as arrays.
        """

        def sample_blocks(t, chunk, pad):
            blocks = arange(chunk[0] + pad, chunk[1] + pad + 1)
            blocks = expand_dims(blocks, axis=0)
            t_selection = t
            blocks = repeat(blocks, len(t_selection), axis=0)
            t_selection = expand_dims(t_selection, axis=1)
            t_selection = repeat(t_selection, blocks.shape[1], axis=1)
            blocks += t_selection
            return blocks

        x_blocks = sample_blocks(selection[:, 2], in_chunk, max_pad)
        y_blocks = sample_blocks(selection[:, 2], out_chunk, 0)
        x_vals = []
        y_vals = []
        s_indx = 0
        # TODO: Here is still a bottleneck. But at least it is just sampling now.
        for s in selection:
            x_vals.append(padded_slices[(s[0], s[1])][x_blocks[s_indx]])
            # y_vals.append(the_data[instances[s[0]]][s[1]]['d'][y_blocks[s_indx], y_map])
            y_vals.append(the_data[instances[s[0]]][s[1]]['d'][y_blocks[s_indx]][:, y_map])

            s_indx += 1
        x_vals = array(x_vals)
        y_vals = array(y_vals)

        return x_vals, y_vals

    @staticmethod
    def __do_padding(vals: array, direction: str, method: str, cap: int):

        """ Pad vals = [timesteps, features] using the
        given method in the given direction up to the given size (cap).
        See https://numpy.org/doc/stable/reference/generated/numpy.pad.html."""

        if direction == 'backward':
            pw = ((cap - vals.shape[0], 0), (0, 0))
        elif direction == 'forward':
            pw = ((0, cap - vals.shape[0]), (0, 0))
        else:
            logger.error('Unknown padding direction %s.', direction)
            exit(101)

        pw = array(pw)
        if method == 'zero':
            ret_vals = pad(vals,
                              pad_width=pw,
                              mode='constant')
        else:
            ret_vals = pad(vals,
                              pad_width=pw,
                              mode=method)

        return ret_vals

    @staticmethod
    def __required_prepared_meta():

        """ These are the variables (and their formats) that need to be given
        when creating a new PreparedDataset object. """

        return {'name': (str,),
                'in_components': (list,),
                'out_components': (list,),
                'in_chunk': (list, tuple),
                'out_chunk': (list, tuple),
                'split_portions': (list, tuple),
                'seed': (int,),
                'batch_size': (int,)}




