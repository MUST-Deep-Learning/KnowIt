"""
------------------
PreparedDataset
------------------

The ``PreparedDataset`` represents a ``BaseDataset`` that is preprocessed for model training.
It inherits from ``BaseDataset``. Based on the provided data path, it will populate the
parent's variables.

-----------------
Prediction points
-----------------

In order to define a PreparedDataset, the input-output dynamics of the model to be trained
must be defined rather precisely. The ``PreparedDataset`` is built on the idea of 'prediction points'.
Each time step in the ``BaseDataset`` can be regarded a prediction point (under some assumptions).
At every prediction point, a model is to predict a specific set of features-over-time from
a specific other set of features-over-time. The specifics are defined as follows.

    in_components : list
        A subset of ``BaseDataset.components`` representing the components that will be used as input to the model.
    out_components : list
        A subset of ``BaseDataset.components`` representing the components that will be used as output to the model.
    in_chunk : list
        A list of two integers [a, b] for which a <= b defining the time steps (of in_components)
        to be used for prediction at point t as [t + a, t + b].
    out_chunk : list
        A list of two integers [a, b] for which a <= b defining the time steps (of out_components)
        to be predicted at prediction point t as [t + a, t + b].

Note that this might seem cumbersome, but it allows us to easily define several different types of tasks.
For example:

- regression (heartrate from an 11-millisecond window given three instantaneous biometrics)
    - in_components = [biometric1, biometric2, biometric2]
    - out_components = [heart rate]
    - in_chunk = [-5, 5]
    - out_chunk = [0, 0]

- autoregressive univariate forcasting (predict a stock's value in 5 days given last 20 days)
    - in_components = [stock,]
    - out_components = [stock,]
    - in_chunk = [-20, 0]
    - out_chunk = [5, 5]
- time series classification (detect a whale call given an audio recording of 5 seconds)
    - in_components = [sound,]
    - out_components = [positive_call,]
    - in_chunk = [-2, 2]
    - out_chunk = [0, 0]

--------------------
Splitting & Limiting
--------------------

The first step in preparing the dataset is to split it into a train-, validation-,
and evaluation set along with limiting it if applicable. This is done with the ``DataSplitter`` module.
The ``split_method`` keyword argument defines the way the dataset is split.
More details can be found in ``DataSplitter``, but we summarize the options here:
    - 'random': Ignore all distinction between instances and slices, and split on time steps randomly.
    - 'chronological' (default): Ignore all distinction between instances and slices, and split on time steps chronologically.
    - 'slice-random': Ignore all distinction between instances, and split on slices randomly.
    - 'slice-chronological': Ignore all distinction between instances, and split on slices chronologically.
    - 'instance-random': Split on instances randomly.
    - 'instance-chronological': Split on instances chronologically.
Note that the data is split ON the relevant level (instance, slice, or timesteps).
I.e. If you split on instances and there are only 3 instances, then split_portions=(0.6, 0.2, 0.2)
will be a wild approximation i.t.o actual time steps.

Note that the data is limited during splitting, and the data is limited by removing
the excess data points from the end of the data block after shuffling or ordering according to time.
Also note that if the data is limited too much for a given ``split_portion`` to have a single entry,
an error will occur confirming it.

----------
Scaling
----------

After the data is split and limited, a scaler is fit to the train set data
which will be applied to all data being extracted for model training.
This is done with the ``DataScaler`` module and the corresponding ``scaling_method`` and ``scaling_tag``,
keyword arguments. More details can be found there, but we summarize the options here:
    - scaling_method='z-norm': Features are scaled by subtracting the mean and dividing by the std.
    - scaling_method='zero-one': Features are scaled linearly to be in the range (0, 1).
    - scaling_method=None: No scaling occurs.
    - scaling_tag='in_only': Only the input features will be scaled.
    - scaling_tag='full': The input and output features will be scaled.
    - scaling_tag=None: No features will be scaled.
Note that scaling of output components is not permitted if performing a classification task. scaling_tag='full' will be
automatically changed to scaling_tag='in_only' for classification tasks.

-------
Padding
-------
At some prediction points the ``in_chunk`` might exceed the corresponding
slice range. In these cases we pad the input values. The output values are never padded.
Prediction points that do not have valid output values for an ``out_chunk`` are not selected
during data splitting. The argument ``padding_method`` is the same as ``mode`` in the numpy.pad
function (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
"""

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the PreparedDataset class for Knowit.'

# external imports
from numpy import (array, random, unique, arange,
                   expand_dims, repeat, pad, zeros)

# internal imports
from data.base_dataset import BaseDataset
from data.data_splitting import DataSplitter
from data.data_scaling import DataScaler
from helpers.logger import get_logger

logger = get_logger()


class PreparedDataset(BaseDataset):
    """This is the PreparedDataset which represents a dataset that is preprocessed for model training.
    It is meant to be an abstract class. It contains all the variables in BaseDataset,
    in addition to metadata regarding data splitting, scaling, sampling, and shuffling.

    Parameters
    ----------
    data_path : str
        The path to the dataset on disk.
    in_components : list, shape=[num_in_components,]
        A subset of the 'components' variable of the BaseDataset to be used as input to the model.
    out_components : list, shape=[num_in_components,]
        A subset of the 'components' variable of the BaseDataset to be used as output to the model.
    in_chunk : list, shape=[2,]
        A list of two integers [a, b] for which a <= b,
        defining the time steps of in_components for each prediction point.
    out_chunk : list, shape=[2,]
        A list of two integers [a, b] for which a <= b,
        defining the time steps of out_components for each prediction point.
    split_portions : tuple, shape=[3,]
        The approximate portions of the (train, valid, eval) splits. Needs to add up to 1.
    seed : int
        The seed for reproducibility.
    batch_size : int
        The mini-batch size for training.
    split_method : str
        The method of splitting data. Options are 'random', 'chronological',
        'instance-random', 'instance-chronological', 'slice-random', or 'slice-chronological'.
        See heading for description.
    scaling_method : str | None
        The method for scaling data features. Options are 'z-norm', 'zero-one', or None.
        See heading for description.
    scaling_tag : str | None
        The mode to scale the data. Options are 'in_only', 'full', or None.
        See heading for description.
    shuffle_train : bool
        Whether the training set is shuffled after every epoch.
    limit : int | None
        The number of elements (instances/slices/time) steps to limit the data to.
        See heading for description.
    padding_method : str
        The method to pad model inputs with.
        Options can be found at (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
        See heading for description.
    min_slice : int | None
        The minimum slice size to consider during data splitting/selection.
        If None, no slice selection is performed.

    Attributes
    ----------
    x_map : array, shape=[n_in_components,]
        An array that contains the indices of BaseDataset.components that correspond to input components.
    y_map : array, shape=[n_out_components,]
        An array that contains the indices of BaseDataset.components that correspond to output components.
    train_set_size : int
        The number of prediction points in the training set.
    valid_set_size : int
        The number of prediction points in the validation set.
    eval_set_size : int
        The number of prediction points in the evaluation set.
    selection : dict[str, array]
        A dictionary containing the selection matrices corresponding to each data split.
    x_scaler : object
        The scaler fitted to the training set input features.
    y_scaler : object
        The scaler fitted to the training set output features.
    in_shape : list, shape=[n_input_time_delays, n_in_components]
        A list that represents the shape of the inputs to the model.
    out_shape : list, shape=[n_output_time_delays, n_output_components]
        A list that represents the shape of the outputs to the model. Is modified for classification tasks.

    Notes
    -----
        - All the listed parameters are also stored as attributes.
        - This method initializes the BaseClass variables while keeping the data in memory.
        - The `_prepare` method is automatically called at initialization to initiate data preparation.
    """
    # to be provided
    in_components = None
    out_components = None
    in_chunk = None
    out_chunk = None
    split_portions = None
    seed = None
    batch_size = None
    split_method = None
    scaling_method = None
    scaling_tag = None
    shuffle_train = None
    limit = None
    padding_method = None
    min_slice = None

    # to be filled automatically
    x_map = None
    y_map = None
    train_set_size = None
    valid_set_size = None
    eval_set_size = None
    selection = None
    x_scaler = None
    y_scaler = None
    in_shape = None
    out_shape = None

    def __init__(self, **kwargs) -> None:

        logger.info('Initializing PreparedClass for %s', kwargs['name'])

        # initialize the BaseClass variables while keeping the_data in memory
        super().__init__(kwargs['data_path'], mem_light=False)

        # storing relevant parameters
        self.in_components = kwargs['in_components']
        self.out_components = kwargs['out_components']
        self.in_chunk = kwargs['in_chunk']
        self.out_chunk = kwargs['out_chunk']
        self.split_portions = kwargs['split_portions']
        self.seed = kwargs['seed']
        self.batch_size = kwargs['batch_size']
        self.split_method = kwargs['split_method']
        self.scaling_method = kwargs['scaling_method']
        self.scaling_tag = kwargs['scaling_tag']
        self.shuffle_train = kwargs['shuffle_train']
        self.limit = kwargs['limit']
        self.padding_method = kwargs['padding_method']
        self.min_slice = kwargs['min_slice']

        # Initiate the data preparation
        random.seed(self.seed)
        self._prepare()

    def get_ist_values(self, set_tag: str) -> list:
        """Get the IST values for a given set tag.

        This function retrieves the Instance-Slice-Time (IST) for each prediction point in a specific dataset split.

        Parameters
        ----------
        set_tag : str
            The data split to retrieve the IST values for.
            Options are 'train', 'valid', 'eval' or 'all'.

        Returns
        -------
        ist_values : list
            The IST values for a given set tag.
            Each row represents one prediction point by the following three values.
                - The first value (Any) indicates the instance that the prediction point belongs to.
                - The second value (int) indicates the index, within the instance, of the slice that the prediction point belongs to.
                - The third value (numpy.datetime64) indicates the timestep at which the prediction point is found.

        Notes
        -----
            - If set_tag='all' then all prediction points are returned in order ['train', 'valid', 'eval'].
        """
        the_data = self.get_the_data()
        ist_values = []
        if set_tag == 'all':
            tags = ['train', 'valid', 'eval']
            for tag in tags:
                for p in self.selection[tag]:
                    i = self.instances[p[0]]
                    s = p[1]
                    t = the_data[i][s]['t'][p[2]]
                    ist_values.append((i, s, t))
        elif set_tag in ('train', 'valid', 'eval'):
            for p in self.selection[set_tag]:
                i = self.instances[p[0]]
                s = p[1]
                t = the_data[i][s]['t'][p[2]]
                ist_values.append((i, s, t))
        else:
            logger.error('Unknown set tag %s', set_tag)
            exit(101)
        return ist_values

    def extract_dataset(self, set_tag: str) -> dict:
        """Extracts the relevant samples for one of the data splits.

        This function retrieves the 'the_data' dictionary,
        procs the function to extract specific prediction points,
        and then scales them if applicable.

        Parameters
        ----------
        set_tag : str
            The data split to retrieve the relevant samples for.
            Options are 'train', 'valid', 'eval'.

        Returns
        -------
        dataset : dict[str, array]
            The relevant samples for the selected data split.
            The dictionary includes the following keys:
                - 'x' (array): The input features with the shape [num_prediction_points, num_in_time_delays, num_in_components]
                - 'y' (array): The output features with the shape [num_prediction_points, num_out_time_delays, num_out_components]

        Notes
        -----
            - The data is scaled if scalers are available.
        """
        # TODO: This method is very heavy on memory if the dataset is large.
        #  Will need to find alternative where we do not put the entire dataset in memory.

        logger.info('Extracting %s set.', set_tag)
        x_vals, y_vals = self._fast_extract(self.get_the_data(), self.selection[set_tag],
                                            self.in_chunk, self.out_chunk,
                                            self.instances, self.x_map, self.y_map,
                                            self.padding_method)

        if self.x_scaler:
            x_vals = self.x_scaler.transform(x_vals)
        if self.y_scaler:
            y_vals = self.y_scaler.transform(y_vals)

        return {'x': x_vals, 'y': y_vals}

    def _prepare(self) -> None:
        """Prepare the dataset by splitting and scaling the data.

        This function calculates some relevant overhead variables (in_shape, out_shape, etc.),
        it defines the correct data splits, and fits the correct scalers.

        Notes
        -----
            - The splits are defined in the 'selection' variable in three selection matrices.
            - The scalers are stored in the 'x_scaler' and 'y_scaler' variables.
            - After the data is prepared the 'the_data' dictionary is removed from memory.

        """
        # check that desired components are in dataset
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

        # infer input output component mappings
        self.y_map = array([i for i, e in enumerate(self.components)
                            if e in self.out_components])
        self.x_map = array([i for i, e in enumerate(self.components)
                            if e in self.in_components])

        # infer model input and output shapes
        self.in_shape = [self.in_chunk[1] - self.in_chunk[0] + 1, len(self.in_components)]
        self.out_shape = [self.out_chunk[1] - self.out_chunk[0] + 1, len(self.out_components)]

        # split the dataset
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

        # scale the dataset
        logger.info('Preparing data scalers, if relevant.')
        self.x_scaler, self.y_scaler = DataScaler(self.extract_dataset('train'),
                                                  self.scaling_method,
                                                  self.scaling_tag).get_scalers()

        if hasattr(self, 'the_data'):
            delattr(self, 'the_data')
            logger.info('the_data structure is removed from memory.')

    @staticmethod
    def _fast_extract(the_data: dict, selection: array, in_chunk: list, out_chunk: list,
                      instances: list, x_map: array, y_map: array, padding_method: str) -> tuple:
        """Pads the data structure and initiates the extraction of input and output values for prediction points defined
        in the provided selection matrix.

        Parameters
        ----------
        the_data : dict
            The 'the_data' dictionary as stored on disk.
        selection : array, shape=[num_prediction_points, 3]
            A selection matrix.
        in_chunk : list, shape=[2,]
            A list of two integers [a, b] for which a <= b,
            defining the time steps of in_components for each prediction point.
        out_chunk : list, shape=[2,]
            A list of two integers [a, b] for which a <= b,
            defining the time steps of out_components for each prediction point.
        instances : list
            A list of instances in the dataset.
        x_map : array, shape=[n_in_components,]
            An array that contains the indices of BaseDataset.components that correspond to input components.
        y_map : array, shape=[n_out_components,]
            An array that contains the indices of BaseDataset.components that correspond to output components.
        padding_method : str
            The method to pad model inputs with.
            Options can be found at (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
            See heading for description.

        Returns
        -------
        tuple
            A tuple containing:
            - x_vals : array
                The input features with the shape [num_prediction_points, num_in_time_delays, num_in_components]
            - y_vals : array
                The output features with the shape [num_prediction_points, num_out_time_delays, num_out_components]

        """
        # get the indices of all relevant slices
        relevant_slice_indx = unique(selection[:, :2], axis=0)
        # calculates the maximum padding that could be required
        max_pad = max(abs(array(in_chunk))) + 1

        # pad the dataset before sampling
        for r in relevant_slice_indx:
            i = instances[r[0]]
            s = r[1]
            the_data[i][s]['d'] = PreparedDataset._do_padding(the_data[i][s]['d'],
                                                                  'backward',
                                                                  padding_method,
                                                                  len(the_data[i][s]['d']) + max_pad)
            the_data[i][s]['d'] = PreparedDataset._do_padding(the_data[i][s]['d'],
                                                                  'forward',
                                                                  padding_method,
                                                                  len(the_data[i][s]['d']) + max_pad)
        # sample input and output values
        x_vals, y_vals = PreparedDataset._parr_sample(selection, in_chunk, out_chunk,
                                                      max_pad, the_data,
                                                      instances, y_map, x_map)

        return x_vals, y_vals

    @staticmethod
    def _parr_sample(selection: array, in_chunk: list, out_chunk: list, max_pad: int,
                     the_data: dict, instances: list, y_map: array, x_map: array) -> tuple:
        """Sample per prediction point data blocks,
        retrieves the corresponding feature values and package as two arrays.


        Parameters
        ----------
        selection : array, shape=[num_prediction_points, 3]
            A selection matrix.
        in_chunk : list, shape=[2,]
            A list of two integers [a, b] for which a <= b,
            defining the time steps of in_components for each prediction point.
        out_chunk : list, shape=[2,]
            A list of two integers [a, b] for which a <= b,
            defining the time steps of out_components for each prediction point.
        max_pad : int
            The maximum padding that could be required.
        the_data : dict
            The 'the_data' dictionary as stored on disk.
        instances : list
            A list of instances in the dataset.
        y_map : array, shape=[n_out_components,]
            An array that contains the indices of BaseDataset.components that correspond to output components.
        x_map : array, shape=[n_in_components,]
            An array that contains the indices of BaseDataset.components that correspond to input components.

        Returns
        -------
        tuple
            A tuple containing:
            - x_vals : array
                The input features with the shape [num_prediction_points, num_in_time_delays, num_in_components]
            - y_vals : array
                The output features with the shape [num_prediction_points, num_out_time_delays, num_out_components]
        """

        def _sample_blocks(t: array, chunk: list, pad_size: int):
            """Generate per prediction point sample blocks based on the provided time indices, chunk size, and padding.

            This function creates an array of sample blocks by adding padding to the specified
            chunk size and applying it to the provided time indices. Each row in the output array represents
            the input time indices that corresponds to each prediction point in t.

            Parameters
            ----------
            t : array, shape=[num_prediction_points,]
                An array of time indices for which the sample blocks are generated.
            chunk : list, shape=[2,]
                A tuple representing the start and end of the chunk size.
            pad_size : int
                The amount of padding to be added to the chunk size.

            Returns
            -------
            blocks : array, shape=[num_prediction_points, num_in_time_delays]
                A 2D array where each row corresponds to a time index from `t` and contains the
                sample blocks generated by adding the chunk size and padding.

            Notes
            -----
                - The `blocks` array is generated by first creating a range of indices based on the chunk size
                  and padding, and then adding these to each time index in `t`.
            """
            blocks = arange(chunk[0] + pad_size, chunk[1] + pad_size + 1)
            blocks = expand_dims(blocks, axis=0)
            t_selection = t
            blocks = repeat(blocks, len(t_selection), axis=0)
            t_selection = expand_dims(t_selection, axis=1)
            t_selection = repeat(t_selection, blocks.shape[1], axis=1)
            blocks += t_selection
            return blocks

        x_blocks = _sample_blocks(selection[:, 2], in_chunk, max_pad)
        y_blocks = _sample_blocks(selection[:, 2], out_chunk, 0)
        x_vals = zeros(shape=(selection.shape[0], in_chunk[1]-in_chunk[0]+1, x_map.size))
        y_vals = zeros(shape=(selection.shape[0], out_chunk[1]-out_chunk[0]+1, y_map.size))
        s_indx = 0
        # TODO: Here is still a bottleneck. But at least it is just sampling now.
        for s in selection:
            x_vals[s_indx, :, :] = the_data[instances[s[0]]][s[1]]['d'][x_blocks[s_indx]][:, x_map]
            y_vals[s_indx, :, :] = the_data[instances[s[0]]][s[1]]['d'][y_blocks[s_indx]][:, y_map]
            s_indx += 1
        del the_data
        return x_vals, y_vals

    @staticmethod
    def _do_padding(vals: array, direction: str, method: str, cap: int) -> array:
        """Pad the input array 'vals' using the specified method and direction up to the given size (cap).

        This function pads the input 2D array `vals` (with shape [num_timesteps, num_features]) based on the
        provided direction and method. The padding is applied either forward or backward to reach
        the specified size (cap).

        Parameters
        ----------
        vals : array, shape=[num_timesteps, num_features]
            The input array that needs to be padded.
        direction : str
            The direction in which to apply the padding. Must be either 'backward' or 'forward'.
            - 'backward': Pads at the beginning of the array.
            - 'forward': Pads at the end of the array.
        method : str
            The padding method to use. Must be one of the methods supported by `numpy.pad`.
            Common options include 'constant' for zero padding, 'reflect', 'symmetric', etc.
        cap : int
            The target size for the number of timesteps after padding.

        Returns
        -------
        ret_vals : array, shape=[cap, num_features]
            The padded array with the number of timesteps equal to `cap`.

        Raises
        ------
        ValueError
            If an unknown padding direction or method is provided.

        Notes
        -----
            The `vals` array is padded with zeros if the method is 'zero', otherwise, the specified method
            is used. Refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html for more details
            on available padding methods.
        """
        if direction == 'backward':
            pw = ((cap - vals.shape[0], 0), (0, 0))
        elif direction == 'forward':
            pw = ((0, cap - vals.shape[0]), (0, 0))
        else:
            logger.error('Unknown padding direction %s.', direction)
            exit(101)

        pw = array(pw)
        if method == 'zero':
            ret_vals = pad(vals, pad_width=pw, mode='constant')
        else:
            ret_vals = pad(vals, pad_width=pw, mode=method)

        return ret_vals




