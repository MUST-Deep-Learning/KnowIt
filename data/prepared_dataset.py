"""
------------------
PreparedDataset
------------------

The ``PreparedDataset`` represents a ``BaseDataset`` that is preprocessed for model training.
It inherits from ``BaseDataset``. Based on the provided path(s) to a dataset, it will populate the
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

- regression (heartrate from an 11-time step window given three instantaneous biometrics)
    - in_components = [biometric1, biometric2, biometric2]
    - out_components = [heart rate]
    - in_chunk = [-5, 5]
    - out_chunk = [0, 0]

- autoregressive univariate forcasting (predict a stock's value in 5 time steps given last 20 time steps)
    - in_components = [stock,]
    - out_components = [stock,]
    - in_chunk = [-20, 0]
    - out_chunk = [5, 5]
- time series classification (detect a whale call given an audio recording of 5 time steps)
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
    - 'custom': User defined split.
Note that the data is split ON the relevant level (instance, slice, or timesteps).
I.e. If you split on instances and there are only 3 instances, then split_portions=(0.6, 0.2, 0.2)
will be a wild approximation i.t.o actual time steps.

Note that, if desired, the data is limited during splitting, and the data is limited by removing
the excess data points from the end of the data block after shuffling or ordering according to time.
Also note that if the data is limited too much for a given ``split_portion`` to have a single entry,
an error will occur confirming it.

Note that the 'custom' split has to be constructed during the data importing.
See the RawDataConverter module for more information.

-------
Scaling
-------

After the data is split and possibly limited, a scaler is fit to the train set data
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
Also note that the scaling happens "online" during sampling.

-------
Padding
-------

At some prediction points the ``in_chunk`` might exceed the corresponding
slice range. In these cases we pad the input values. The output values are never padded.
Prediction points that do not have valid output values for an ``out_chunk`` are not selected
during data splitting. The argument ``padding_method`` is the same as ``mode`` in the numpy.pad
function (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).

-------------
CustomDataset
-------------

``PreparedDataset.get_dataloader()`` can be called to return a Pytorch dataloader that can be used for model training.
This function compiles a dataloader with a ``CustomDataset`` class.
This class inherits from ``torch.utils.data.Dataset``.
It defines how the samples will look when enumerating over the dataloader.
This includes sampling, padding, and normalizing.

---------------------------
CustomClassificationDataset
---------------------------

If the task is a regression task a ``CustomClassificationDataset`` will be used instead of ``CustomDataset``.
``CustomClassificationDataset`` inherits from ``CustomDataset`` and adds some classification specific methods.

-------------
CustomSampler
-------------

In addition to ``CustomDataset``, ``PreparedDataset`` also uses a custom batch sampler ``CustomSampler``.
This class supports three different modes of temporal contiguity. See the module for details.

"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = ('Contains the PreparedDataset, CustomDataset, and CustomClassificationDataset, '
                   'and CustomSampler class for Knowit.')

# external imports
from numpy import (array, random, unique, pad, isnan, arange, expand_dims, concatenate,
                   diff, where, split)
from numpy.random import Generator
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import from_numpy, is_tensor, Tensor
from torch import zeros as zeros_tensor

# internal imports
from data.base_dataset import BaseDataset
from data.data_splitting import DataSplitter
from data.data_scaling import DataScaler
from helpers.logger import get_logger

logger = get_logger()


class PreparedDataset(BaseDataset):
    """This is the PreparedDataset which represents a dataset that is preprocessed for model training.
    It contains all the variables in BaseDataset in addition to metadata regarding
    data splitting, scaling, and sampling.

    Parameters
    ----------
    meta_path : str
        The path to the desired dataset metadata. The path should point to a pickle file.
    package_path : str
        The path to the desired dataset package. The path should point to a directory containing the
        partitioned parquet.
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
        'instance-random', 'instance-chronological', 'slice-random', 'slice-chronological', or 'custom.
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
    batch_sampling_mode : int | None
        The sampling mode for generating batches in the CustomSampler class.
        Options are 0, 1, or 2.
    succession_length : int | None
        Number of batches to maintain temporal succession if batch_sampling_mode=2.
    skip_max : int
        Maximum number of initial time steps to skip if batch_sampling_mode=1 and 2 and shuffle_train=True.

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
    class_set : dict
        A dictionary that maps each class name to an integer class ID.
        Only created if task='classification'.
    class_counts : dict
        A dictionary that maps each class ID to its size.
        Only created if task='classification'.
    class_splits: dict
        A dictionary defining the custom selection matrices.

    Notes
    -----
        - All the listed parameters are also stored as attributes.
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
    batch_sampling_mode = None
    succession_length = None
    skip_max = None

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
    class_set = None # only filled if task='classification'
    class_counts = None  # only filled if task='classification'
    class_splits = None

    def __init__(self, **kwargs) -> None:

        logger.info('Initializing PreparedClass for %s', kwargs['name'])

        # initialize the BaseClass variables while keeping the_data in memory
        super().__init__(kwargs['meta_path'], kwargs['package_path'])

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
        self.task = kwargs['task']
        self.batch_sampling_mode = kwargs['batch_sampling_mode']
        self.succession_length = kwargs['succession_length']
        self.skip_max = kwargs['skip_max']

        # Initiate the data preparation
        random.seed(self.seed)
        self._prepare()

    def get_dataset(self, set_tag: str, preload: bool = False):
        """Creates and returns a PyTorch Dataset for a specified dataset split.

        Parameters
        ----------
        set_tag : str
            A string indicating the dataset split to load ('train', 'valid', 'eval').
        preload : bool, default = False
            Whether to preload the raw relevant instances and slice into memory when sampling feature values.

        Returns
        -------
        Object
            A PyTorch derived Dataset for the specified dataset split.
        """
        if self.task == 'regression':
            dataset = CustomDataset(self.get_extractor(), self.selection[set_tag],
                          self.x_map, self.y_map,
                          self.x_scaler, self.y_scaler,
                          self.in_chunk, self.out_chunk,
                          self.padding_method,
                          preload=preload)
        elif self.task == 'classification':
            dataset = CustomClassificationDataset(self.get_extractor(), self.selection[set_tag],
                                                  self.x_map, self.y_map,
                                                  self.x_scaler, self.y_scaler,
                                                  self.in_chunk, self.out_chunk,
                                                  self.class_set, self.padding_method,
                                                  preload=preload)
        else:
            logger.error('Unknown task: %s', self.task)
            exit(101)

        return dataset


    def get_dataloader(self, set_tag: str,
                       analysis: bool = False,
                       num_workers: int = 4,
                       preload: bool = False) -> DataLoader:
        """Creates and returns a PyTorch DataLoader for a specified dataset split.

        This method generates a DataLoader for a given dataset split (e.g., train, valid, or eval).
        It uses the `CustomDataset` class to create the dataset and then initializes
        a DataLoader with a `CustomSampler` for batch generation.

        Parameters
        ----------
        set_tag : str
            A string indicating the dataset split to load ('train', 'valid', 'eval').
        analysis : bool, default = False
            A flag indicating whether the dataloader is being used for analysis purposes. If set to True,
            the `drop_last` and `shuffle` parameters of the DataLoader will be set to False.
        num_workers : int, default = 4
            Sets the number of workers to use for loading the dataset.
        preload : bool, default = False
            Whether to preload the raw relevant instances and slice into memory when sampling feature values.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader for the specified dataset split.

        Notes
        -----
        If set_tag=`valid` or `eval` or analysis=True,
        then the dataloader will not be shuffled and no smaller than batch_size batches will be dropped.
        """

        if set_tag == 'train' and not analysis:
            shuffle = self.shuffle_train
            drop_small = True
        else:
            shuffle = False
            drop_small = False

        sampler = CustomSampler(selection=self.selection[set_tag],
                                batch_size=self.batch_size,
                                seed=self.seed,
                                mode=self.batch_sampling_mode,
                                drop_small=drop_small,
                                succession_length=self.succession_length,
                                shuffle=shuffle,
                                skip_max=self.skip_max)

        dataset = self.get_dataset(set_tag, preload=preload)

        dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=num_workers)

        return dataloader


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

        the_data = self.get_extractor()
        ist_values = []
        if set_tag == 'all':
            tags = ['train', 'valid', 'eval']
        elif set_tag in ['train', 'valid', 'eval']:
            tags = [set_tag]
        else:
            logger.error('Unknown set_tag {}'.format(set_tag))
            exit(101)
        for tag in tags:
            for p in self.selection[tag]:
                i = p[0]
                s = p[1]
                t_step = the_data.time_step(p[0], p[1], p[2])
                t = t_step.name
                ist_values.append((i, s, t))

        return ist_values

    def fetch_input_points_manually(self, set_tag: str, point_ids: int | list) -> Tensor:
        """ Manually fetch data points from the datamodule based on provided point IDs.

        Parameters
        ----------
        set_tag : str
            A string indicating the dataset split to load ('train', 'valid', 'eval').
        point_ids : int | list[int]
            The IDs of the data points to fetch. These indices are defined as the relative position in the selection
            matrix corresponding to the set tag. Can be a single integer, a list of integers,
            or a tuple specifying a range (start, end).

        Returns
        -------
        Tensor
            A tensor containing the data points corresponding to the provided
            IDs.

        Raises
        ------
        ValueError
            If the provided point IDs are invalid or out of range.

        """
        dataset = self.get_dataset(set_tag, preload=False)

        if isinstance(point_ids, tuple):
            ids = list(range(point_ids[0], point_ids[1]))
        else:
            ids = point_ids

        try:
            tensor = dataset.__getitem__(idx=ids)["x"]
        except ValueError:
            logger.error('Invalid: ids %s not in choice "%s" (which has range %s)',
                         str(point_ids), set_tag,
                         str((0, len(self.selection[set_tag]))))
            exit(101)

        return tensor

    def _prepare(self) -> None:
        """Prepare the dataset by splitting and scaling the data.

        This function calculates some relevant overhead variables (in_shape, out_shape, etc.),
        it defines the correct data splits, and fits the correct scalers.

        Notes
        -----
            - The splits are defined in the 'selection' variable in three selection matrices.
            - The scalers are stored in the 'x_scaler' and 'y_scaler' variables.
            - After the data is prepared the 'the_data' dictionary is removed from memory.
            - If custom splits are defined, the selection matrices are constructed by data.raw_data_conversion.
              which is used to indicate the set splits

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
        if self.task == 'classification':
            if self.out_chunk[0] != self.out_chunk[1]:
                logger.error('Currently, KnowIt can only perform classification at one specific time step at a time. '
                             'Please change the out_chunk %s argument to reflect this. Both values must match.',
                             str(self.out_chunk))
                exit(101)
            self._get_classes()
            self._count_classes()
            self.out_shape = [1, len(self.class_set)]

        # split the dataset
        logger.info('Preparing data splits (selection).')
        if self.split_method != 'custom':
            self.selection = DataSplitter(self.get_extractor(),
                                          self.split_method,
                                          self.split_portions,
                                          self.limit, self.x_map, self.y_map,
                                          self.in_chunk, self.out_chunk,
                                          self.min_slice).get_selection()
        elif self.custom_splits != None:
            self.selection = self.custom_splits
            missing_split_components = set(self.selection) - {'valid', 'train', 'eval'}
            if len(missing_split_components) > 0:
                logger.error('Defined set selection %s not in custom splits.',
                             str(missing_out_components))
                exit(101)
        else:
            logger.error('Custom splits not defined.')
            exit(101)

        self.train_set_size = len(self.selection['train'])
        self.valid_set_size = len(self.selection['valid'])
        self.eval_set_size = len(self.selection['eval'])
        logger.info('Data split sizes: ' + str((self.train_set_size,
                                                self.valid_set_size,
                                                self.eval_set_size)))

        # scale the dataset
        logger.info('Preparing data scalers, if relevant.')

        if self.task == 'classification' and self.scaling_tag == 'full':
            logger.warning('scaling_tag cannot be full for classification tasks. Changing to scaling_tag=in_only.')
            self.scaling_tag = 'in_only'

        self.x_scaler, self.y_scaler = DataScaler(self.get_extractor(),
                                                  self.selection['train'],
                                                  self.scaling_method,
                                                  self.scaling_tag,
                                                  self.x_map,
                                                  self.y_map).get_scalers()

    def _get_classes(self) -> None:
        """Identify unique classes in the dataset.

        This method processes the dataset to determine the unique classes present in the data.
        The unique classes are stored in the `class_set` attribute.
        """

        if self.task != 'classification':
            logger.error('Task must be classification to determine classes.')
            exit(101)

        data_extractor = self.get_extractor()
        found_class_set = set()
        for i in data_extractor.data_structure:
            vals = data_extractor.instance(i).to_numpy()
            vals = vals[:, self.y_map][~isnan(vals[:, self.y_map]).any(axis=1)]
            unique_entries = unique(vals, axis=0)
            unique_entries_list = []
            for u in unique_entries:
                if len(u) > 1:
                    unique_entries_list.append(tuple(u))
                else:
                    unique_entries_list.append(u.item())
            unique_entries = unique_entries_list
            found_class_set.update(set(unique_entries))

        self.class_set = {}
        tick = 0
        for c in found_class_set:
            self.class_set[c] = tick
            tick += 1

        logger.info('Found %s unique classes.',
                    str(len(self.class_set)))
        logger.info(self.class_set)

    def _count_classes(self) -> None:
        """ Count the number of instances of each class and store in a dictionary as attribute."""

        if self.task != 'classification' or not hasattr(self, 'class_set'):
            logger.error('Task must be classification and classes must have been determined to count classes.')
            exit(101)

        data_extractor = self.get_extractor()
        self.class_counts = {}
        for c in self.class_set:
            class_count = 0
            for i in data_extractor.data_structure:
                instance = data_extractor.instance(i).to_numpy()[:, self.y_map]
                class_count += len(where(instance == c)[0])
            self.class_counts[self.class_set[c]] = class_count


class CustomSampler(Sampler):
    """
    Custom batch sampler for deep time series models, supporting different modes of temporal contiguity.

    Parameters
    ----------
    selection : array, shape=[n_samples, 3]
        Selection matrix containing instance IDs, slice IDs, time steps.
    batch_size : int
        Number of sequences per batch.
    seed : int, default=None
        Random seed for reproducibility.
    mode : int, default=0
        Sampling mode (0, 1, or 2) as described below.
    drop_small : bool, default=True
        Whether to drop batches smaller than batch_size.
    succession_length : int, default=10
        Number of batches to maintain temporal succession in mode 2. Only applicable if mode=2.
    shuffle : bool, default=True
        Whether to apply shuffling.
    skip_max : int, default=10
        Maximum number of initial time steps to skip when shuffling.
        Only applicable if mode=1 or 2 and shuffle=True.

    Attributes
    ----------
    selection : array, shape=[n_samples, 3]
        Selection matrix containing instance IDs, slice IDs, time steps.
    batch_size : int
        Number of sequences per batch.
    seed : int, default=None
        Random seed for reproducibility.
    mode : int, default=0
        Sampling mode (0, 1, or 2) as described below.
    drop_small : bool, default=True
        Whether to drop batches smaller than batch_size.
    succession_length : int, default=10
        Number of batches to maintain temporal succession in mode 2. Only applicable if mode=2.
    shuffle : bool, default=True
        Whether to apply shuffling.
    skip_max : int, default=10
        Maximum number of initial time steps to skip when shuffling.
        Only applicable if mode=1 or 2 and shuffle=True.
    batches : list
        The current set of batches that will be iterated over.
    epoch : int
        Current epoch. Used for random seeding.

    Notes
    -----
    - Mode 0: Time is contiguous within sequences but not enforced across sequences or batches.
    - Mode 1: Time is contiguous within sequences and within batches, but not across batches.
    - Mode 2: Time is contiguous within sequences and across batches, but not withing batches,
    maintaining temporal succession.
    - When shuffle=False:
        - Mode 0: Batches are constructed in dataset order as per the "selection" array.
        - Mode 1 & 2: Batches are constructed greedily from the first available prediction point.
    - When shuffle=True:
        - Mode 0: Sequences within and across batches are randomly shuffled.
        - Mode 1 & 2: A random number of sequences at the beginning of each contiguous block is skipped,
          later added back as new batches. Additionally, batches or successive batch blocks are shuffled.
    """

    def __init__(self,
                 selection: array,
                 batch_size: int,
                 seed: int = None,
                 mode: int = 0,
                 drop_small: bool = True,
                 succession_length: int = 10,
                 shuffle: bool = True,
                 skip_max: int = 10) -> None:

        self.selection = selection
        self.batch_size = batch_size
        self.seed = seed
        self.mode = mode
        self.drop_small = drop_small
        self.succession_length = succession_length
        self.shuffle = shuffle
        self.skip_max = skip_max

        self.batches = []
        self.epoch = 0

    def __iter__(self):
        """
        Generates batches according to the selected mode and shuffling settings.

        Returns
        -------
        iterator
            An iterator over the generated batches.
        """
        self.epoch += 1
        if self.mode == 0:
            self._create_default_batches()
        elif self.mode == 1:
            self._create_contiguous_batches(tag='within')
        elif self.mode == 2:
            self._create_contiguous_batches(tag='between')
        else:
            logger.error('Unknown sampler mode %s. Expected 0, 1, or 2.', self.mode)
            exit(101)

        self._check_unique()
        self._check_small()

        return iter(self.batches)

    def __len__(self):
        """
        Returns the number of batches.

        Returns
        -------
        int
            Number of generated batches.
        """
        return len(self.batches)

    def _create_default_batches(self) -> None:
        """
        Creates batches in a simple sequential order. Shuffles if enabled.
        """
        self.batches = []
        sampling = arange(len(self.selection))
        if self.shuffle:
            rng = self._get_rng()
            rng.shuffle(sampling)
        for b in range(0, len(self.selection), self.batch_size):
            batch = [t for t in sampling[b:min(b + self.batch_size, len(self.selection))]]
            self.batches.append(batch)

    def _create_contiguous_batches(self, tag: str) -> None:
        """
        Creates batches while maintaining temporal contiguity within or across batches.

        Parameters
        ----------
        tag : str
            Determines contiguity level ('within' for mode 1, 'between' for mode 2).
        """
        self.batches = []

        # add indices to selection matrix corresponding to dataset-wide position
        inx = expand_dims(arange(len(self.selection)), 1)
        selection = concatenate((self.selection, inx), axis=1)

        # if shuffle is on prepare rng
        rng = None
        if self.shuffle:
            rng = self._get_rng()

        # for each unique instance slice in selection
        for s in unique(selection[:, :2], axis=0):
            # get subselection
            idx = where((selection[:, 0] == s[0]) & (selection[:, 1] == s[1]))[0]
            s_block = selection[idx]

            # sort according to time steps
            s_block = s_block[s_block[:, 2].argsort()]

            # find a list of contiguous sub-blocks in subselection
            breakpoints = where(diff(s_block[:, 2]) != 1)[0] + 1
            s_block = split(s_block, breakpoints)

            # compile batches from found sub-blocks
            if tag == 'within':
                self._contiguous_within(s_block, rng)
            elif tag == 'between':
                self._contiguous_between(s_block, rng)

        # if shuffle is on we also shuffle the order of batches for mode 1
        if self.shuffle and tag == 'within':
            rng.shuffle(self.batches)

        if tag == 'between':
            # compile full batches from remainder batches created in mode 2
            self._reconcile_remainder()
            # if shuffle is on we also shuffle successive blocks for mode 2
            if self.shuffle:
                self._shuffle_sucblocks(rng)

    def _contiguous_within(self, s_block: list, rng: Generator) -> None:
        """
        Constructs batches where sequences remain contiguous within each batch.

        Parameters
        ----------
        s_block : list
            List of contiguous blocks of time steps.
        rng : np.random.Generator or None
            Random number generator for shuffling, if applicable.
        """

        # for each contiguous sub-block
        for sb in s_block:
            sb_batches = []
            # determine how many sequences to skip at the start
            skip = 0
            if self.shuffle:
                skip = rng.integers(0, min(self.skip_max, len(sb)))
            # compile batches greedily
            for b in range(skip, len(sb), self.batch_size):
                batch = sb[arange(b, min(b + self.batch_size, len(sb))), 3].tolist()
                sb_batches.append(batch)
            # if skipped, add back tot start as new batches
            if self.shuffle:
                skipped = []
                for b in range(0, skip, self.batch_size):
                    batch = sb[arange(b, min(b + self.batch_size, skip)), 3].tolist()
                    skipped.append(batch)
                sb_batches = skipped + sb_batches
            # add results to core list of batches
            self.batches.extend(sb_batches)

    def _contiguous_between(self, s_block: list, rng: Generator) -> None:
        """
        Constructs batches where sequences remain contiguous across batches,
        not necessarily within each batch.

        Parameters
        ----------
        s_block : list
            List of contiguous blocks of time steps.
        rng : np.random.Generator or None
            Random number generator for shuffling, if applicable.
        """

        def _proc_block(sb: array, skip: int) -> list:
            """
            Processes a contiguous sub-block into successive batches.

            Parameters
            ----------
            sb : array, shape=[n_samples, 4]
                Contiguous sub-block of the selection matrix.
            skip : int
                Number of initial time steps to skip.

            Returns
            -------
            sb_batches : list
                Processed batches with temporal succession.
            """
            sb_batches = []
            # init a new block of successive batches
            new_suc_batches = [[] for _ in range(self.succession_length)]
            s_tick = 0
            b_tick = 0
            # for every prediction point in the contiguous sub-block
            for t in range(skip, len(sb)):
                # add the sequence ID
                new_suc_batches[s_tick].append(sb[t][3])
                s_tick += 1
                # if enough prediciton points are added to reach succession length, move to next position in batches
                if s_tick == self.succession_length:
                    s_tick = 0
                    b_tick += 1
                # if last poisition in batches is reached
                if b_tick >= self.batch_size:
                    # add new succession block to resulting batches
                    sb_batches.extend(new_suc_batches)
                    # re-init a new block of successive batches
                    new_suc_batches = [[] for _ in range(self.succession_length)]
                    s_tick = 0
                    b_tick = 0
            # if there are prediction points remaining in the candidate succession block add them to
            if s_tick != 0 or b_tick != 0:
                sb_batches.extend(new_suc_batches)
            return sb_batches

        # for each contiguous sub-block
        for sb in s_block:
            sb_batches = []
            # determine how many sequences to skip at the start
            skip = 0
            if self.shuffle:
                skip = rng.integers(0, self.skip_max)
            # compile batches greedily
            full_sb_batches = _proc_block(sb, skip)
            sb_batches.extend(full_sb_batches)
            # if skipped, add back tot start as new batches
            if self.shuffle:
                sb_batches = _proc_block(sb[0:skip], 0) + sb_batches
            # add results to core list of batches
            self.batches.extend(sb_batches)

    def _shuffle_sucblocks(self, rng) -> None:
        """ Shuffles the set of batches that will be iterated over,
        successive block by successive block."""
        list_of_blocks = []
        for s in range(0, len(self.batches), self.succession_length):
            list_of_blocks.append(self.batches[s:s + self.succession_length])
        rng.shuffle(list_of_blocks)
        self.batches = [batch for block in list_of_blocks for batch in block]

    def _reconcile_remainder(self) -> None:
        """
        Adjusts batch sizes by separating full batches, attempting to merge smaller batches,
        and handling unresolvable small batches.

        This function processes the generated batches in Mode 2 to ensure that as many batches
        as possible reach the specified batch size. It classifies batches into three categories:

        - Full-sized batches that meet the batch_size requirement.
        - Remainder batches that are smaller than batch_size but can be combined.
        - True loss batches that are too small to be reconciled.

        The function then attempts to merge remainder batches into full-sized batches where possible,
        while appending any unresolved batches to the final batch list.
        """

        # split the core set of batches into the following
        #  - batches that are already appropriately sized
        full_batches = []
        #  - batches that are smaller than batch_size,
        #    but can be joined with others to create a full sized batch
        remainder = []
        #  - batches that are smaller than batch_size,
        #    and cannot be joined with others to create a full sized batch
        true_loss = []

        for s in range(0, len(self.batches), self.succession_length):
            suc_block = self.batches[s:s + self.succession_length]
            min_length = min(len(b) for b in suc_block)
            if min_length == self.batch_size:
                full_batches.extend(suc_block)
            elif min_length > 0:
                for b in suc_block:
                    remainder.append(b[:min_length])
                    true_loss.append(b[min_length:])
            elif min_length == 0:
                true_loss.extend(suc_block)
            else:
                logger.error('Something went terribly wrong with the mode 2 batch sampler.')
                exit(101)

        # store appropriately sized batches
        self.batches = full_batches

        # combine remainder batches into new full size batches
        new_full_batches = []
        for s in range(0, len(remainder), self.succession_length):
            if s == 0:
                suc_block0 = remainder[s:s + self.succession_length]
            l0 = min(len(b) for b in suc_block0)
            suc_block1 = remainder[s + self.succession_length:s + 2 * self.succession_length]
            s1_part = self.batch_size - l0
            new_full_suc_block = [row_a + row_b[:s1_part] for row_a, row_b in zip(suc_block0, suc_block1)]
            if len(new_full_suc_block) != 0:
                l_new = min(len(b) for b in new_full_suc_block)
                if l_new == self.batch_size:
                    new_full_batches.extend(new_full_suc_block)
                    suc_block0 = [b[s1_part:] for b in suc_block1]
                else:
                    suc_block0 = new_full_suc_block
            else:
                true_loss.extend(suc_block0)

        # store new full size batches
        self.batches = self.batches + new_full_batches
        # also add final unreconcilable batches too
        self.batches = self.batches + true_loss

    def _check_small(self, threshold: int = 0.05) -> None:
        """
        Drops small batches if drop_small is enabled and logs the number of dropped sequences.
        """
        total_before = sum([len(b) for b in self.batches])
        if self.drop_small:
            self.batches = [b for b in self.batches if len(b) >= self.batch_size]
        total_after = sum([len(b) for b in self.batches])
        if total_before != total_after and threshold < (total_before - total_after)/total_before:
            logger.warning('%s/%s prediction points dropped from dataloader when dropping batches smaller than %s!',
                           total_before - total_after, total_before, self.batch_size)

    def _check_unique(self) -> None:
        """
        Ensures that all selected prediction points are included exactly once in the final batches.
        """
        raw_total = len(self.selection)
        unique_t = []
        for batch in self.batches:
            for b in batch:
                unique_t.append(b)
        batched_total = len(unique_t)
        if batched_total != raw_total:
            logger.error("Uh oh, total batched prediction points do not match total raw prediction points.")
            exit(101)

    def _get_rng(self) -> Generator:
        """
        Returns a random number generator (RNG) instance based on the shuffle setting.

        If shuffling is enabled, the RNG seed is adjusted by incorporating the current epoch
        to ensure different randomization patterns across epochs. Otherwise, the RNG is
        initialized with a fixed seed for reproducibility.

        Returns
        -------
        random.Generator
            A NumPy random number generator instance.

        """
        if self.shuffle:
            rng = random.default_rng(self.seed + self.epoch)
        else:
            rng = random.default_rng(self.seed)
        return rng


class CustomDataset(Dataset):
    """
    A custom dataset for deep time series models, using KnowIts data extraction protocols.

    Parameters
    ----------
    data_extractor : DataExtractor
        Object responsible for extracting data from disk.
    selection_matrix : array
        Matrix defining the selection of instances, slices, and time steps.
    x_map : array, shape=[n_in_components,]
        An array that contains the indices of BaseDataset.components that correspond to input components.
    y_map : array, shape=[n_out_components,]
        An array that contains the indices of BaseDataset.components that correspond to output components.
    x_scaler : object
        The scaler fitted to the training set input features.
    y_scaler : object
        The scaler fitted to the training set output features.
    in_chunk : list, shape=[2,]
        A list of two integers [a, b] for which a <= b,
        defining the time steps of in_components for each prediction point.
    out_chunk : list, shape=[2,]
        A list of two integers [a, b] for which a <= b,
        defining the time steps of out_components for each prediction point.
    padding_method : str
        The method to pad model inputs with.
        Options can be found at (https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
        See heading for description.
    preload : bool, default=False
        Whether to preload the dataset into memory.

    """
    def __init__(self, data_extractor, selection_matrix,
                 x_map, y_map,
                 x_scaler, y_scaler,
                 in_chunk, out_chunk,
                 padding_method,
                 preload: bool = False) -> None:

        self.data_extractor = data_extractor
        self.selection_matrix = selection_matrix
        self.x_map = x_map
        self.y_map = y_map
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.in_chunk = in_chunk
        self.out_chunk = out_chunk
        self.padding_method = padding_method
        self.preload = preload

        self.preloaded_slices = {}
        if preload:
            logger.info("Preloading relevant slices into memory. This could take a while, but spead up actual training.")
            instances = unique(self.selection_matrix[:, 0], axis=0)
            for i in instances:
                slices = unique(self.selection_matrix[self.selection_matrix[:, 0] == i, 1])
                for s in slices:
                    self.preloaded_slices[(i, s)] = self.data_extractor.slice(i, s)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
        --------
        int
            The number of samples in the dataset.
        """
        return self.selection_matrix.shape[0]

    def __getitem__(self, idx: int | list | Tensor) -> dict:
        """Return a single sample from the dataset at the given index.

        Parameters
        ----------
        idx : int | list | Tensor
            The index of the sample to retrieve or list of indices.

        Returns
        -------
        dict[str, any]
            A dictionary containing
                -   'x' (Tensor): the input features,
                -   'y' (Tensor): the one-hot encoded labels,
                -   's_id' (int): the sample ID 's_id'.
        """

        if type(idx) is list:
            idx_list = idx
        elif is_tensor(idx):
            idx_list = idx.tolist()
        else:
            # assumes idx is an integer
            idx_list = [idx]

        input_x = []
        output_y = []
        for pp in idx_list:
            # get sample selection
            selection = self.selection_matrix[pp]

            # get relevant slice information
            if self.preload:
                slice_vals = self.preloaded_slices[(selection[0], selection[1])].to_numpy()
            else:
                slice_vals = self.data_extractor.slice(selection[0], selection[1]).to_numpy()

            # get input values and pad if necessary
            x_vals = self._sample_and_pad(slice_vals, selection, self.in_chunk, self.x_map, self.padding_method)

            # get output values (no padding allowed)
            y_vals = slice_vals[selection[2] + self.out_chunk[0]: selection[2] + self.out_chunk[1] + 1, self.y_map]

            # scale inputs and outputs if applicable
            input_x.append(self.x_scaler.transform(x_vals))
            output_y.append(self.y_scaler.transform(y_vals))

        if type(idx) is list or is_tensor(idx):
            input_x = array(input_x)
            output_y = array(output_y)
        else:
            input_x = input_x[0]
            output_y = output_y[0]

        return self._package_output(input_x, output_y, idx)

    @staticmethod
    def _sample_and_pad(slice_vals, selection, s_chunk, s_map, pad_mode):

        far_left = 0
        far_right = slice_vals.shape[0]
        left = selection[2] + s_chunk[0]
        right = selection[2] + s_chunk[1]

        vals = slice_vals[max(far_left, left): min(right, far_right) + 1, s_map]

        if left < far_left:
            pw = ((far_left - left, 0), (0, 0))
            vals = pad(vals, pad_width=pw, mode=pad_mode)
        if right >= far_right:
            pw = ((0, right - far_right + 1), (0, 0))
            vals = pad(vals, pad_width=pw, mode=pad_mode)

        # vals = random.rand(49, 3)

        return vals

    @staticmethod
    def _package_output(input_x, output_y, idx):
        """ Package the sample values for a basic regression problem. """
        sample = {'x': from_numpy(input_x).float(),
                  'y': from_numpy(output_y).float(), 's_id': idx}
        return sample


class CustomClassificationDataset(CustomDataset):
    """A custom dataset for deep time series classification models, using KnowIts data extraction protocols.
    Inherits from CustomDataset.
    """
    class_set = {}
    class_counts = {}

    def __init__(self, data_extractor, selection_matrix,
                 x_map, y_map,
                 x_scaler, y_scaler,
                 in_chunk, out_chunk,
                 class_set, padding_method,
                 preload: bool = False) -> None:

        if out_chunk[0] != out_chunk[1]:
            logger.error('Currently, KnowIt can only perform classification at one specific time step at a time. '
                         'Please change the out_chunk %s argument to reflect this. Both values must match.',
                         str(out_chunk))
            exit(101)

        super().__init__(data_extractor, selection_matrix, x_map, y_map,
                         x_scaler, y_scaler,
                         in_chunk, out_chunk,
                         padding_method, preload)
        self.class_set = class_set

    def _package_output(self, input_x, output_y, idx):
        """ Package the sample values for a basic classification problem. """

        if len(output_y.shape) == 2:
            new_output_y = zeros_tensor(len(self.class_set))
            new_output_y[self.class_set[output_y.item()]] = 1
            output_y = new_output_y
        elif len(output_y.shape) == 3:
            new_output_y = zeros_tensor(size=(output_y.shape[0], len(self.class_set)))
            new_output_y[:, output_y] = 1
            output_y = new_output_y

        sample = {'x': from_numpy(input_x).float(),
                  'y': output_y, 's_id': idx}

        return sample

