"""
------------
DataSplitter
------------

This module selects a set of prediction points from a provided ``DataExtractor`` object (see ``KnowIt.Basedataset``)
that are appropriate for model training and then splits them into a train, validation, and evaluation set.
A prediction point is appropriate for model training if:
    1. Its corresponding slice is larger or equal to the minimum slice length. (optional)
    2. There would be no missing values in the resulting output feature values.
    3. The resulting output window would be within its corresponding slice.
    4. A significant portion (default 0.5) of the resulting input window would be within its corresponding slice.

----------------
Selection matrix
----------------

Appropriate prediction points are stored in a "selection matrix".
This is an n-row, 3-column matrix where each row corresponds to an appropriate prediction point.
The value in each column indicates:
    1.   The ID of the instance to which the prediction point belongs.
    2.   The relative position, within said instance, of the slice to which the prediction point belongs.
    3.   The relative position, within said slice, of the timestep at which a prediction is to be made.
This is also the format that the data splits are stored in (i.e. one matrix for each split).
We sometimes refer to the three positions above as the "IST" index, as in Instance-Slice-Time,
of a prediction point or time step.

---------
Splitting
---------

Given a selection matrix of all available prediction points, they need to be split into 3
sets according to the proportions given by the "portions" tuple. In practice, they are all
split in order (train-then-valid-then-eval), however the elements being split on, and the
order of these elements are defined by the 'method' argument:
    -   random: split on time steps in random order
    -   chronological (default): split on time steps in chronological order
    -   instance-random: split on instances in random order
    -   instance-chronological: split on instances in chronological order. The ordering is based on the first (chronologically) time step in the instance.
    -   slice-chronological: split on slices in chronological order. The ordering is based on the first (chronologically) time step in the slice.
    -   slice-random: split on slices in random order

This means that 'portions' are defined i.t.o the specific 'method'. For examples:
portions=(0.8, 0.1, 0.1) and method=instance-random means that the prediction points of a random 80% of instances,
will constitute training data, another random 10% will constitute validation data, and another 10% will constitute
evaluation data. Alternatively, if method=random, a random 80% of time steps (regardless of instance or slice),
will constitute training data, etc.

Note that this also means that the provided portions do not necessarily correspond to the portions of prediction points,
and therefore they might not directly control the number of examples trained and tested on.
Only for method=random or method=chronological would they.

--------
Limiting
--------
The user has the option of limiting the number of prediction points in the overall dataset.
This is useful for debugging.

The limiting of the data occurs after the ordering of the elements (by instance, slice, or timestep) and before
the data is split. The value of 'limit' is therefore also defined i.t.o 'method'. The data will be limited to
the first n elements if limit=n. E.g. if limit=500 and method=instance-random, the data is limited to the first
500 random instances, if limit=10000 and method=chronological, the data is limited to the first 10000 prediction points
that occur chronologically.

"""

__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataSplitter class for KnowIt.'

# external imports
from numpy import (array, random, argwhere, isnan,
                   count_nonzero, vstack, concatenate,
                   argsort, arange, unique, append, floor, full)

# internal imports
from data.base_dataset import DataExtractor
from helpers.logger import get_logger
logger = get_logger()


class DataSplitter:
    """The DataSplitter module is used by PreparedDataset to split the raw data into train/valid/eval sets,
    and produce the corresponding selection matrices.

    Parameters
    ----------
    data_extractor : DataExtractor
        The data extractor object to read data from disk.
    method : str
        Method for data splitting. Options are 'random', 'chronological',
        'instance-random', 'instance-chronological', 'slice-random', or 'slice-chronological'.
    portions : tuple, shape=[3,]
        Tuple of three floats representing the portions for training, validation, and evaluation
        datasets respectively. The sum of these portions should be 1.0.
    limit : int
        Maximum number of elements (depends on method) to consider.
    x_map : array
        Mapping for input data components. This defines the indices of desired input components.
    y_map : array
        Mapping for target data components. This defines the indices of desired output components.
    in_chunk : list, shape=[2,]
        Input data chunk parameters.
    out_chunk : list, shape=[2,]
        Target data chunk parameters.
    min_slice : int
        Minimum slice length.
    in_portion : float, default=0.5
        The portions of the input window that must still be within a slice to correspond to an appropriate slice.
        Value must be between 0 and 1.
    load_level : str, default='instance'
        What level to load values from disk with.
        If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
        If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.
    custom_splits: dict | None, default=None
        A dictionary defining the custom selection matrices.

    Attributes
    ----------
    train_points : array, shape=[n_train_prediction_points, 3]
        The selection matrix corresponding to the train set.
    valid_points : array, shape=[n_valid_prediction_points, 3]
        The selection matrix corresponding to the validation set.
    eval_points : array, shape=[n_eval_prediction_points, 3]
        The selection matrix corresponding to the evaluation set.

    Raises
    ------
    ValueError
        If the sum of the portions does not equal 1.0.
    """
    train_points = None
    valid_points = None
    eval_points = None

    def __init__(self, data_extractor: DataExtractor,
                 method: str, portions: tuple, limit: int,
                 x_map: array, y_map: array,
                 in_chunk: list, out_chunk: list,
                 min_slice: int, in_portion: float = 0.5,
                 load_level: str = 'instance',
                 custom_splits: dict = None) -> None:

        # check that defined portions are valid
        if abs(1.0 - sum(portions)) > 1e-6:
            logger.error('Split portions do not add up to one.')
            exit(101)

        options = ('random', 'chronological',
                   'instance-random', 'instance-chronological',
                   'slice-random', 'slice-chronological', 'custom')
        if method not in options:
            logger.error('split_method %s is not recognized; must be one of %s', method, options)
            exit(101)

        # 1. Find all appropriate prediction points
        prediction_points, times = self._select_prediction_points(data_extractor,
                                                                  x_map, y_map,
                                                                  in_chunk, out_chunk,
                                                                  min_slice, in_portion,
                                                                  load_level)

        if method == 'custom':

            # Check if all sets are present in the custom split
            missing_split_components = set(custom_splits) - {'valid', 'train', 'eval'}
            if len(missing_split_components) > 0:
                logger.error('Defined set selection %s not in custom splits.',
                                     str(missing_split_components))
                exit(101)

            # Check that the splits are not to be limited
            if limit is not None:
                logger.error("KnowIt does not currently support custom splits and limiting the dataset.")
                exit(101)

            # Remove custom selected prediction points that are not valid
            # @TODO this loop can probably be done much faster, TBD if required.
            converted_prediction_set = set(map(tuple, prediction_points))
            for data_set in ['train', 'valid', 'eval']:
                mask = [tuple(row) in converted_prediction_set for row in custom_splits[data_set]]
                custom_splits[data_set] = custom_splits[data_set][mask]

            self.train_points = custom_splits['train']
            self.valid_points = custom_splits['valid']
            self.eval_points = custom_splits['eval']
        else:
            # 2. Split (and limit)
            self.train_points, self.valid_points, self.eval_points = self._do_split(prediction_points, times,
                                                                                portions, method, limit)

    def get_selection(self) -> dict:
        """Returns the obtained data splits as a dictionary of selection matrices.

        Returns
        -------
        dict[str, array]
            A dictionary containing the following keys:
                - 'train' (array): The selection matrix corresponding to the train set.
                - 'valid' (array): The selection matrix corresponding to the validation set.
                - 'eval' (array): The selection matrix corresponding to the evaluation set.
        """
        return {'train': self.train_points,
                'valid': self.valid_points,
                'eval': self.eval_points}

    @staticmethod
    def _sample_and_stack(prediction_points: array, start_stop_indxs: array, elements: array) -> array:
        """Sample blocks of prediction points from selection matrix.

        This function samples blocks from 'prediction_points' based on the 'elements' provided
        and stacks these sampled blocks vertically to create a new array.

        Parameters
        ----------
        prediction_points : array, shape=[n_prediction_points, 3]
            The data from which blocks are sampled.
        start_stop_indxs : array, shape=[n_elements, 2]
            Start and stop indices of each element.
        elements : array, shape=[n_selected_elements,]
            Selected element indices.

        Returns
        -------
        blocks : array, shape=[n_selected_prediction_points, 3]
            A stacked array containing the sampled blocks.

        """
        blocks = [prediction_points[start_stop_indxs[i, 0]:start_stop_indxs[i, 1]] for i in elements]
        blocks = vstack(blocks)

        return blocks

    @staticmethod
    def _ordered_split(elements: array, portions: tuple) -> tuple:
        """Splits an array of elements into a train, valid, and eval sets, based on the portions provided.

        The train set constitutes the first portions[0]*len(elements) elements.
        The evaluation set constitutes the last portions[2]*len(elements) elements.
        The validation set constitutes the rest.

        Parameters
        ----------
        elements : array, shape=[n_elements,]
            A 1D array of unique scalar values, representing elements to make an ordered split on.
        portions : tuple, shape=[3,]
            A three tuple of floats adding up to 1, representing (train, valid, eval) split portions.

        Returns
        -------
        tuple
            train_elements (array): The first portions[0]*len(elements) elements.
            valid_elements (array): The remaining elements.
            eval_elements (array): The last portions[2]*len(elements) elements.
        """
        train_size = int(floor(portions[0] * len(elements)))
        eval_size = int(floor(portions[2] * len(elements)))
        eval_elements = elements[-eval_size:]
        train_elements = elements[:train_size]
        valid_elements = elements[train_size:-eval_size]

        return train_elements, valid_elements, eval_elements

    @staticmethod
    def _split_indices_on(prediction_points: array, tag: str) -> array:
        """Split selection matrix at one of three levels.

        This function takes a selection matrix (prediction_points) and splits it into blocks based on
        the specified criteria (tag), such as 'instance', 'slice', or 'timestep'.
        It then returns the start and stop indices for each block.

        Parameters
        ----------
        prediction_points : array, shape=[n_prediction_points, 3]
            The selection matrix to split.
        tag : str
            The criteria for splitting (e.g., 'instance', 'slice', 'timestep').

        Returns
        -------
        start_stop_indxs : array, shape=[n_elements, 2]
            An array of start-and-stop indices for each split block.
        """
        if tag == 'instance':
            _, indices = unique(prediction_points[:, 0], return_index=True)
        elif tag == 'slice':
            _, indices = unique(prediction_points[:, :2], return_index=True, axis=0)
        elif tag == 'timestep':
            _, indices = unique(prediction_points[:, :3], return_index=True, axis=0)
        else:
            logger.error("Unknown split-on tag %s.", tag)
            exit(101)

        indices = append(indices, prediction_points.shape[0])

        start_stop_indxs = []
        for i in range(len(indices) - 1):
            start_stop_indxs.append([indices[i], indices[i + 1]])
        start_stop_indxs = array(start_stop_indxs)

        return start_stop_indxs

    @staticmethod
    def _do_split(prediction_points: array, times: array, portions: tuple, method: str, limit: int) -> tuple:
        """Performs the overall splitting and limiting of prediction points based on the specified method.

        Parameters
        ----------
        prediction_points : array, shape=[n_prediction_points, 3]
            An array containing the prediction points to be split (i.e. the full selection matrix).
        times : array, shape=[n_elements,]
            An array containing the times (in as timestamps) corresponding to the prediction points.
        portions : tuple | list, shape=[3,]
            A tuple containing three float values that represent the portions for the training, validation,
            and evaluation splits, respectively. The values should sum to 1.0.
        method : str
            A string specifying the splitting method. It can be one of the following:
            - 'slice-random'
            - 'slice-chronological'
            - 'instance-random'
            - 'instance-chronological'
            - 'random'
            - 'chronological'
        limit : int
            An integer that limits the number of elements to be used for splitting.
            If 0 or None, no limit is applied.

        Returns
        -------
        tuple
            train_points (array): The selection matrix corresponding to the train set.
            valid_points (array): The selection matrix corresponding to the validation set.
            eval_points (array): The selection matrix corresponding to the evaluation set.

        Raises
        ------
        SystemExit
            If an unknown method is provided, if the limit is greater than the available elements, or if the data
            segments are too limited for the selected splitting method.

        Notes
        -----
            - This method categorizes the splitting level based on the provided method and generates start-stop indices.
            - The order of the indices is determined based on whether the method is chronological or random.
            - The elements are limited based on the 'limit' parameter.
            - The elements are split into training, validation, and evaluation sets based on the specified portions.
            - The split points are sampled and stacked based on the separated elements and start-stop indices.
        """

        # 1. Get the start stop indices based on method level

        level = None
        if method in ('slice-random', 'slice-chronological'):
            level = 'slice'
        elif method in ('instance-random', 'instance-chronological'):
            level = 'instance'
        elif method in ('random', 'chronological'):
            level = 'timestep'

        start_stop_indxs = DataSplitter._split_indices_on(prediction_points, level)

        # 2. Define the order of the start-stop indices based on method order

        if method in ('slice-chronological', 'instance-chronological', 'chronological'):
            sub_starts = times[start_stop_indxs[:, 0]]
            elements = argsort(sub_starts)
        elif method in ('slice-random', 'instance-random', 'random'):
            elements = arange(0, start_stop_indxs.shape[0])
            random.shuffle(elements)
        else:
            logger.error("Unknown split method %s.", method)
            exit(101)

        # 3. Limit the elements based on limit argument

        if limit:
            if limit > len(elements):
                logger.error('Data limiter %s larger than %s available %s(s).',
                             str(limit), str(len(elements)), str(level))
                exit(101)
            else:
                logger.warning('Limiting data to %s %s(s).', str(limit), level)
                elements = elements[:limit]

        # 4. Split the remaining elements in three based on portions

        train_elements, valid_elements, eval_elements = DataSplitter._ordered_split(elements, portions)

        if len(train_elements) < 1 or len(valid_elements) < 1 or len(eval_elements) < 1:
            logger.error('Data segments too limited for selected splitting method %s', method)
            exit(101)

        # 5. Sample the split points based on the separated elements and start_stop_indxs

        train_points = DataSplitter._sample_and_stack(prediction_points, start_stop_indxs, train_elements)
        valid_points = DataSplitter._sample_and_stack(prediction_points, start_stop_indxs, valid_elements)
        eval_points = DataSplitter._sample_and_stack(prediction_points, start_stop_indxs, eval_elements)

        return train_points, valid_points, eval_points

    @staticmethod
    def _select_prediction_points(data_extractor: DataExtractor,
                                  x_map: array, y_map: array,
                                  in_chunk: list, out_chunk: list,
                                  min_slice: int, in_portion: float = 0.5,
                                  load_level: str = 'instance') -> tuple:
        """Construct the full selection matrix for all appropriate prediction points available in the given
        DataExtractor.

        Parameters
        ----------
        data_extractor : DataExtractor
            The data extractor object to read data from disk.
        x_map : array
            An array of column indices specifying which columns to check for input edge cases.
        y_map : array
            An array of column indices specifying which columns to check for NaN values and output edge cases.
        in_chunk : list
            A list of two integers specifying which portions of the slices to consider when creating the mask.
            Negative values refer to positions from the end of the array.
        out_chunk : list
            A list of integers specifying which portions of the slices to consider when creating the mask.
            Negative values refer to positions from the end of the array.
        min_slice : int
            The minimum size a slice must have to be considered. Slices smaller than this value are ignored.
        in_portion : float, default=0.5
            The portions of the input window that must still be within a slice to correspond to an appropriate slice.
            Value must be between 0 and 1.
        load_level : str, default='instance'
            What level to load values from disk with.
            If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
            If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.

        Returns
        -------
        tuple
            A tuple containing:
                - prediction_points: A 2D array where each row represents a prediction point
                    with [instance, slice, relative position].
                - times: An array of corresponding times for the prediction points.

        Notes
        -----
            - The method iterates through each instance and each slice within the instance,
                applying a mask to exclude NaN values and short slices.
            - Slices that are smaller than `min_slice` are ignored.
            - The resulting prediction points and times are concatenated and returned.
        """

        def _appropriate_mask(arr: array,
                        in_chunk: list, out_chunk: list,
                        x_map: array, y_map: array,
                        in_portion: float = 0.5) -> array:
            """Create a mask to exclude rows with NaN values and specified edge chunks for output components,
            and softened edge chunks for input components.

            Parameters
            ----------
            arr : array
                The array containing component values over a slice.
            x_map : array
                An array of column indices specifying which columns to check for input edge cases.
            y_map : array
                An array of column indices specifying which columns to check for NaN values and edge cases.
            in_chunk : list
                A list of integers specifying which portions of the slices to consider when creating the mask.
                Negative values refer to positions from the end of the array.
            out_chunk : list
                A list of integers specifying which portions of the slices to consider when creating the mask.
                Negative values refer to positions from the end of the array.
            in_portion : float, default=0.5
                The portions of the input window that must still be within a slice to correspond to an appropriate slice.
                Value must be between 0 and 1.

            Returns
            -------
            array
                A boolean mask array where True indicates valid rows and False indicates rows to be excluded.
            """

            def _trim(mask: array, chunk: array):
                negatives = chunk[chunk < 0]
                positives = chunk[chunk > 0]
                if len(negatives) > 0:
                    mask[:abs(min(negatives))] = False
                if len(positives) > 0:
                    mask[-max(positives):] = False
                return mask

            if in_portion < 0. or in_portion > 1.:
                logger.error('in_portion must be between 0 and 1 for prediction point selection.')
                exit(101)

            y_arr = arr[:, y_map]
            y_mask = ~isnan(y_arr).any(axis=1)
            out_chunk = array(out_chunk)
            y_mask = _trim(y_mask, out_chunk)

            x_arr = arr[:, x_map]
            x_mask = full(x_arr.shape[0], True)
            in_chunk = array(in_chunk)
            in_chunk = floor(in_chunk * in_portion).astype(int)
            x_mask = _trim(x_mask, in_chunk)

            mask = y_mask * x_mask

            return mask

        def _constant_col(count: int, val: any) -> array:
            """Create an array filled with a constant value.

            Parameters
            ----------
            count : int
                The number of elements in the array.

            val : any
                The value to fill the array with.

            Returns
            -------
            array
                An array of length `count`, where each element is `val`.

            """
            return array([val for _ in range(count)])

        def _relative_col(mask: array) -> array:
            """Create an array of relative positions based on a boolean mask.

            Parameters
            ----------
            mask : array
                A boolean mask array where True indicates positions to be included.

            Returns
            -------
            array
                An array of relative positions where the mask is True. If the mask is 1D, the result is a 1D array.
                If the mask is multidimensional, the result will be squeezed to remove single-dimensional entries.

            Notes
            -----
                - The function uses `argwhere` to find the indices of True values in the mask and then squeezes the
                  result to remove any extra dimensions.

            """
            vals = argwhere(mask)
            return vals.squeeze()

        logger.info('Gathering all appropriate prediction points from base dataset.')

        times = []
        prediction_points = []
        ignored_slices = 0
        for i in data_extractor.data_structure:
            if load_level == 'instance':
                instance = data_extractor.instance(i)
                slices = {category: group for category, group in instance.groupby("slice")}
                del instance
            num_slices = len(data_extractor.data_structure[i])
            for s in range(num_slices):
                if load_level == 'instance':
                    slice = slices[s]
                else:
                    slice = data_extractor.slice(i, s)
                if min_slice is None or min_slice < slice.shape[0]:
                    slice_d = slice.to_numpy()
                    slice_t = slice.index.to_numpy()
                    slice_mask = _appropriate_mask(slice_d, in_chunk, out_chunk, x_map, y_map, in_portion)
                    times.append(slice_t[slice_mask])
                    nn_count = count_nonzero(slice_mask)
                    if nn_count > 0:
                        prediction_points.append(vstack((_constant_col(nn_count, i),
                                                         _constant_col(nn_count, s),
                                                         _relative_col(slice_mask))))
                    else:
                        ignored_slices += 1
                else:
                    ignored_slices += 1

        if ignored_slices > 0:
            logger.warning(str(ignored_slices) + ' slices ignored because they were smaller than min_slice=%s or contained no appropriate prediction points.',
                           str(min_slice))

        prediction_points = concatenate(prediction_points, axis=1)
        prediction_points = prediction_points.transpose()
        times = concatenate(times)

        return prediction_points, times
