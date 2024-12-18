"""
---------------
DataSplitter
---------------

This module selects a set of prediction points from "the_data" structure (see ``KnowIt.Basedataset``)
that are appropriate for model training and then splits them into a train, validation, and evaluation set.
A prediction point is appropriate for model training if:
    1. Its corresponding slice is larger or equal to the minimum slice length.
    2. There are no missing values in the features corresponding to the model output components.
    3. The out_chunk corresponding to it is within its corresponding slice.

------------------
Selection matrix
------------------

Appropriate prediction points are stored in a "selection matrix".
This is an n-row, 3-column matrix where each row corresponds to a prediction point.
The value in each column indicates:
    1.   The relative position, in ``BaseDataset.instances``, of the instance to which the prediction point belongs.
    2.   The relative position, within said instance, of the slice to which the prediction point belongs.
    3.   The relative position, within said slice, of the timestep at which a prediction is to be made.
This is also the format that the data splits are stored in (i.e. one matrix for each split).
We sometimes refer to the three positions above as the "IST" index, as in Instance-Slice-Time,
of a prediction point or time step.

---------
Splitting
---------

Given a selection matrix for all available prediction points, they need to be split into 3
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

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataSplitter class for KnowIt.'

# external imports
from numpy import (array, random, argwhere, isnan,
                   count_nonzero, vstack, concatenate,
                   argsort, arange, unique, append, floor)

# internal imports
from helpers.logger import get_logger
logger = get_logger()


class DataSplitter:
    """The DataSplitter module is used by PreparedDataset to split the raw data into train/valid/eval sets.

    Parameters
    ----------
    the_data : dict[any, list]
        Raw data structure as defined in BaseDataset.
    method : str
        Method for data splitting. Options are 'random', 'chronological',
        'instance-random', 'instance-chronological', 'slice-random', or 'slice-chronological'.
    portions : tuple, shape=[3,]
        Tuple of three floats representing the portions for training, validation, and evaluation
        datasets respectively. The sum of these portions should be 1.0.
    instances : list
        List of instances in the_data to be considered for splitting.
    limit : int
        Maximum number of elements (depends on method) to consider.
    y_map : array
        Mapping for target data components. This defines the indices desired output components.
    out_chunk : list, shape=[2,]
        Target data chunk parameters.
    min_slice : int
        Minimum slice size.

    Attributes
    ----------
    train_points : array, shape=[n_train_prediction_points, 3]
        The selection matrix corresponding to the train set.
    valid_points : array, shape=[n_valid_prediction_points, 3]
        The selection matrix corresponding to the validation set.
    train_points : array, shape=[n_eval_prediction_points, 3]
        The selection matrix corresponding to the evaluation set.

    Raises
    ------
    ValueError
        If the sum of the portions does not equal 1.0.
    """
    train_points = None
    valid_points = None
    eval_points = None

    def __init__(self, the_data: dict, method: str, portions: tuple,
                 instances: list, limit: int, y_map: array, out_chunk: list, min_slice: int) -> None:
        # check that defined portions are valid
        if abs(1.0 - sum(portions)) > 1e-6:
            logger.error('Split portions do not add up to one.')
            exit(101)

        # 1. Find all appropriate prediction points
        prediction_points, times = self._select_prediction_points(the_data, instances, y_map, out_chunk, min_slice)

        # [instance, slice, relative position]

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
    def _select_prediction_points(data: dict, instances: list, y: array, out_chunk: list, min_slice: int) -> tuple:
        """Construct the full selection matrix for all appropriate prediction points.

        Parameters
        ----------
        data : dict
            A dictionary where keys are instance identifiers and values are lists of slices. Each slice is
            a dictionary containing 'd' (data array) and 't' (time array).

        instances : list
            A list of instance identifiers to process.

        y : array
            An array of column indices specifying which columns in 'd' to check for NaN values.

        out_chunk : list
            A list of integers specifying which portions of the slices to consider when creating the mask.
            Negative values refer to positions from the end of the array.

        min_slice : int
            The minimum size a slice must have to be considered. Slices smaller than this value are ignored.

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

        def no_nan_mask(arr: array, chunk: array) -> array:
            """Create a mask to exclude rows with NaN values and specified edge chunks.

            Parameters
            ----------
            arr : array
                The array to check for NaN values. Each row is considered a separate entity.

            chunk : array
                An array specifying which portions of the array to mask out. Negative values refer to positions
                from the end of the array, while positive values refer to positions from the start.

            Returns
            -------
            array
                A boolean mask array where True indicates valid rows and False indicates rows to be excluded.

            Notes
            -----
                - The function first creates a mask that is True for rows without any NaN values.
                - It then modifies this mask to exclude the specified edge chunks. For example, if `chunk` contains
                  -3, the last three rows will be masked out. If it contains 3, the first three rows will be masked out.

            """
            mask = ~isnan(arr).any(axis=1)
            chunk = array(chunk)
            negatives = chunk[chunk < 0]
            positives = chunk[chunk > 0]
            if len(negatives) > 0:
                mask[:abs(min(negatives))] = False
            if len(positives) > 0:
                mask[-max(positives):] = False

            return mask

        def constant_col(count: int, val: any) -> array:
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

        def relative_col(mask: array) -> array:
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

        times = []
        prediction_points = []
        ignored_slices = 0
        for i in range(len(instances)):
            s = 0
            for slice in data[instances[i]]:
                if min_slice is None or min_slice < slice['d'].shape[0]:
                    slice_mask = no_nan_mask(slice['d'][:, y], out_chunk)

                    # slice_mask = slice_mask * no_nan_mask(slice['d'][:, x], out_chunk)

                    times.append(slice['t'][slice_mask])
                    nn_count = count_nonzero(slice_mask)
                    prediction_points.append(vstack((constant_col(nn_count, i),
                                                     constant_col(nn_count, s),
                                                     relative_col(slice_mask))))
                else:
                    ignored_slices += 1
                s += 1

        if min_slice and ignored_slices > 0:
            logger.warning(str(ignored_slices) + ' slices ignored because they were smaller than min_slice=%s.',
                           str(min_slice))

        prediction_points = concatenate(prediction_points, axis=1)
        prediction_points = prediction_points.transpose()
        times = concatenate(times)

        # targets   [instance, slice, relative position]
        # times     [timestep]

        return prediction_points, times
