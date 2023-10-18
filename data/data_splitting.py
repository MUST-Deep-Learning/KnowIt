__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataSplitter class for Knowit.'

"""
---------------
DataSplitter
---------------

This module selects a set of prediction points from the_data structure that are appropriate
for model training and then splits them into a train, validation, and evaluation set.
A prediction point is appropriate for model training if:
    1. Its corresponding slice is larger or equal to the minimum slice length.
    2. There are no missing values in the features corresponding to the model outputs.
    3. The out_chunk corresponding to it is within its corresponding slice.
    
------------------
Selection matrix
------------------

Appropriate prediction points are stored in a `selection matrix'.
This is an n-row, 3-column matrix where each row corresponds to a prediction point.
The value in each column indicates:
    -   The relative position, in BaseDataset.instances, of the instance to which 
            the prediction point belongs.
    -   The relative position, within said instance, of the slice to which 
            the prediction point belongs.
    -   The relative position, within said slice, of the timestep at which 
            a prediction is to be made. 
This is also the format that the data splits are stored in (i.e. one matrix for each split).
            
------------------
Splitting
------------------

Given a selection matrix for all available prediction points, they need to be split into 3 
sets according to the proportions given by the `portions' tuple. In practice, they are all 
split train-then-valid-then-eval, however the elements being split on, and the order of these 
elements are defined by the 'method' argument.
    -   random:                 : split on timesteps in random order
    -   chronological (default) : split on timesteps in chronological order
    -   instance-random         : split on instances in random order
    -   instance-chronological  : split on instances in chronological order
            The ordering is based on the first time step in the instance.
    -   slice-chronological     : split on slices in chronological order
            The ordering is based on the first time step in the slice.
    -   slice-random            : split on slices in random order

This means that 'portions' are defined i.t.o the specific 'method'. For examples: 
portions=(0.8, 0.1, 0.1) and method=instance-random means that the prediction points of a random 80% of instances,
will constitute training data, another random 10% will constitute valid data, and another 10% will constitute 
eval data. Alternatively, if method=random, a random 80% of time steps (regardless of instance or slice), 
will constitute training data, etc.

 
------------------
Limiting
------------------           
The limiting of the data occurs after the ordering of the elements (instance, slice, or timestep) and before 
the data is split. The value of 'limit' is then also defined i.t.o 'method'. The data will be limited to 
 the first n elements if limit=n. E.g. if limit=500 and method=instance-random, the data is limited to the first 
 500 random instances.  

"""

# external imports
from numpy import (array, random, argwhere, isnan,
                   count_nonzero, vstack, concatenate,
                   argsort, arange, unique, append, floor)

# internal imports
from helpers.logger import get_logger

logger = get_logger()


class DataSplitter:

    def __init__(self, the_data: dict, method: str,
                 portions: tuple, instances: list, limit: int, y_map: array,
                 out_chunk: list, min_slice: int):

        """
        Instantiate a DataSplitter object and perform its main operations.

        This constructor initializes a DataSplitter object with the provided arguments and executes the main
        operations of data splitting based on the defined method, portions, and limits.

        Parameters:
            the_data (dict): Raw data structure as defined in BaseDataset.
            method (str): Method for data splitting (e.g., 'random', 'chronological').
            portions (tuple): List of portions for training, validation, and evaluation datasets.
            instances (list): List of instances in the_data.
            limit (int): Maximum number of elements (depends on method) to consider.
            y_map (array): Mapping for target data components.
            out_chunk (list): Target data chunk size.
            min_slice (int): Minimum slice size.

        Main Operations:
            - Find appropriate prediction points and select target data.
            - Split and limit the data according to defined portions, method, and limit.

        """

        # log given values
        self.the_data = the_data
        self.method = method
        self.portions = portions
        self.instances = instances
        self.limit = limit
        self.y_map = y_map
        self.out_chunk = out_chunk
        self.min_slice = min_slice

        # check that defined portions are valid
        if sum(self.portions) != 1.0:
            logger.error('Split portions do not add up to one.')
            exit(101)

        # 1. Find all appropriate prediction points
        prediction_points, times = self.__select_prediction_points(self.the_data, self.instances,
                                                self.y_map, self.out_chunk, self.min_slice)

        # [instance, slice, relative position]

        # 2. Split (and limit)
        self.train_points, self.valid_points, self.eval_points = self.__do_split(prediction_points, times,
                                                                   self.portions,
                                                                   self.method, self.limit)

    def get_selection(self):
        """ Returns the obtained data splits as a dictionary of selection matrices. """
        return {'train': self.train_points,
                'valid': self.valid_points,
                'eval': self.eval_points}

    @staticmethod
    def __sample_and_stack(prediction_points: array,
                           start_stop_indxs: array,
                           elements: array):
        """
        This function samples blocks from 'prediction_points' based on the 'elements' provided
        and stacks these sampled blocks vertically to create a new array.

        Parameters:
            prediction_points (array): The data from which blocks are sampled.
            start_stop_indxs (array): Start and stop indices of each element.
            elements (array): Selected element indices.

        Returns:
            blocks (array): A stacked array containing the sampled blocks.

        """
        blocks = [prediction_points[start_stop_indxs[i, 0]:start_stop_indxs[i, 1]] for i in elements]
        blocks = vstack(blocks)
        return blocks

    @staticmethod
    def __ordered_split(elements: array, portions: tuple):
        """ Split the elements array (1D array) into 3 based on portions, in order. """
        train_size = int(floor(portions[0] * len(elements)))
        eval_size = int(floor(portions[2] * len(elements)))
        eval_elements = elements[-eval_size:]
        train_elements = elements[:train_size]
        valid_elements = elements[train_size:-eval_size]

        return train_elements, valid_elements, eval_elements

    @staticmethod
    def __split_indices_on(prediction_points: array, tag: str):
        """
        This function takes a selection matrix (prediction_points) and splits it into blocks based on
        the specified criteria (tag), such as 'instance', 'slice', or 'timestep'.
        It then returns the start and stop indices for each block.

        Parameters:
            prediction_points (array-like): The selection matrix to split.
            tag (str): The criteria for splitting (e.g., 'instance', 'slice', 'timestep').

        Returns:
            start_stop_indxs (array-like): An array of start-and-stop indices for each split block.

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
    def __do_split(prediction_points: array, times: array,
                   portions: tuple, method: str, limit: int):

        """ Performs the overall splitting and limiting of prediction points. """

        # 1. Get the start stop indices based on method level

        level = None
        if method in ('slice-random', 'slice-chronological'):
            level = 'slice'
        elif method in ('instance-random', 'instance-chronological'):
            level = 'instance'
        elif method in ('random', 'chronological'):
            level = 'timestep'

        start_stop_indxs = DataSplitter.__split_indices_on(prediction_points, level)

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

        train_elements, valid_elements, eval_elements = DataSplitter.__ordered_split(elements, portions)

        if len(train_elements) < 1 or len(valid_elements) < 1 or len(eval_elements) < 1:
            logger.error('Data segments too limited for selected splitting method %s', method)
            exit(101)

        # 5. Sample the split points based on the separated elements and start_stop_indxs

        train_points = DataSplitter.__sample_and_stack(prediction_points, start_stop_indxs, train_elements)
        valid_points = DataSplitter.__sample_and_stack(prediction_points, start_stop_indxs, valid_elements)
        eval_points = DataSplitter.__sample_and_stack(prediction_points, start_stop_indxs, eval_elements)

        return train_points, valid_points, eval_points

    @staticmethod
    def __select_prediction_points(data: dict, instances: list, y: array,
                                   out_chunk: list, min_slice: int):

        """ Construct the full selection matrix (all appropriate prediction points). """

        def no_nan_mask(arr, chunk):
            mask = ~isnan(arr).any(axis=1)
            chunk = array(chunk)
            negatives = chunk[chunk < 0]
            positives = chunk[chunk > 0]
            if len(negatives) > 0:
                mask[:abs(min(negatives))] = False
            if len(positives) > 0:
                mask[-max(positives):] = False

            return mask

        def constant_col(count, val):
            return array([val for t in range(count)])

        def relative_col(mask):
            vals = argwhere(mask)
            return vals.squeeze()

        times = []
        prediction_points = []
        ignored_slices = 0
        for i in range(len(instances)):
            s = 0
            for slice in data[instances[i]]:
                if min_slice is None or min_slice < slice['d'].shape[0]:
                    mask = no_nan_mask(slice['d'][:, y], out_chunk)
                    times.append(slice['t'][mask])
                    nn_count = count_nonzero(mask)
                    prediction_points.append(vstack((constant_col(nn_count, i),
                                              constant_col(nn_count, s),
                                              relative_col(mask))))
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
