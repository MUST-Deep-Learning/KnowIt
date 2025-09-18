"""Functions used to select prediction points for interpretation. """

__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains functions to select the appropriate prediction points for interpretation.'

# external imports
import numpy as np
import os

# internal imports
from helpers.file_dir_procs import load_from_path
from helpers.logger import get_logger
logger = get_logger()


def get_interpretation_inx(data_tag: str, data_selection_matrix: np.array,
                           i_size: int, i_selection_tag: str,
                           predictions_dir: str, seed: int) -> tuple:
    """
    Determines the indices of prediction points to be interpreted based on selection criteria.

    Parameters
    ----------
    data_tag : str
        The dataset split, which can be 'train', 'valid', or 'eval'.
    data_selection_matrix : np.array, shape=[(num_prediction_points, 3]
        The selection matrix for the given data split.
    i_size : int
        The number of prediction points to be interpreted.
    i_selection_tag : str
        The selection strategy, which can be 'all', 'random', 'success', or 'failure'.
    predictions_dir : str
        The path to the directory containing model prediction files.
    seed : int
        The random seed for reproducibility when selecting random points.

    Returns
    -------
    tuple
        A tuple containing:
        - inx : tuple
            The start and end indices of the selected prediction points.
        - points : np.array
            The selected prediction point indices. This is their positions in the selection matrix.
        - predictions : dict
            The model predictions corresponding to each point.
        - targets : dict
            The ground-truths corresponding to each point.
        - timestamps : dict
            The associated IST values.

    Raises
    ------
    SystemExit
        If `i_size` is greater than the available data size or if no valid contiguous blocks
        are found for selection.
    """

    data_size = len(data_selection_matrix)

    if data_size < i_size:
        logger.error('Interpretation size %s > %s-set size %s.',i_size, data_tag, data_size)
        exit(101)

    if i_selection_tag == 'all':
        # select all points in data split for interpretation
        point_ids = []
        s_blocks = _get_contiguous_subblocks(data_selection_matrix)
        for s_block in s_blocks:
            point_ids.extend(s_block)
    else:
        # find contiguous blocks
        s_blocks = _get_contiguous_subblocks(data_selection_matrix)
        selected_blocks = []
        for s_block in s_blocks:
            if len(s_block) >= i_size:
                selected_blocks.append(s_block)
        del s_blocks
        if len(selected_blocks) == 0:
            logger.error('No contiguous blocks within %s-set larger or equal to desired interpretation set size %s.',
                         data_tag, i_size)
            exit(101)
        if i_selection_tag == 'random':
            # select a random contiguous subset of a random contiguous block
            rng = np.random.default_rng(seed)
            block = selected_blocks[rng.choice(len(selected_blocks))]
            start = rng.integers(0, len(block) + 1 - i_size)
            point_ids = block[start:start + i_size]
        else:
            # select a contiguous subset of a contiguous block based on a specific criterion
            point_ids = _special_select(i_size, i_selection_tag, predictions_dir, selected_blocks, data_tag, data_size)

    points, predictions, targets, timestamps = get_predictions(predictions_dir, data_tag, data_size, point_ids)

    return points, predictions, targets, timestamps


def get_predictions(predictions_dir: str,
                    data_tag: str,
                    data_size: int,
                    point_ids: list = None,
                    w_mae: bool = False) -> tuple:
    """
    Loads and retrieves model predictions, targets, and timestamps for a given dataset split.

    Parameters
    ----------
    predictions_dir : str
        Path to the directory containing the model prediction files.
    data_tag : str
        The dataset split, which can be 'train', 'valid', or 'eval'.
    data_size : int
        The expected number of prediction points in the dataset split.
    point_ids : list, optional
        A list of specific data points to retrieve. If None, all available points are loaded.
    w_mae : bool, default=False
        Whether to compute and return the mean absolute error (MAE) for each point.

    Returns
    -------
    tuple
        - points : np.array
            Sorted array of selected prediction points.
        - predictions : dict
            Dictionary mapping each point ID to its predicted value.
        - targets : dict
            Dictionary mapping each point ID to its ground-truth target value.
        - timestamps : dict
            Dictionary mapping each point ID to its associated timestamp.
        - mae : dict, optional
            Dictionary mapping each point ID to its computed MAE (only if `w_mae=True`).

    Raises
    ------
    SystemExit
        If no prediction files exist for the given dataset split.
        If not all relevant prediction points are found when `point_ids` is None.
    """

    ist_file_name = f'_{data_tag}-ist_inx_dict.pickle'
    if not os.path.exists(os.path.join(predictions_dir, ist_file_name)):
        logger.error('No %s set predictions generated at %s', data_tag, predictions_dir)
        logger.error('Please generate predictions before interpreting them. e.g. KI.generate_predictions(model_name, kwargs)')
        exit(101)

    batch_paths = {}
    for b in os.listdir(predictions_dir):
        batch_prefix = f'{data_tag}-batch_'
        if b.startswith(batch_prefix):
            batch_id = int(b.split(batch_prefix)[1].split('.pickle')[0])
            batch_paths[batch_id] = os.path.join(predictions_dir, b)

    ist_path = os.path.join(predictions_dir, ist_file_name)
    ist_values, batch_map = load_from_path(ist_path)

    predictions = {}
    targets = {}
    timestamps = {}
    if point_ids is None:
        for b in batch_paths:
            batch = load_from_path(batch_paths[b])
            s_inx = batch[0]
            for p in range(len(s_inx)):
                s = s_inx[p]
                if len(s_inx.shape) == 1:
                    # assumes non-variable length data
                    s = s.item()
                    predictions[s] = batch[1][p]
                    targets[s] = batch[2][p]
                    timestamps[s] = ist_values[s]
                else:
                    # assumes variable length data
                    for ss in s:
                        ss = ss.item()
                        position = np.argwhere(batch[0][p] == ss).item()
                        predictions[ss] = batch[1][p, position].reshape(1, -1)
                        targets[ss] = batch[2][p, position].reshape(1, -1)
                        timestamps[ss] = ist_values[ss]
    else:
        for point in point_ids:
            batch_id = batch_map[point]
            batch_path = batch_paths[batch_id]
            batch = load_from_path(batch_path)
            if len(batch[0].shape) == 1:
                # assumes non-variable length data
                position = np.argwhere(batch[0] == point).item()
                predictions[point] = batch[1][position]
                targets[point] = batch[2][position]
                timestamps[point] = ist_values[point]
            else:
                # assumes variable length data
                # TODO: revise this loop (it is used for interpretation of variable length data)
                for s in batch[0]:
                    position = np.argwhere(batch[0] == point)
                    predictions[point] = batch[1][position[0], position[1]].reshape(1, -1)
                    targets[point] = batch[2][position[0], position[1]].reshape(1, -1)
                    timestamps[point] = ist_values[point]

    if point_ids is None and len(predictions) != data_size:
        logger.error('Could not find all relevant prediction points for interpretation selection.')
        exit(101)

    points = np.array(list(predictions.keys()))
    points = np.sort(points)

    if w_mae:
        mae = {}
        for p in points:
            a = predictions[p]
            b = targets[p]
            mae[p] = np.mean(np.abs(a - b))
        return points, predictions, targets, timestamps, mae
    else:
        return points, predictions, targets, timestamps


def _get_contiguous_subblocks(selection: np.array):
    # TODO: Borrowed and modified from prepared_dataset.CustomSampler._create_contiguous_batches, to refactor.
    inx = np.expand_dims(np.arange(len(selection)), 1)
    selection = np.concatenate((selection, inx), axis=1)
    s_blocks = []
    for s in np.unique(selection[:, :2], axis=0):
        idx = np.where((selection[:, 0] == s[0]) & (selection[:, 1] == s[1]))[0]
        s_block = selection[idx]
        s_block = s_block[s_block[:, 2].argsort()]
        breakpoints = np.where(np.diff(s_block[:, 2]) != 1)[0] + 1
        s_block = np.split(s_block[:, 3], breakpoints)
        s_blocks.extend(s_block)
    return s_blocks


def _special_select(i_size: int,
                    selection_tag: str,
                    predictions_dir: str,
                    selected_blocks: list,
                    data_tag: str,
                    data_size: int) -> tuple:
    """
    Selects a contiguous subset of a block based on a success or failure criterion.

    Parameters
    ----------
    i_size : int
        The number of prediction points to be selected.
    selection_tag : str
        Criterion for selection; can be 'success' (minimize MAE) or 'failure' (maximize MAE).
    predictions_dir : str
        Path to the directory containing the model prediction files.
    selected_blocks : list
        A list of contiguous blocks of data points to consider for selection.
    data_tag : str
        The dataset split, which can be 'train', 'valid', or 'eval'.
    data_size : int
        The expected number of prediction points in the dataset split.

    Returns
    -------
    array
        An array representing prediction point indices of the selected contiguous subset.

    Raises
    ------
    SystemExit
        If an unknown `selection_tag` is provided.
    """

    points, predictions, targets, timestamps, mae = get_predictions(predictions_dir, data_tag, data_size, w_mae=True)

    chunk_score = []
    chunk = []
    for block in selected_blocks:
        relevant_mae = np.array([mae[b] for b in block])
        relevant_mmae = np.convolve(relevant_mae, np.ones(i_size) / i_size, mode='valid')
        consider_range = np.arange(0, len(relevant_mae) + 1 - i_size)
        if selection_tag == 'success':
            select_chunk_inx = np.argmin(relevant_mmae[consider_range])
        elif selection_tag == 'failure':
            select_chunk_inx = np.argmax(relevant_mmae[consider_range])
        else:
            logger.error('Unknown selection tag %s.', selection_tag)
            exit(101)
        chunk_score.append(relevant_mmae[select_chunk_inx])
        chunk.append(block[select_chunk_inx:select_chunk_inx + i_size])
    if selection_tag == 'success':
        best_chunk = np.argmin(np.array(chunk_score))
    elif selection_tag == 'failure':
        best_chunk = np.argmax(np.array(chunk_score))
    else:
        logger.error('Unknown selection tag %s.', selection_tag)
        exit(101)
    selected_points = chunk[best_chunk]
    return selected_points

