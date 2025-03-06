"""Functions used to select prediction points for interpretation. """

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains functions to select the appropriate prediction points for interpretation.'

# external imports
import numpy as np
import os

# internal imports
from helpers.file_dir_procs import load_from_path
from helpers.logger import get_logger
logger = get_logger()


def get_interpretation_inx(interpretation_args: dict, model_args: dict, predictions_dir: str) -> tuple:
    """
    Determines the indices for interpretation based on specified criteria.

    This function identifies the range of indices in a dataset for interpretation, as specified
    by `interpretation_args`, and based on the dataset split sizes in `model_args`. The range of
    indices is chosen based on the desired subset (`train`, `valid`, or `eval`), the size, and the
    selection method (`random`, `all`, `success`, or `failure`).

    Parameters
    ----------
    interpretation_args : dict
        Dictionary containing arguments related to interpretation settings. Keys include:
        - 'interpretation_set': Specifies the data split to use, one of {'train', 'valid', 'eval'}.
        - 'size': The number of samples to interpret.
        - 'selection': The selection criteria, one of {'random', 'all', 'success', 'failure'}.
    model_args : dict
        Dictionary containing model configuration and data split sizes. Relevant structure:
        - data_dynamics : dict containing keys:
            - 'train_size': Size of the training dataset.
            - 'valid_size': Size of the validation dataset.
            - 'eval_size': Size of the evaluation dataset.
    predictions_dir : str
        Directory where model prediction results are stored, used for certain selection criteria.

    Returns
    -------
    tuple
        A tuple representing the start and end indices of the selected data range.

    Raises
    ------
    SystemExit
        If `interpretation_args['interpretation_set']` is not a recognized set, if the requested
        interpretation size exceeds the selected set size, or if an invalid selection method
        is specified in `interpretation_args['selection']`.

    Notes
    -----
    - For the 'random' selection, a random starting index is chosen within the set size limits.
    - 'all' selection returns the full range of the set.
    - 'success' and 'failure' selections are handled by `select_chunk()` using the predictions data
      in `predictions_dir`.

    """
    inx = 0

    if interpretation_args['interpretation_set'] == 'train':
        set_size = model_args['data_dynamics']['train_size']
    elif interpretation_args['interpretation_set'] == 'valid':
        set_size = model_args['data_dynamics']['valid_size']
    elif interpretation_args['interpretation_set'] == 'eval':
        set_size = model_args['data_dynamics']['eval_size']
    else:
        logger.error('Unknown interpretation_set %s', interpretation_args['interpretation_set'])
        exit(101)

    size = interpretation_args['size']
    if set_size < size:
        logger.error('Size of desired interpretation %s larger than desired set %s.',
                     size,
                     interpretation_args['interpretation_set'])
        exit(101)

    if interpretation_args['selection'] == 'random':
        start = np.random.randint(0, set_size - size)
        inx = (start, start + size)
    elif interpretation_args['selection'] == 'all':
        inx = (0, set_size)
    elif interpretation_args['selection'] == 'success':
        inx = select_chunk(interpretation_args, model_args, selection='success', predictions_dir=predictions_dir)
    elif interpretation_args['selection'] == 'failure':
        inx = select_chunk(interpretation_args, model_args, selection='failure', predictions_dir=predictions_dir)
    else:
        logger.error('Invalid interpretation selection %s.', interpretation_args['selection'])
        exit(101)

    return inx


def select_chunk(interpretation_args: dict, model_args: dict, selection: str, predictions_dir: str) -> tuple:
    """
    Selects a contiguous chunk of data based on model performance.

    This function identifies a segment of data based on Mean Absolute Error (MAE) performance scores,
    either selecting the "best" (success) or "worst" (failure) performing chunk of a specified size.
    It does so by convolving the MAE values to get a moving average performance and finding either
    the minimum (for 'success') or maximum (for 'failure') chunk.

    Parameters
    ----------
    interpretation_args : dict
        Dictionary containing arguments related to interpretation settings, including:
        - 'size': Integer specifying the size of the chunk to select.
    model_args : dict
        Dictionary containing model configuration settings.
    selection : str
        Specifies the chunk selection criteria, one of {'success', 'failure'}.
        - 'success': Selects the chunk with minimum average MAE.
        - 'failure': Selects the chunk with maximum average MAE.
    predictions_dir : str
        Directory where prediction results are stored, used to calculate performance scores.

    Returns
    -------
    tuple
        A tuple containing the start and end indices of the selected chunk.

    Raises
    ------
    ValueError
        If `selection` is not one of {'success', 'failure'}.

    Notes
    -----
    This function uses `get_mae_performance()` to retrieve MAE scores, and then convolves these
    scores with a window of specified size to determine average performance over each segment.

    """
    chunk = interpretation_args['size']

    mae = get_mae_performance(interpretation_args, model_args, predictions_dir)
    chunk_perf = np.convolve(mae, np.ones(chunk) / chunk, mode='valid')

    if selection == 'success':
        select_chunk_inx = np.argmin(chunk_perf)
    else:
        select_chunk_inx = np.argmax(chunk_perf)

    select_chunk = (select_chunk_inx, select_chunk_inx + chunk)

    return select_chunk


def get_mae_performance(interpretation_args: dict, model_args: dict, predictions_dir: str) -> np.ndarray:
    """
    Computes the Mean Absolute Error (MAE) for model predictions on a specified data set.

    This function calculates the MAE between predictions and targets for each prediction point
    in the given dataset. The predictions and targets are retrieved from the `predictions_dir`
    based on the interpretation set specified in `interpretation_args`. The resulting MAE values
    are returned as an array, where each element represents the MAE for a corresponding data point.

    Parameters
    ----------
    interpretation_args : dict
        Dictionary containing settings for the interpretation, including:
        - 'interpretation_set': str, the name of the dataset split to evaluate,
          e.g., 'train', 'valid', or 'eval'.
    model_args : dict
        Dictionary containing model-specific configuration details.
    predictions_dir : str
        Path to the directory where model predictions and targets are stored.

    Returns
    -------
    np.ndarray
        A numpy array of MAE values for each evaluated data point.

    Notes
    -----
    - The function calls `get_predictions()` to retrieve points, predictions, targets, and timestamps.
    - It computes MAE as `mean(abs(predictions - targets))` for each data point.
    - This function currently overlaps with `viz.set_predictions` functionality, which should be refactored later.
    """
    # TODO: This function has overlap with viz.set_predictions. Need to refactor later.

    points, predictions, targets, timestamps = get_predictions(predictions_dir,
                                                   interpretation_args['interpretation_set'], model_args)

    performance = []
    for p in points:
        a = predictions[p]
        b = targets[p]
        perf = np.mean(np.abs(a - b))
        performance.append(perf)

    performance = np.array(performance)

    return performance


def get_predictions(predictions_dir: str, data_tag: str, model_args: dict) -> tuple:
    """
    Retrieve model predictions, target values, and associated timestamps for a specified dataset portion.

    This function loads prediction data from a directory and organizes it based on batch files
    corresponding to the specified `data_tag`. It verifies that all expected prediction points
    are available, ensuring complete data for interpretation or evaluation.

    Parameters
    ----------
    predictions_dir : str
        Directory containing saved prediction batch files and an index mapping file.
    data_tag : str
        Name of the dataset split for which predictions are retrieved (e.g., 'train', 'valid', 'eval').
    model_args : dict
        Model configuration and metadata, including dataset sizes and settings for the specified portion.

    Returns
    -------
    tuple
        - points : np.ndarray
            Sorted array of indices for the prediction points.
        - predictions : dict
            Dictionary mapping each index to the model's predicted values.
        - targets : dict
            Dictionary mapping each index to the actual target values.
        - timestamps : dict
            Dictionary mapping each index to the corresponding timestamp from the dataset.

    Raises
    ------
    SystemExit
        If the `predictions_dir` does not exist or if the required prediction points are incomplete.
    """

    ist_file_name = f'_{data_tag}-ist_inx_dict.pickle'
    if not os.path.exists(os.path.join(predictions_dir, ist_file_name)):
        logger.error('No %s set predictions generated at %s', data_tag, predictions_dir)
        logger.error('Please generate predictions before interpreting them. e.g. KI.generate_predictions(model_name, kwargs)')
        exit(101)

    batch_paths = [os.path.join(predictions_dir, b) for b in os.listdir(predictions_dir) if
                   b.startswith(f'{data_tag}-batch')]

    ist_path = os.path.join(predictions_dir, ist_file_name)
    ist_values, _ = load_from_path(ist_path)

    predictions = {}
    targets = {}
    timestamps = {}
    for b in batch_paths:
        batch = load_from_path(b)
        s_inx = batch[0]
        for p in range(len(s_inx)):
            s = s_inx[p].item()
            y_hat = batch[1][p]
            y = batch[2][p]
            predictions[s] = y_hat
            targets[s] = y
            timestamps[s] = ist_values[s]

    if len(predictions) != model_args['data_dynamics'][data_tag + '_size']:
        logger.error('Could not find all relevant prediction points for interpretation selection.')
        exit(101)

    points = np.array(list(predictions.keys()))
    points = np.sort(points)
    return points, predictions, targets, timestamps
