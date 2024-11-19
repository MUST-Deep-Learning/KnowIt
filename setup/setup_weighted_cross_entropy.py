"""Functions used to facilitate weighted cross entropy. """

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains a function to prepare a weighted CE function.'

# external imports
import torch

# internal imports
from helpers.logger import get_logger
logger = get_logger()


def proc_weighted_cross_entropy(task: str, device: str, class_counts: dict) -> dict:
    """
    Configure a weighted cross-entropy loss function for classification tasks based on class counts.

    This function calculates weights for each class inversely proportional to its count, helping to
    balance classes in tasks where class distributions are uneven. The weights are normalized so
    that the maximum weight equals 1. This weighted loss function is only available for
    classification tasks.

    Parameters
    ----------
    task : str
        The task type, expected to be 'classification' to support weighted loss.
    device : str
        Device on which the weights will be used ('gpu' for CUDA compatibility, otherwise CPU).
    class_counts : dict
        Dictionary where keys are class labels and values are counts of samples in each class.

    Returns
    -------
    dict
        Dictionary containing the loss function configuration with normalized class weights.

    Raises
    ------
    SystemExit
        If the task is not 'classification' or if `class_counts` is empty.
    """

    if task != 'classification' or not class_counts:
        logger.error('Weighted loss function only supported for classification tasks.')
        exit(101)

    cc = torch.tensor([class_counts[c] for c in class_counts])
    weights = torch.sum(cc) / cc
    weights /= torch.max(weights)

    if device == 'gpu':
        weights = weights.to('cuda')

    loss_fn = {'cross_entropy': {'weight': weights}}
    return loss_fn





