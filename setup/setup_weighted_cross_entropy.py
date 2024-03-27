__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains a function to prepare a weighted CE function.'

# external imports
import torch

# internal imports
from helpers.logger import get_logger
logger = get_logger()


def proc_weighted_cross_entropy(task, device, class_counts):

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





