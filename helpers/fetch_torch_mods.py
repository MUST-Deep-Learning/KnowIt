from __future__ import annotations  # noqa: D100
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com"
__description__ = "Helper functions used in KnowIt's trainer module."


from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from torch import Tensor

import os

import torch
from torch import optim
from torch.nn import functional as f
from torch.optim import lr_scheduler
import torchmetrics as tm
import pandas as pd

from helpers.logger import get_logger
logger = get_logger()

ARGMAX_METRICS = {"Accuracy", "Precision", "Recall", "F1Score", "AUROC", "accuracy", "f1_score", "precision", "recall"}
FLATTEN_METRICS = {"R2Score": (0, 1),}


def get_optim(
    optimizer: str,
) -> type[tuple[Callable[[], Iterator[bool]], float, None | dict[str, Any]]]:
    """Return user's choice of optimizer.

    A helper method to retrieve the user's choice of optimizer.

    Args:
    ----
        optimizer (str):    The optimizer as specified in torch.optim.

    Returns
    -------
        (type):             An uninitialized Pytorch optimizer. Takes as input
                            the model class' parameters method, the learning
                            rate, and an optional dictionary of kwargs.

    """
    return getattr(optim, optimizer)


def get_lr_scheduler(
    scheduler: str,
) -> type[tuple[type, float, None | dict[str, Any]]]:
    """Return user's choice of learning rate scheduler.

    A helper method to retrieve the user's choice of learning rate scheduler.

    Args:
    ----
        scheduler (str):    The scheduler as specified in torch.optim.

    Returns
    -------
        (type):             An uninitialized Pytorch learning rate scheduler.
                            Takes as input a Pytorch optimizer object and an
                            optional dictionary of kwargs.

    """
    return getattr(lr_scheduler, scheduler)


def prepare_functions(user_args: str | dict[str, Any], is_loss: bool) -> dict:
    """Set up and return a user's choice of functions along with configuration requirements.

    Unpacks user_args and fetches the correct functions with any kwargs.
    This is only performed once during the training run.
    Loss functions are found in `torch.nn.functional`, otherwise found in `torchmetrics`.

    Parameters
    ----------
    user_args : str | dict[str, Any]
        User-specified arguments for the function, either as a string or a dictionary.

    Returns
    -------
    dict
        functions (dict): A dictionary where each key is a metric name and each value is a tuple
                          containing a prepared function suitable for a task in the trainer module
                          and required configurations for the metric.

    """
    functions = {}
    if isinstance(user_args, dict):
        for _metric in user_args:
            kwargs = user_args[_metric]
            if is_loss:
                loss_f_cls = getattr(f, _metric)
                loss_f = loss_f_cls(**kwargs)
                # loss_f = partial(getattr(f, _metric), **kwargs)
            else:
                loss_f_cls = getattr(tm, _metric)
                loss_f = loss_f_cls(**kwargs)
                # loss_f = partial(getattr(tm, _metric), **kwargs)

            loss_f.requires_argmax = requires_argmax(_metric)
            loss_f.requires_flatten = requires_flatten(_metric)
            functions[_metric] = loss_f

    else:
        if is_loss:
            loss_f = getattr(f, user_args)
        else:
            loss_f = getattr(tm, user_args)
            loss_f = loss_f(**{})

        loss_f.requires_argmax = requires_argmax(user_args)
        loss_f.requires_flatten = requires_flatten(user_args)
        functions[user_args] = loss_f

    return functions


def requires_argmax(metric: str) -> bool:
    """ Returns a bool indicating whether the defined metric requires
    that the predictions and targets be argmaxed.

    See the following link for details on additional metrics that might need to be added
    to this list in the future:
    https://lightning.ai/docs/torchmetrics/stable/
    """

    if metric in ARGMAX_METRICS:
        return True
    else:
        return False

def requires_flatten(metric: str) -> tuple:
    """ Returns a tuple indicating whether the defined metric requires
    that the predictions and targets be flattened along with from-to-what dimensions to flatten.

    See the following link for details on additional metrics that might need to be added
    to this list in the future:
    https://lightning.ai/docs/torchmetrics/stable/
    """

    if metric in FLATTEN_METRICS:
        return True, FLATTEN_METRICS[metric]
    else:
        return False, None

def get_model_score(model_dir: str) -> tuple:
    """
    This function scans the specified directory for a checkpoint file (indicated by "ckpt" in the filename),
    loads the checkpoint, and extracts the best score, the corresponding monitored metric, and epoch number,
    from the callbacks section. If there is no best score, it is assumed that the model at the last epoch was
    stored and the score is retrieved from the `/lightning_logs/version_0/metrics.csv` file.

    Parameters
    ----------
        model_dir (str): Path to the directory containing the model's checkpoint file.

    Returns
    -------
        tuple: A tuple containing:
            - best_model_score (float): The best score achieved by the model.
            - monitor (str): The name of the metric being monitored (e.g., 'val_loss').
            - epoch (int): The epoch number at which the best score was achieved.

    Notes:
        - Assumes only one checkpoint file exists in the directory.
        - Requires PyTorch for loading the checkpoint file.
    """
    ckpt_name = [f for f in os.listdir(model_dir) if "ckpt" in f]
    ckpt_name = next(iter(ckpt_name))
    ckpt_path = os.path.join(model_dir, ckpt_name)
    ckpt = torch.load(f=ckpt_path)
    keys = list(ckpt["callbacks"].keys())

    callback_key = None
    for key in keys:
        if key.startswith("ModelCheckpoint"):
            callback_key = key
    if callback_key is None:
        logger.error("No ModelCheckpoint callback key found in the checkpoint file. Cannot get model score. Aborting.")
        exit(101)

    best_score = ckpt["callbacks"][callback_key]["best_model_score"]
    metric = ckpt["callbacks"][callback_key]["monitor"]
    epoch = ckpt["epoch"]
    if best_score is not None:
        best_score = best_score.item()
    else:
        lc_info = os.path.join(model_dir, "lightning_logs", "version_0", "metrics.csv")
        metrics_df = pd.read_csv(lc_info)
        valid_loss_df = metrics_df[metric]
        best_score = valid_loss_df.loc[valid_loss_df.last_valid_index()]

    return best_score, metric, epoch
