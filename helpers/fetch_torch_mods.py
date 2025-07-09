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
from torchmetrics import functional as mf
import pandas as pd


def get_loss_function(loss: str) -> Callable[..., float | Tensor]:
    """Return user's choice of loss function.

    A helper method to retrieve the user's choice of loss function. The loss
    function name must be the functional version from Pytorch's torch.nn
    module.

    Args:
    ----
        loss (str):         The loss function as specified in
                            torch.nn.functional.

    Returns
    -------
        (Callable):         A Pytorch loss function. Takes as input either
                            two integers/floats and an optional dictionary of
                            kwargs. Returns an integer/float value.

    """
    return getattr(f, loss)


def get_performance_metric(metric: str) -> Callable[..., float | Tensor]:
    """Return user's choice of performance metrics.

    A helper method to retrieve the user's choice of performance metrics. The
    metrics is the functional version from torchmetrics.

    Args:
    ----
        metric (str):       The metric as specified in Torchmetrics. The metric
                            name must be the functional version.

    Returns
    -------
        (Callable):         A Torchmetrics metric function. Takes as input
                            either two integers/floats and an optional
                            dictionary of kwargs. Returns an integer/float
                            value.

    """
    return getattr(mf, metric)


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


def prepare_function(user_args: str | dict[str, Any], *, is_loss: bool) -> (
        dict[str, tuple[Callable[..., float | Tensor], bool]]
    ):
        """Set up and return a user's choice of function along with OHE requirement.

        Unpacks user_args and fetches the correct functions with any kwargs.
        This is only performed once during the training run.

        Parameters
        ----------
        user_args : str | dict[str, Any]
            User-specified arguments for the function, either as a string or a dictionary.
        is_loss : bool
            Flag indicating whether to prepare a loss function or a performance metric.

        Returns
        -------
        dict
            functions (dict): A dictionary where each key is a metric name and each value is a tuple
                              containing a prepared function suitable for a task in the trainer module
                              and a boolean indicating whether one-hot encoding is required for the function.

        """
        function: dict[str, Callable[..., float|Tensor]] = {}
        if is_loss and isinstance(user_args, dict):
            for _metric in user_args:
                kwargs = user_args[_metric]
                loss_f = partial(
                    get_loss_function(_metric),
                    **kwargs,
                )
                function[_metric] = (loss_f, requires_ohe(_metric))
        elif is_loss and not isinstance(user_args, dict):
                loss_f = get_loss_function(user_args)
                function[user_args] = (loss_f, requires_ohe(user_args))
        elif not is_loss and isinstance(user_args, dict):
            for _metric in user_args:
                kwargs = user_args[_metric]
                perf_f = partial(
                    get_performance_metric(_metric),
                    **kwargs,
                )
                function[_metric] = (perf_f, requires_ohe(_metric))
        elif not is_loss and not isinstance(user_args, dict):
                perf_f = get_performance_metric(user_args)
                function[user_args] = (perf_f, requires_ohe(user_args))

        return function

def requires_ohe(metric: str) -> bool:
    """ Returns a bool indicating whether the defined metric requires
    that the targets be one hot encoded.

    See the following link for details on additional metrics that might need to be added
    to this list in the future:
    https://lightning.ai/docs/torchmetrics/stable/
    """

    if metric in ("accuracy", ):
        return True
    else:
        return False

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
    keys = list(ckpt["callbacks"].keys())[0]

    best_score = ckpt["callbacks"][keys]["best_model_score"]
    metric = ckpt["callbacks"][keys]["monitor"]
    epoch = ckpt["epoch"]
    if best_score is not None:
        best_score = best_score.item()
    else:
        lc_info = os.path.join(model_dir, "lightning_logs", "version_0", "metrics.csv")
        metrics_df = pd.read_csv(lc_info)
        valid_loss_df = metrics_df[metric]
        best_score = valid_loss_df.loc[valid_loss_df.last_valid_index()]

    return best_score, metric, epoch
