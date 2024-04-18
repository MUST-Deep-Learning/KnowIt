from __future__ import annotations  # noqa: INP001, D100

__author__ = "randlerabe@gmail.com"
__description__ = "Helper functions used in KnowIt's trainer module."


from typing import Any, Callable, Iterator

from torch import optim
from torch.nn import functional as f
from torch.optim import lr_scheduler
from torchmetrics import functional as mf


def get_loss_function(
    loss: str,
) -> Callable[
    [int | float, int | float, None | dict[str, Any]],
    int | float,
]:
    """Return user's choice of loss function.

    A helper method to retrieve the user's choice of loss function. The loss
    function name must be the functional version from Pytorch's torch.nn
    module.

    Args:
    ----
        loss (str):         The loss function as specified in
                            torch.nn.functional.

    Returns:
    -------
        (Callable):         A Pytorch loss function. Takes as input either
                            two integers/floats and an optional dictionary of
                            kwargs. Returns an integer/float value.

    """
    return getattr(f, loss)


def get_performance_metric(
    metric: str,
) -> Callable[
    [int | float, int | float, None | dict[str, Any]],
    int | float,
]:
    """Return user's choice of performance metrics.

    A helper method to retrieve the user's choice of performance metrics. The
    metrics is the functional version from torchmetrics.

    Args:
    ----
        metric (str):       The metric as specified in Torchmetrics. The metric
                            name must be the functional version.

    Returns:
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

    Returns:
    -------
        (type):             An uninitiliazed Pytorch optimizer. Takes as input
                            the model parameters method, the learning rate,
                            and an optional dictionary of kwargs.

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

    Returns:
    -------
        (type):             An uninitiliazed Pytorch learning rate scheduler.
                            Takes as input a Pytorch optimizer object and an
                            optional dictionary of kwargs.

    """
    return getattr(lr_scheduler, scheduler)
