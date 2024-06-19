from __future__ import annotations  # noqa: INP001, D100

__author__ = "randlerabe@gmail.com"
__description__ = "Helper functions used in KnowIt's trainer module."


from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from torch import Tensor

from torch import optim
from torch.nn import functional as f
from torch.optim import lr_scheduler
from torchmetrics import functional as mf


def get_loss_function(loss: str) -> Callable[..., float | Tensor]:
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


def get_performance_metric(metric: str) -> Callable[..., float | Tensor]:
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

    Returns:
    -------
        (type):             An uninitialized Pytorch learning rate scheduler.
                            Takes as input a Pytorch optimizer object and an
                            optional dictionary of kwargs.

    """
    return getattr(lr_scheduler, scheduler)


def prepare_function(user_args: str | dict[str, Any], *, is_loss: bool) -> (
        dict[str, Callable[..., float | Tensor]]
    ):
        """Set up and return a user's choice of function.

        Unpacks user_args and fetches the correct functions with any kwargs.
        This is only performed once during the training run.

        Returns
        -------
            functions (dict):   A prepared function suitable for a task in the
                                trainer module.

        """
        function: dict[str, Callable[..., float|Tensor]] = {}
        if is_loss and isinstance(user_args, dict):
            for _metric in user_args:
                kwargs = user_args[_metric]
                loss_f = partial(
                    get_loss_function(_metric),
                    **kwargs,
                )
                function[_metric] = loss_f
        elif is_loss and not isinstance(user_args, dict):
                loss_f = get_loss_function(user_args)
                function[user_args] = loss_f
        elif not is_loss and isinstance(user_args, dict):
            for _metric in user_args:
                kwargs = user_args[_metric]
                perf_f = partial(
                    get_performance_metric(_metric),
                    **kwargs,
                )
                function[_metric] = perf_f
        elif not is_loss and not isinstance(user_args, dict):
                perf_f = get_performance_metric(user_args)
                function[user_args] = perf_f

        return function
