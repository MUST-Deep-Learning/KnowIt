from __future__ import annotations  # noqa: D100
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com"
__description__ = "Helper functions used when loading and configuring Pytorch or Torchmetrics modules and functions."

# standard library imports
from dataclasses import dataclass
from typing import Any
import os
import functools

# external imports
import torch
import torch.nn.functional as F
import torchmetrics
from pandas import read_csv

# internal imports
from helpers.logger import get_logger
logger = get_logger()

ARGMAX_METRICS = {"Accuracy", "Precision", "Recall", "F1Score", "AUROC", "accuracy", "f1_score", "precision", "recall"}
FLATTEN_METRICS = {"R2Score": (0, 1)}

@dataclass
class PreparedFunction:
    """Container for a prepared loss function or performance metric.

    Attributes
    ----------
    name : str
        The string name of the function or metric.
    fn : Any
        The instantiated callable (torchmetrics metric instance or
        ``torch.nn.functional`` function).
    requires_argmax : bool
        Whether predictions and targets must be argmaxed before being
        passed to ``fn``.
    requires_flatten : bool
        Whether predictions and targets must be flattened before being
        passed to ``fn``.
    flatten_dims : tuple | None
        The ``(start_dim, end_dim)`` pair to use when flattening.
        ``None`` when ``requires_flatten`` is ``False``.
    """

    name: str
    fn: Any
    requires_argmax: bool
    requires_flatten: bool
    flatten_dims: tuple | None


def get_optim(optimizer: str) -> type[torch.optim.Optimizer]:
    """
    Return user's choice of optimizer class.

    A helper method to retrieve the user's choice of optimizer from the
    torch.optim module.

    Parameters
    ----------
    optimizer : str
        The name of the optimizer as specified in `torch.optim`
        (e.g., "Adam", "SGD").

    Returns
    -------
    type[torch.optim.Optimizer]
        The uninitialized PyTorch optimizer class. This class can be
        instantiated by passing the model parameters and learning rate.
    """
    return getattr(torch.optim, optimizer)


def get_lr_scheduler(scheduler: str) -> type[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Return user's choice of learning rate scheduler class.

    A helper method to retrieve the uninitialized class of a learning rate
    scheduler from the `torch.optim.lr_scheduler` module.

    Parameters
    ----------
    scheduler : str
        The name of the scheduler as specified in `torch.optim.lr_scheduler`
        (e.g., "StepLR", "CosineAnnealingLR").

    Returns
    -------
    type[torch.optim.lr_scheduler.LRScheduler] | type[torch.optim.lr_scheduler.ReduceLROnPlateau]
        The uninitialized learning rate scheduler class. This class typically
        requires a `torch.optim.Optimizer` instance and specific hyperparameter
        kwargs for instantiation.
    """
    return getattr(torch.optim.lr_scheduler, scheduler)


def prepare_functions(user_args: str | dict, is_loss: bool) -> dict[str, PreparedFunction]:
    """Set up and return a user's choice of performance measuring functions along with configuration requirements.

    Unpacks user_args and fetches the correct functions with any kwargs.
    If ``is_loss=True`` functions are found in ``torch.nn.functional``,
    otherwise found in ``torchmetrics``.

    Parameters
    ----------
    user_args : str | dict
        User-specified arguments for the function, either as a string (no
        kwargs) or a dictionary mapping function names to their kwargs.
    is_loss : bool
        Flag indicating whether to prepare a loss function or a performance
        metric.

    Returns
    -------
    dict[str, PreparedFunction]
        A dictionary where each key is a metric/loss name and each value is
        a :class:`PreparedFunction` instance carrying the callable and its
        required configurations.
    """
    functions: dict[str, PreparedFunction] = {}

    if isinstance(user_args, dict):
        for name, kwargs in user_args.items():
            fn = _build_fn(name, kwargs, is_loss)
            functions[name] = _wrap(name, fn)

    elif isinstance(user_args, str):
        fn = _build_fn(user_args, {}, is_loss)
        functions[user_args] = _wrap(user_args, fn)

    else:
        logger.error('Function must be defined as a dict or str.')
        exit(101)

    return functions


def _build_fn(name: str, kwargs: dict, is_loss: bool) -> Any:
    """Instantiate (or retrieve) a loss function or torchmetrics metric.

    Loss functions from ``torch.nn.functional`` are plain callables. If kwargs
    are provided they are bound eagerly via :func:`functools.partial`, since
    these functions accept their arguments at call time rather than construction
    time. Torchmetrics classes are instantiated with ``**kwargs``.

    Parameters
    ----------
    name : str
        Attribute name on ``torch.nn.functional`` or ``torchmetrics``.
    kwargs : dict
        Keyword arguments bound via ``functools.partial`` for loss functions,
        or forwarded to the torchmetrics constructor for metrics.
    is_loss : bool
        When ``True`` the callable is sourced from ``torch.nn.functional``;
        otherwise it is sourced from ``torchmetrics``.

    Returns
    -------
    Any
        A ready-to-call loss function (optionally with partial kwargs) or an
        initialised torchmetrics metric.
    """
    if is_loss:
        fn = getattr(F, name)
        return functools.partial(fn, **kwargs) if kwargs else fn
    return getattr(torchmetrics, name)(**kwargs)


def _wrap(name: str, fn: Any) -> PreparedFunction:
    """Wrap a callable in a :class:`PreparedFunction` with its config flags.

    Parameters
    ----------
    name : str
        The string identifier of the metric or loss function.
    fn : Any
        The callable to wrap.

    Returns
    -------
    PreparedFunction
        The callable bundled with its argmax / flatten requirements.
    """
    _requires_flatten, _flatten_dims = requires_flatten(name)
    return PreparedFunction(
        name=name,
        fn=fn,
        requires_argmax=requires_argmax(name),
        requires_flatten=_requires_flatten,
        flatten_dims=_flatten_dims,
    )


def requires_argmax(metric: str) -> bool:
    """Return whether the metric requires argmaxed predictions and targets.

    The check is first performed with the metric name as-is against
    :data:`ARGMAX_METRICS`.  If no match is found a second case-insensitive
    check is performed by lowercasing both the query and every entry in the
    set.

    See https://lightning.ai/docs/torchmetrics/stable/
    or https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions for a full list of
    available metrics.

    Parameters
    ----------
    metric : str
        The name of the metric.

    Returns
    -------
    bool
        ``True`` if the metric requires argmax, ``False`` otherwise.
    """
    if metric in ARGMAX_METRICS:
        return True
    return metric.lower() in {m.lower() for m in ARGMAX_METRICS}


def requires_flatten(metric: str) -> tuple[bool, tuple | None]:
    """Return whether the metric requires flattened predictions and targets.

    The check is first performed with the metric name as-is against
    :data:`FLATTEN_METRICS`.  If no match is found a second case-insensitive
    check is performed by lowercasing both the query and every key in the
    dictionary.

    See https://lightning.ai/docs/torchmetrics/stable/
    or https://docs.pytorch.org/docs/stable/nn.functional.html#loss-functions for a full list of
    available metrics.

    Parameters
    ----------
    metric : str
        The name of the metric.

    Returns
    -------
    tuple[bool, tuple | None]
        A ``(requires_flatten, flatten_dims)`` pair.  ``flatten_dims`` is the
        ``(start_dim, end_dim)`` tuple when flattening is required, otherwise
        ``None``.
    """
    if metric in FLATTEN_METRICS:
        return True, FLATTEN_METRICS[metric]

    lower_map = {k.lower(): v for k, v in FLATTEN_METRICS.items()}
    if metric.lower() in lower_map:
        return True, lower_map[metric.lower()]

    return False, None


def get_model_score(model_dir: str) -> tuple:
    """
    Scan a directory for a checkpoint file, load it, and extract the best score,
    monitored metric, and corresponding epoch.

    If no best score is found in the checkpoint, the function assumes that the
    final epoch model was saved and retrieves the score from the
    ``/lightning_logs/version_0/metrics.csv`` file.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the model checkpoint file.

    Returns
    -------
    best_model_score : float
        The best score achieved by the model.
    monitor : str
        Name of the monitored metric (e.g., ``'val_loss'``).
    epoch : int
        Epoch at which the best score was achieved.

    Notes
    -----
    - Assumes only one checkpoint file exists in the directory.
    - Requires PyTorch to load the checkpoint file.
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
        metrics_df = read_csv(lc_info)
        valid_loss_df = metrics_df[metric]
        best_score = valid_loss_df.loc[valid_loss_df.last_valid_index()]

    return best_score, metric, epoch
