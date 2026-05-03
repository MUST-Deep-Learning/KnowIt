"""
--------------
metric_adapter
--------------

Provides :class:`MetricAdapter`, the single object that ``model_config.py``
interacts with for both loss functions and performance metrics.

Responsibilities
----------------
*  Resolve a user-supplied metric name to the correct callable from
   ``torch.nn.functional``, ``torchmetrics``, or ``torchvision.ops``.
*  Validate the metric against the active task name.
*  Build and apply an ordered pipeline of ``(preds, targets) →
   (preds, targets)`` transforms before each forward call.
*  Call the underlying function positionally (``fn(preds, targets)``)
   to avoid argument-name inconsistencies across libraries.

Public API used by ``model_config.py``
---------------------------------------
::

    adapters = build_adapters(user_args, is_loss=True,  task_name="regression")
    adapters = build_adapters(user_args, is_loss=False, task_name="classification")

    for name, adapter in adapters.items():
        preds, targets = adapter.transform(preds_raw, targets_raw)
        loss = adapter(preds, targets)           # __call__

    # For torchmetrics stateful metrics only:
    adapter.update(preds, targets)
    value = adapter.compute()
    adapter.reset()
"""

from __future__ import annotations

__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com"
__description__ = "MetricAdapter: couples predictions/targets to metric functions."

import functools
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.ops as tvops

from helpers.metric_registry import DEFAULT_METRIC_CONFIG, METRIC_REGISTRY, MetricConfig
from helpers.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Transform primitives
# Each primitive is a pure function:  (preds, targets) → (preds, targets)
# ---------------------------------------------------------------------------

def _argmax(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax over dim=1 — converts one-hot/logit tensors to class indices."""
    return torch.argmax(preds, dim=1), torch.argmax(targets, dim=1)


def _to_float(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast both tensors to float32."""
    return preds.float(), targets.float()


def _to_long(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast targets to int64 (predictions are left unchanged)."""
    return preds, targets.long()


def _flatten_01(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten dims 0–1 (batch × time-steps → samples).

    For regression outputs shaped ``(batch, time, components)`` this
    produces ``(batch*time, components)``, which is what metrics such as
    ``R2Score`` expect.
    """
    return preds.flatten(0, 1), targets.flatten(0, 1)


def _squeeze_last(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove trailing size-1 dimensions from both tensors."""
    return preds.squeeze(-1), targets.squeeze(-1)


def _softmax_preds(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply softmax over dim=1 to predictions only (targets are left unchanged)."""
    return F.softmax(preds, dim=1), targets


def _log_softmax_preds(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply log-softmax over dim=1 to predictions only (targets are left unchanged)."""
    return F.log_softmax(preds, dim=1), targets


def _argmax_targets(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax over dim=1 on targets only — converts one-hot targets to class indices."""
    return preds, torch.argmax(targets, dim=1)


# ---------------------------------------------------------------------------
# Registry of named transforms — extend this dict to add new primitives
# ---------------------------------------------------------------------------
TRANSFORM_REGISTRY: dict[str, Callable] = {
    "argmax":            _argmax,
    "to_float":          _to_float,
    "to_long":           _to_long,
    "flatten_01":        _flatten_01,
    "squeeze_last":      _squeeze_last,
    "softmax_preds":     _softmax_preds,
    "log_softmax_preds": _log_softmax_preds,
    "argmax_targets":    _argmax_targets,
}


# ---------------------------------------------------------------------------
# MetricAdapter
# ---------------------------------------------------------------------------

class MetricAdapter:
    """Couples a metric/loss callable to its required tensor transforms.

    Parameters
    ----------
    name : str
        Canonical name of the metric (used for logging and dict keys).
    fn : callable
        The resolved callable.  For torchmetrics this is a stateful
        ``Metric`` instance; for functional losses it is a plain function
        (possibly wrapped with ``functools.partial``).
    pre_transforms : list[callable]
        Ordered list of ``(preds, targets) → (preds, targets)`` functions
        applied by :meth:`transform` before the metric is called.
    is_stateful : bool
        ``True`` when ``fn`` is a torchmetrics ``Metric`` instance with
        ``update`` / ``compute`` / ``reset`` semantics.
    """

    def __init__(
        self,
        name: str,
        fn: Any,
        pre_transforms: list[Callable],
        is_stateful: bool,
    ) -> None:
        self.name = name
        self.fn = fn
        self.pre_transforms = pre_transforms
        self.is_stateful = is_stateful

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------

    def transform(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all pre-transforms in order.

        Parameters
        ----------
        preds : Tensor
            Raw model predictions.
        targets : Tensor
            Raw ground-truth targets.

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed ``(preds, targets)`` ready to be passed to the
            underlying callable.
        """
        for t in self.pre_transforms:
            preds, targets = t(preds, targets)
        return preds, targets

    # ------------------------------------------------------------------
    # Calling the underlying function
    # ------------------------------------------------------------------

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Call the underlying metric/loss function positionally.

        Passing arguments positionally avoids argument-name inconsistencies
        across ``torch.nn.functional`` (``input``/``target``),
        ``torchmetrics`` (``preds``/``target``), and ``torchvision.ops``
        (e.g. ``inputs``/``targets``).

        Parameters
        ----------
        preds : Tensor
            Already-transformed predictions (output of :meth:`transform`).
        targets : Tensor
            Already-transformed targets (output of :meth:`transform`).

        Returns
        -------
        Tensor
            Scalar loss or metric value.
        """
        return self.fn(preds, targets)

    # ------------------------------------------------------------------
    # Stateful torchmetrics interface (no-ops for functional losses)
    # ------------------------------------------------------------------

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate batch statistics (torchmetrics only).

        Parameters
        ----------
        preds : Tensor
            Already-transformed predictions.
        targets : Tensor
            Already-transformed targets.
        """
        if self.is_stateful:
            self.fn.update(preds, targets)

    def compute(self) -> torch.Tensor:
        """Return the epoch-level aggregated metric value (torchmetrics only)."""
        if self.is_stateful:
            return self.fn.compute()
        raise RuntimeError(
            f"MetricAdapter '{self.name}' wraps a functional loss and does "
            "not support compute()."
        )

    def reset(self) -> None:
        """Reset accumulated state ready for the next epoch (torchmetrics only)."""
        if self.is_stateful:
            self.fn.reset()

    def __deepcopy__(self, memo: dict) -> "MetricAdapter":
        """Deep-copy this adapter (needed for per-split metric isolation)."""
        import copy
        return MetricAdapter(
            name=self.name,
            fn=copy.deepcopy(self.fn, memo),
            pre_transforms=list(self.pre_transforms),
            is_stateful=self.is_stateful,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_fn(name: str, config: MetricConfig, kwargs: dict, is_loss: bool) -> Any:
    """Resolve the callable for ``name`` from the appropriate library namespace.

    Parameters
    ----------
    name : str
        Canonical metric name.
    config : MetricConfig
        Registry entry for this metric.
    kwargs : dict
        User-supplied keyword arguments.
    is_loss : bool
        When ``True`` the result is a plain callable (possibly
        ``functools.partial``-wrapped); when ``False`` a torchmetrics
        instance is returned.

    Returns
    -------
    Any
        Ready-to-call function or initialised torchmetrics ``Metric``.
    """
    # Custom escape hatch — caller provided an explicit callable.
    if config.custom_fn is not None:
        fn = config.custom_fn
        return functools.partial(fn, **kwargs) if kwargs else fn

    source = config.source

    if source == "functional":
        fn = getattr(F, name, None)
        if fn is None:
            logger.error(f"Loss '{name}' not found in torch.nn.functional.")
            exit(101)
        return functools.partial(fn, **kwargs) if kwargs else fn

    if source == "torchvision":
        fn = getattr(tvops, name, None)
        if fn is None:
            logger.error(f"Loss '{name}' not found in torchvision.ops.")
            exit(101)
        return functools.partial(fn, **kwargs) if kwargs else fn

    if source == "torchmetrics":
        # Try the name as-is, then fall back to PascalCase.
        cls = getattr(torchmetrics, name, None)
        if cls is None:
            pascal = _to_pascal_case(name)
            cls = getattr(torchmetrics, pascal, None)
            if cls is not None:
                logger.warning(
                    f"Metric '{name}' not found in torchmetrics; "
                    f"falling back to '{pascal}'."
                )
        if cls is None:
            logger.error(
                f"Metric '{name}' could not be resolved in torchmetrics."
            )
            exit(101)
        return cls(**kwargs)

    if source == "auto":
        fn = getattr(F, name, None)
        if fn is not None:
            return functools.partial(fn, **kwargs) if kwargs else fn
        cls = getattr(torchmetrics, name, None)
        if cls is None:
            cls = getattr(torchmetrics, _to_pascal_case(name), None)
        if cls is not None:
            return cls(**kwargs)
        fn = getattr(tvops, name, None)
        if fn is not None:
            return functools.partial(fn, **kwargs) if kwargs else fn
        logger.error(
            f"Metric '{name}' could not be resolved from torch.nn.functional, "
            "torchmetrics, or torchvision.ops."
        )
        exit(101)

    logger.error(
        f"Unknown source '{source}' for metric '{name}'. "
        "Expected 'functional', 'torchmetrics', 'torchvision', or 'auto'."
    )
    exit(101)


def _to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _resolve_config(name: str) -> MetricConfig:
    """Look up the registry entry for *name*, with a case-insensitive fallback.

    Parameters
    ----------
    name : str
        User-supplied metric name.

    Returns
    -------
    MetricConfig
        The registry entry.
    """
    if name in METRIC_REGISTRY:
        return METRIC_REGISTRY[name]

    lower_map = {k.lower(): v for k, v in METRIC_REGISTRY.items()}
    if name.lower() in lower_map:
        logger.warning(
            f"Metric '{name}' matched via case-insensitive lookup."
        )
        return lower_map[name.lower()]

    logger.warning(
        f"Metric '{name}' is not registered in metric_registry.py. "
        "Attempting auto-resolution with no pre-transforms. "
        "Register it explicitly to specify transforms and task affinity."
    )
    return DEFAULT_METRIC_CONFIG


def _validate_task(name: str, config: MetricConfig, task_name: str) -> None:
    """Hard-error if the metric is not valid for the active task.

    Parameters
    ----------
    name : str
        Metric name (for the error message).
    config : MetricConfig
        Registry entry carrying ``task_affinity``.
    task_name : str
        The task name from ``model_params['task_name']``.
    """
    if task_name not in config.task_affinity:
        logger.error(
            f"Metric '{name}' is not compatible with task '{task_name}'. "
            f"Valid tasks for this metric: {sorted(config.task_affinity)}."
        )
        exit(101)


def _build_adapter(
    name: str,
    kwargs: dict,
    is_loss: bool,
    task_name: str,
) -> MetricAdapter:
    """Build a single :class:`MetricAdapter` for the named metric/loss.

    Parameters
    ----------
    name : str
        Canonical metric name.
    kwargs : dict
        User-supplied kwargs forwarded to the callable.
    is_loss : bool
        Whether this is a loss function (``True``) or a performance metric
        (``False``).
    task_name : str
        Active task name, used for task-affinity validation.

    Returns
    -------
    MetricAdapter
        Fully configured adapter ready for use in ``model_config.py``.
    """
    config = _resolve_config(name)
    _validate_task(name, config, task_name)

    fn = _resolve_fn(name, config, kwargs, is_loss)

    pre_transforms: list[Callable] = []
    for key in config.pre_transforms:
        if key not in TRANSFORM_REGISTRY:
            logger.error(
                f"Transform '{key}' listed for metric '{name}' is not "
                "defined in TRANSFORM_REGISTRY in metric_adapter.py."
            )
            exit(101)
        pre_transforms.append(TRANSFORM_REGISTRY[key])

    is_stateful = isinstance(fn, torchmetrics.Metric)

    return MetricAdapter(
        name=name,
        fn=fn,
        pre_transforms=pre_transforms,
        is_stateful=is_stateful,
    )


# ---------------------------------------------------------------------------
# Public factory — replaces prepare_functions in fetch_torch_mods.py
# ---------------------------------------------------------------------------

def build_adapters(
    user_args: str | dict,
    is_loss: bool,
    task_name: str,
) -> dict[str, MetricAdapter]:
    """Build :class:`MetricAdapter` instances from user config.

    This function replaces ``prepare_functions`` from ``fetch_torch_mods.py``
    and is the sole entry point used by ``model_config.py``.

    Parameters
    ----------
    user_args : str | dict
        Either a plain metric name (``"mse_loss"``) or a dict mapping
        metric names to their kwargs
        (``{"R2Score": {"num_outputs": 3}}``).
    is_loss : bool
        ``True`` when preparing loss functions, ``False`` for performance
        metrics.
    task_name : str
        The value of ``model_params['task_name']``, e.g.
        ``"regression"``, ``"vl_regression"``, or ``"classification"``.

    Returns
    -------
    dict[str, MetricAdapter]
        Mapping from canonical metric name to its :class:`MetricAdapter`.

    Examples
    --------
    >>> adapters = build_adapters("mse_loss", is_loss=True, task_name="regression")
    >>> adapters = build_adapters(
    ...     {"R2Score": {"num_outputs": 3}, "MeanAbsoluteError": {}},
    ...     is_loss=False,
    ...     task_name="regression",
    ... )
    >>> adapters = build_adapters(
    ...     {"sigmoid_focal_loss": {"reduction": "mean"}},
    ...     is_loss=True,
    ...     task_name="classification",
    ... )
    """
    adapters: dict[str, MetricAdapter] = {}

    if isinstance(user_args, str):
        adapters[user_args] = _build_adapter(user_args, {}, is_loss, task_name)

    elif isinstance(user_args, dict):
        for name, kwargs in user_args.items():
            adapters[name] = _build_adapter(name, kwargs or {}, is_loss, task_name)

    else:
        logger.error(
            "Metric/loss config must be a str or dict; "
            f"got {type(user_args).__name__}."
        )
        exit(101)

    return adapters
