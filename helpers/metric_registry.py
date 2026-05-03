"""
---------------
metric_registry
---------------

Central registry mapping every supported metric/loss name to a
:class:`MetricConfig`.  This is the **only** file that needs to be
edited when adding support for a new metric or loss function.

Adding a new entry
------------------
1.  Choose a canonical ``name`` — this is what the user passes in their
    config (e.g. ``"sigmoid_focal_loss"``).
2.  Set ``source`` to ``"functional"``, ``"torchmetrics"``, or
    ``"torchvision"``.  (``"auto"`` is reserved for
    :data:`DEFAULT_METRIC_CONFIG` and should not be used in explicit
    entries.)
3.  Fill ``task_affinity`` with the set of task names for which this
    metric is valid.  Use ``{"regression", "vl_regression",
    "classification"}`` to allow all tasks.
4.  List ``pre_transforms`` in the order they should be applied.
    Each string must match a key in
    :data:`metric_adapter.TRANSFORM_REGISTRY`.

Available pre-transforms (see ``metric_adapter.py`` for implementations):
    ``"argmax"``              — argmax over dim=1 on both tensors; converts
                                one-hot/logits to class indices.
    ``"argmax_targets"``      — argmax over dim=1 on targets only; use when
                                preds must keep their full distribution (e.g.
                                AUROC).
    ``"to_float"``            — cast both tensors to ``torch.float32``.
    ``"to_long"``             — cast targets to ``torch.int64`` (preds
                                unchanged).
    ``"flatten_01"``          — flatten dims 0–1 (batch × time → samples).
    ``"softmax_preds"``       — apply softmax over dim=1 to predictions only.
    ``"log_softmax_preds"``   — apply log-softmax over dim=1 to predictions
                                only.
    ``"squeeze_last"``        — remove trailing size-1 dimensions.

Notes
-----
*  Always register both the snake_case functional name **and** the
   PascalCase OO name as separate explicit entries when both forms are
   expected to be used by the user.  The case-insensitive fallback in
   :func:`metric_adapter.build_adapter` cannot bridge the underscore
   gap between e.g. ``"f1score"`` and ``"f1_score"``.

*  ``task_affinity`` is validated at build time.  A mismatch causes a
   hard error (``logger.error`` + ``exit(101)``).

*  ``custom_fn`` is an optional escape hatch for callables that cannot
   be resolved from the standard library namespaces.  When provided,
   ``source`` is ignored during function lookup.

*  If a name is not found in the registry, ``metric_adapter`` falls back to
   ``DEFAULT_METRIC_CONFIG``: source ``"auto"`` (tries functional →
   torchmetrics → torchvision), all tasks allowed, no pre-transforms.  A
   warning is emitted.  Register the metric explicitly to set transforms and
   restrict task affinity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricConfig:
    """Static descriptor for a single metric or loss function.

    Parameters
    ----------
    source : str
        Where the callable lives.  One of ``"functional"``
        (``torch.nn.functional``), ``"torchmetrics"``,
        ``"torchvision"`` (``torchvision.ops``), or ``"auto"``
        (reserved for :data:`DEFAULT_METRIC_CONFIG` — tries all three
        namespaces in order).
    task_affinity : frozenset[str]
        The set of task names for which this metric is valid.
    pre_transforms : tuple[str, ...]
        Ordered sequence of transform keys applied to ``(preds, targets)``
        before the metric is called.  Each key must exist in
        :data:`metric_adapter.TRANSFORM_REGISTRY`.
    custom_fn : callable | None
        Optional escape hatch.  When set, this callable is used directly
        instead of resolving via ``source``.  Useful for lambdas or
        functions from libraries outside the three supported namespaces.
    """

    source: str
    task_affinity: frozenset
    pre_transforms: tuple = field(default_factory=tuple)
    custom_fn: object = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Convenience aliases so registry entries stay concise
# ---------------------------------------------------------------------------
_ALL_TASKS  = frozenset({"regression", "vl_regression", "classification"})
_REG_TASKS  = frozenset({"regression", "vl_regression"})
_CLF_TASKS  = frozenset({"classification"})

# ---------------------------------------------------------------------------
# Default config — used when a metric name is not explicitly registered
# ---------------------------------------------------------------------------
DEFAULT_METRIC_CONFIG = MetricConfig(
    source="auto",
    task_affinity=_ALL_TASKS,
    pre_transforms=(),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
METRIC_REGISTRY: dict[str, MetricConfig] = {

    # ------------------------------------------------------------------
    # torch.nn.functional  —  regression losses
    # ------------------------------------------------------------------
    "mse_loss": MetricConfig(
        source="functional",
        task_affinity=_REG_TASKS,
    ),
    "l1_loss": MetricConfig(
        source="functional",
        task_affinity=_REG_TASKS,
    ),
    "huber_loss": MetricConfig(
        source="functional",
        task_affinity=_REG_TASKS,
    ),
    "smooth_l1_loss": MetricConfig(
        source="functional",
        task_affinity=_REG_TASKS,
    ),

    # ------------------------------------------------------------------
    # torch.nn.functional  —  classification losses
    # ------------------------------------------------------------------
    "cross_entropy": MetricConfig(
        source="functional",
        task_affinity=_CLF_TASKS,
        pre_transforms=("to_float",),
    ),
    "nll_loss": MetricConfig(
        source="functional",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax_targets",),
        # Expects log-probabilities as predictions — use with architectures whose
        # final layer outputs log_softmax. For raw logits, use cross_entropy.
    ),

    # ------------------------------------------------------------------
    # torchvision.ops  —  classification losses
    # ------------------------------------------------------------------
    "sigmoid_focal_loss": MetricConfig(
        source="torchvision",
        task_affinity=_CLF_TASKS,
        pre_transforms=("to_float", "squeeze_last"),
        # Best suited for binary classification with severe class imbalance.
        # Applies sigmoid independently per class (not softmax), so it does not
        # model mutual exclusivity between classes. For general multi-class
        # problems, use cross_entropy instead.
    ),

    # ------------------------------------------------------------------
    # torchmetrics  —  regression metrics
    # ------------------------------------------------------------------
    "R2Score": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
        pre_transforms=("flatten_01",),
        # flatten_01 collapses (batch, time, components) → (batch*time, components)
        # so each time step is treated as an independent sample. Pass
        # num_outputs=<components> as a kwarg when components > 1.
    ),
    "r2_score": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
        pre_transforms=("flatten_01",),
        # flatten_01 collapses (batch, time, components) → (batch*time, components)
        # so each time step is treated as an independent sample. Pass
        # num_outputs=<components> as a kwarg when components > 1.
    ),
    "MeanAbsoluteError": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),
    "mean_absolute_error": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),
    "MeanSquaredError": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),
    "mean_squared_error": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),
    "MeanAbsolutePercentageError": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),
    "mean_absolute_percentage_error": MetricConfig(
        source="torchmetrics",
        task_affinity=_REG_TASKS,
    ),

    # ------------------------------------------------------------------
    # torchmetrics  —  classification metrics
    # All entries below require at minimum task="multiclass" and
    # num_classes=<N> to be passed as kwargs in the user config
    # (torchmetrics >= 0.10 enforces this at construction time).
    # ------------------------------------------------------------------
    "Accuracy": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
    ),
    "accuracy": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
    ),
    "F1Score": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "f1_score": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "Precision": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "precision": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "Recall": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "recall": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
        # Also requires average=<"macro"|"micro"|"weighted"|"none"> as a kwarg.
    ),
    "AUROC": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("softmax_preds", "argmax_targets"),
    ),
    "auroc": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("softmax_preds", "argmax_targets"),
    ),
    "CohenKappa": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
    ),
    "cohen_kappa": MetricConfig(
        source="torchmetrics",
        task_affinity=_CLF_TASKS,
        pre_transforms=("argmax",),
    ),
}
