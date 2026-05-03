"""
-------
PLModel
-------

The "PLModel" class is required in order to be able to use Pytorch Lightning's
Trainer. It initializes a Pytorch model and defines the train, validation, and
test steps as well as the optimizers and any learning rate schedulers.

For more information, see Pytorch Lightning's documentation here:
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

"""  # noqa: D400, D205

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com, moutoncoenraad@gmail.com"
__description__ = "Constructs a Pytorch Lightning model class."

# standard library imports
import copy
from typing import TYPE_CHECKING, Any

# external imports
from pytorch_lightning import LightningModule
from torch import nn

# internal imports
from helpers.fetch_torch_mods import get_lr_scheduler, get_optim
from helpers.metric_adapter import MetricAdapter, build_adapters
from helpers.logger import get_logger

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = get_logger()


class PLModel(LightningModule):

    def __init__(
        self,
        loss: str | dict[str, Any],
        learning_rate: float,
        optimizer: str | dict[str, Any],
        learning_rate_scheduler: None | str | dict[str, Any],
        performance_metrics: None | str | dict[str, Any],
        model: Module,
        model_params: dict[str, Any],
        output_scaler: None | object = None,
    ) -> None:
        """Initialize the PLModel.

        Parameters
        ----------
        loss : str | dict[str, Any]
            Loss function name or a dictionary mapping a loss function name to
            its kwargs.  Must be registered in ``metric_registry.py``.
        learning_rate : float
            Learning rate passed to the optimizer.
        optimizer : str | dict[str, Any]
            Optimizer name or a dictionary mapping an optimizer name to its
            kwargs, as specified in ``torch.optim``.
        learning_rate_scheduler : None | str | dict[str, Any]
            Learning rate scheduler name or a dictionary mapping a scheduler
            name to its kwargs, as specified in
            ``torch.optim.lr_scheduler``. Pass ``None`` to disable.
        performance_metrics : None | str | dict[str, Any]
            Performance metric name or a dictionary mapping metric names to
            their kwargs.  Must be registered in ``metric_registry.py``.
            Pass ``None`` to disable performance metric tracking.
        model : Module
            Uninitialised PyTorch model class.
        model_params : dict[str, Any]
            Keyword arguments forwarded to ``model`` at construction time.
            Must include a ``"task_name"`` key with one of
            ``"regression"``, ``"vl_regression"``, or ``"classification"``.
        output_scaler : None | object, optional
            A fitted scaler with an ``inverse_transform`` method.
            When provided, predictions and targets are
            inverse-transformed before loss and metric computation.
            Default is ``None``.
        """
        super().__init__()

        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics
        self.output_scaler = output_scaler
        self.task_name = model_params['task_name']

        self.model = model(**model_params)
        self.loss_functions: dict[str, MetricAdapter] = build_adapters(
            user_args=loss,
            is_loss=True,
            task_name=self.task_name,
        )

        if self.performance_metrics is not None:
            self.metrics = self._make_metrics()

        self.save_hyperparameters(ignore=['model'])

    # ----------------------------
    # Initial construction methods
    # ----------------------------

    def _make_metrics(self) -> nn.ModuleDict:
        """Build per-split metric module dicts from the user's metric config.

        Calls :func:`build_adapters` once to obtain :class:`MetricAdapter`
        instances, then deep-copies them for each data split to ensure
        per-split state isolation.

        Only stateful torchmetrics metrics need to live inside the
        ``ModuleDict`` (so Lightning tracks their parameters/buffers).
        Functional metrics are stored as plain Python objects.

        Returns
        -------
        nn.ModuleDict
            A nested ``ModuleDict`` keyed first by split name and then by
            metric name, each holding an independent adapter per split.
        """
        base_adapters = build_adapters(
            user_args=self.performance_metrics,
            is_loss=False,
            task_name=self.task_name,
        )

        # Only torchmetrics Metric objects need to be tracked by Lightning.
        # We keep adapters as plain attributes keyed by split; the stateful
        # .fn objects are registered inside a ModuleDict so that Lightning
        # can move them to the correct device.
        splits = ("trn", "val", "result_train", "result_valid", "result_eval")
        metric_modules = nn.ModuleDict({
            split: nn.ModuleDict({
                name: copy.deepcopy(adapter.fn)
                for name, adapter in base_adapters.items()
                if adapter.is_stateful
            })
            for split in splits
        })

        # Store the full adapters (including transform pipelines) per split.
        # The .fn inside each adapter is replaced by the tracked module copy.
        self._metric_adapters: dict[str, dict[str, MetricAdapter]] = {}
        for split in splits:
            self._metric_adapters[split] = {}
            for name, adapter in base_adapters.items():
                split_adapter = copy.deepcopy(adapter)
                if adapter.is_stateful:
                    # Point to the Lightning-tracked copy so device moves work.
                    split_adapter.fn = metric_modules[split][name]
                self._metric_adapters[split][name] = split_adapter

        return metric_modules

    def configure_optimizers(self) -> dict[str, Any]:
        """Return configured optimizer and optional learning rate scheduler.

        Overrides :meth:`pytorch_lightning.LightningModule.configure_optimizers`.

        Returns
        -------
        dict[str, Any]
            A dictionary with an ``"optimizer"`` key and, if a learning rate
            scheduler is configured, an ``"lr_scheduler"`` key.
        """
        if not self.optimizer:
            logger.error("No optimizer specified. Cannot configure optimizers.")
            exit(101)

        if isinstance(self.optimizer, dict):
            for optim_name, opt_kwargs in self.optimizer.items():
                optimizer = get_optim(optim_name)(
                    params=self.model.parameters(),
                    lr=self.lr,
                    **opt_kwargs,
                )
        else:
            optimizer = get_optim(self.optimizer)(
                params=self.model.parameters(),
                lr=self.lr,
            )

        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, dict):
                lr_dict = {}
                for sched in self.lr_scheduler:
                    sched_kwargs = self.lr_scheduler[sched]
                    if "monitor" in sched_kwargs:
                        monitor = sched_kwargs.pop("monitor")
                        lr_dict["monitor"] = monitor
                    scheduler = get_lr_scheduler(sched)(
                        optimizer=optimizer,
                        **sched_kwargs,
                    )
                    lr_dict["scheduler"] = scheduler
                return {"optimizer": optimizer, "lr_scheduler": lr_dict}
            scheduler = get_lr_scheduler(self.lr_scheduler)(optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    # -----------------------
    # Main stepping functions
    # -----------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : dict[str, Any]
            A dictionary containing at least ``"x"`` (model inputs) and
            ``"y"`` (targets).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tensor
            The combined training loss for the current batch.
        """
        y_pred = self.model.forward(batch['x'])

        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label="train_loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(batch['y'], y_pred, split="trn")

        loss_log_metrics['epoch'] = float(self.current_epoch)
        self.log_dict(loss_log_metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Perform a single validation step.

        Parameters
        ----------
        batch : dict[str, Any]
            A dictionary containing at least ``"x"`` (model inputs) and
            ``"y"`` (targets).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tensor
            The combined validation loss for the current batch.
        """
        y_pred = self.model.forward(batch['x'])

        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label="valid_loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(batch['y'], y_pred, split="val")

        loss_log_metrics['epoch'] = float(self.current_epoch)
        self.log_dict(loss_log_metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> dict[str, Tensor]:
        """Perform a single test step.

        Parameters
        ----------
        batch : dict[str, Any]
            A dictionary containing at least ``"x"`` (model inputs) and
            ``"y"`` (targets).
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int
            Index of the current dataloader. ``0`` corresponds to the training
            set, ``1`` to the validation set, and ``2`` to the evaluation set.

        Returns
        -------
        dict[str, Tensor]
            A dictionary of logged loss metrics for the current batch.
        """
        loaders = {0: "result_train_", 1: "result_valid_", 2: "result_eval_"}
        splits  = {0: "result_train",  1: "result_valid",  2: "result_eval"}
        current_loader = loaders[dataloader_idx]
        current_split  = splits[dataloader_idx]

        y_pred = self.model.forward(batch['x'])

        _, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label=current_loader + "loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(batch['y'], y_pred, split=current_split)

        self.log_dict(
            loss_log_metrics,
            on_epoch=True, on_step=False, prog_bar=True,
            logger=True, add_dataloader_idx=False,
        )
        return loss_log_metrics

    # -------------------------------
    # Performance measuring functions
    # -------------------------------

    def _compute_loss(
        self,
        y: Tensor,
        y_pred: Tensor,
        loss_label: str,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the loss over all configured loss functions for a single batch.

        For each configured loss the adapter's :meth:`~MetricAdapter.transform`
        pipeline is applied to ``(y_pred, y)`` before the loss function is
        called.  If an ``output_scaler`` is set the loss is also computed on
        inverse-transformed outputs and overwrites the original entry in the
        log dict.

        Parameters
        ----------
        y : Tensor
            Ground-truth targets for the current batch.
        y_pred : Tensor
            Model predictions for the current batch.
        loss_label : str
            Prefix used when constructing log metric keys.

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            ``(combined_loss, log_metrics)`` where ``combined_loss`` is the
            scalar used for backpropagation and ``log_metrics`` maps
            ``"{loss_label}_{name}"`` to each individual loss value.
        """
        log_metrics: dict[str, Tensor] = {}
        losses = []

        for name, adapter in self.loss_functions.items():
            preds   = y_pred.clone()
            targets = y.clone()

            preds, targets = adapter.transform(preds, targets)

            loss_val = adapter(preds, targets)
            losses.append(loss_val)
            log_metrics[f"{loss_label}_{name}"] = loss_val

            if self.output_scaler is not None:
                preds_inv = self.output_scaler.inverse_transform(
                    preds.clone().detach().cpu()
                ).to(self.device)
                targets_inv = self.output_scaler.inverse_transform(
                    targets.clone().detach().cpu()
                ).to(self.device)
                log_metrics[f"{loss_label}_{name}"] = adapter(preds_inv, targets_inv)

        loss = self._loss_combiner(losses)

        if loss is None:
            logger.error("Something went wrong when trying to compute the loss.")
            raise TypeError("The variable 'loss' cannot be None.")

        return loss, log_metrics

    def _loss_combiner(self, losses: list[Tensor]) -> Tensor:
        """Combine multiple loss values into a single scalar loss.

        Currently returns the first loss only. Intended to be extended to
        support weighted combinations of multiple losses.

        Parameters
        ----------
        losses : list[Tensor]
            A list of scalar loss tensors, one per configured loss function.

        Returns
        -------
        Tensor
            The combined scalar loss.
        """
        return losses[0]

    def _update_performance(self, y: Tensor, y_pred: Tensor, split: str) -> None:
        """Accumulate per-batch predictions and targets into metric states.

        Applies inverse scaling if an ``output_scaler`` is set, then runs
        each metric adapter's transform pipeline before calling the metric's
        :meth:`~MetricAdapter.update` method.

        Parameters
        ----------
        y : Tensor
            Ground-truth targets for the current batch.
        y_pred : Tensor
            Model predictions for the current batch.
        split : str
            The data split key (e.g., ``"trn"``, ``"val"``, ``"result_eval"``).
        """
        targets     = y.detach()
        predictions = y_pred.detach()

        if self.output_scaler is not None:
            predictions = self.output_scaler.inverse_transform(
                predictions.clone().cpu()
            ).to(self.device)
            targets = self.output_scaler.inverse_transform(
                targets.clone().cpu()
            ).to(self.device)

        for name, adapter in self._metric_adapters[split].items():
            p = predictions.clone()
            t = targets.clone()

            p, t = adapter.transform(p, t)
            adapter.update(p, t)

    def _compute_and_log_performance(self, split: str, perf_label: str) -> None:
        """Compute epoch-level metrics from accumulated state and log them.

        Calls :meth:`~MetricAdapter.compute` on each metric for the given split,
        logs all metrics via :meth:`log_dict`, then resets each metric's
        internal state ready for the next epoch.

        Parameters
        ----------
        split : str
            The data split key (e.g., ``"trn"``, ``"val"``, ``"result_eval"``).
        perf_label : str
            Prefix used when constructing log metric keys.
        """
        log_metrics = {}
        for name, adapter in self._metric_adapters[split].items():
            log_metrics[perf_label + name] = adapter.compute()
            adapter.reset()
        self.log_dict(log_metrics, prog_bar=True)

    # ---------
    # Callbacks
    # ---------

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics, then advance the sampler epoch."""
        if self.performance_metrics is not None:
            self._compute_and_log_performance("trn", "train_perf_")
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
            self.trainer.train_dataloader.batch_sampler.set_epoch(
                self.current_epoch + 1
            )

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of each epoch."""
        if self.performance_metrics is not None:
            self._compute_and_log_performance("val", "valid_perf_")

    def on_train_epoch_start(self) -> None:
        """Reset model internal states for a new training epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_validation_epoch_start(self) -> None:
        """Reset model internal states for a new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics for all result splits at epoch end."""
        if self.performance_metrics is not None:
            for split, label in [
                ("result_train", "result_train_perf_"),
                ("result_valid", "result_valid_perf_"),
                ("result_eval",  "result_eval_perf_"),
            ]:
                self._compute_and_log_performance(split, label)

    def on_test_epoch_start(self) -> None:
        """Reset model internal states for a new test epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_batch_start(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Update and optionally reset model internal states at test batch start.

        Parameters
        ----------
        batch : dict[str, Any]
            The current batch dictionary.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the current dataloader. Default is ``0``.
        """
        if batch_idx == 0:
            if hasattr(self.model, 'force_reset'):
                self.model.force_reset()
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_train_batch_start(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Update model internal states at training batch start."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_validation_batch_start(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Update model internal states at validation batch start."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_predict_batch_start(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Update model internal states at predict batch start."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])


class PLModel_custom(PLModel):

    def __init__(
        self,
        loss,
        learning_rate,
        optimizer,
        learning_rate_scheduler,
        performance_metrics,
        model,
        model_params,
        output_scaler=None,
        custom_pl_model_kwargs=None,
    ) -> None:
        super().__init__(
            loss, learning_rate, optimizer, learning_rate_scheduler,
            performance_metrics, model, model_params, output_scaler,
        )
    # YOU CAN ADD YOUR OWN CUSTOM CALLBACKS OR OVERWRITE THOSE ALREADY DEFINED IN `PLModel`
    # e.g. add on_train_batch_end to add some extra terms to the loss during training

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     super().on_train_batch_start(batch, batch_idx, dataloader_idx)
