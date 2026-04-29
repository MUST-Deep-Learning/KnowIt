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
from torch import argmax
from torch import nn

# internal imports
from helpers.fetch_torch_mods import (
    get_lr_scheduler,
    get_optim,
    prepare_functions,
)
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
        output_scaler: None | object = None
    ) -> None:
        """Initialize the PLModel.

        Parameters
        ----------
        loss : str | dict[str, Any]
            Loss function name or a dictionary mapping a loss function name to
            its kwargs, as specified in ``torch.nn.functional``.
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
            their kwargs, as specified in ``torchmetrics``. Pass ``None`` to
            disable performance metric tracking.
        model : Module
            Uninitialised PyTorch model class.
        model_params : dict[str, Any]
            Keyword arguments forwarded to ``model`` at construction time.
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

        self.model = model(**model_params)
        self.loss_functions = prepare_functions(user_args=loss, is_loss=True)

        if self.performance_metrics is not None:
            self._metric_configs, self.metrics = self._make_metrics()

        self.save_hyperparameters(ignore=['model'])

    # ----------------------------
    # Initial construction methods
    # ----------------------------

    def _make_metrics(self) -> tuple[dict, nn.ModuleDict]:
        """Build metric configs and per-split metric module dicts.

        Calls ``prepare_functions`` once to obtain :class:`PreparedFunction`
        instances, stores their configs for later use during stepping, and
        creates independent deep-copied metric instances for each data split
        to ensure per-split state isolation.

        Returns
        -------
        tuple[dict, nn.ModuleDict]
            A tuple of:

            - **configs** (*dict*): Maps metric names to their
              :class:`PreparedFunction` instances, used to look up
              ``requires_argmax`` and ``requires_flatten`` during stepping.
            - **metrics** (*nn.ModuleDict*): A nested ``ModuleDict`` keyed
              first by split name and then by metric name, each holding an
              independent torchmetrics metric instance.
        """

        prepared = prepare_functions(self.performance_metrics, is_loss=False)
        configs = {name: pf for name, pf in prepared.items()}
        metrics = nn.ModuleDict({
            split: nn.ModuleDict({name: copy.deepcopy(pf.fn) for name, pf in prepared.items()})
            for split in ("trn", "val", "result_train", "result_valid", "result_eval")
        })
        return configs, metrics

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

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int) -> dict[str, Tensor]:
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
        splits = {0: "result_train", 1: "result_valid", 2: "result_eval"}
        current_loader = loaders[dataloader_idx]
        current_split = splits[dataloader_idx]

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

    def _compute_loss(self, y: float | Tensor, y_pred: float | Tensor, loss_label: str) -> tuple[float | Tensor, dict[str, float | Tensor]]:

        """Compute the loss over all configured loss functions for a single batch.

        Applies any required argmax or flatten transformations to the predictions
        and targets before computing each loss. If an ``output_scaler`` is set,
        the loss is also computed on inverse-transformed outputs and overwrites
        the original loss in the log. The final combined loss is produced by
        :meth:`_loss_combiner`.

        Parameters
        ----------
        y : Tensor
            Ground-truth targets for the current batch.
        y_pred : Tensor
            Model predictions for the current batch.
        loss_label : str
            Prefix used when constructing log metric keys
            (e.g., ``"train_loss"``).

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            A tuple of:

            - **loss** (*Tensor*): The combined scalar loss used for
              backpropagation.
            - **log_metrics** (*dict[str, Tensor]*): A dictionary mapping
              ``"{loss_label}_{function_name}"`` to the corresponding loss
              value for logging.
        """

        log_metrics: dict[str, float | Tensor] = {}
        losses = []

        for name, prepared in self.loss_functions.items():

            targets = y.clone()
            predictions = y_pred.clone()

            if prepared.requires_argmax:
                targets = argmax(targets, dim=1).to(self.device)
                predictions = argmax(predictions, dim=1).to(self.device)
            if prepared.requires_flatten:
                targets = targets.flatten(start_dim=prepared.flatten_dims[0], end_dim=prepared.flatten_dims[1]).to(self.device)
                predictions = predictions.flatten(start_dim=prepared.flatten_dims[0], end_dim=prepared.flatten_dims[1]).to(self.device)

            loss = prepared.fn(input=predictions, target=targets)
            losses.append(loss)
            log_metrics[loss_label + '_' + name] = loss

            if self.output_scaler is not None:
                predictions = self.output_scaler.inverse_transform(predictions.clone().detach().cpu()).to(self.device)
                if not prepared.requires_argmax:
                    targets = self.output_scaler.inverse_transform(targets.clone().detach().cpu()).to(self.device)
                loss_rescaled = prepared.fn(input=predictions, target=targets)
                log_metrics[loss_label + '_' + name] = loss_rescaled

        loss = self._loss_combiner(losses)

        if loss is None:
            logger.error(
                "Something went wrong when trying to compute the loss.",
            )
            e_msg = "The variable 'loss' cannot be None."
            raise TypeError(e_msg)

        return loss, log_metrics

    def _loss_combiner(self, losses: list[Tensor]) -> Tensor:
        """Combine multiple loss values into a single scalar loss.

        Currently returns the first loss only. This is a placeholder intended
        to be extended to support weighted combinations of multiple losses in
        a future iteration.

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

        Applies inverse scaling if an ``output_scaler`` is set, then applies
        any required argmax or flatten transformations before calling
        ``metric.update()`` on each configured metric for the given split.
        Metrics accumulate state across batches; call
        :meth:`_compute_and_log_performance` at epoch end to finalize and log.

        Parameters
        ----------
        y : Tensor
            Ground-truth targets for the current batch.
        y_pred : Tensor
            Model predictions for the current batch.
        split : str
            The data split key (e.g., ``"trn"``, ``"val"``, ``"result_eval"``).
        """

        targets = y.detach()
        predictions = y_pred.detach()

        if self.output_scaler is not None:
            predictions = self.output_scaler.inverse_transform(
                predictions.clone().cpu()
            ).to(self.device)
            targets = self.output_scaler.inverse_transform(
                targets.clone().cpu()
            ).to(self.device)

        for metric_name, metric in self.metrics[split].items():
            config = self._metric_configs[metric_name]
            t = targets.clone()
            p = predictions.clone()

            if config.requires_argmax:
                t = argmax(t, dim=1).to(self.device)
                p = argmax(p, dim=1).to(self.device)
            if config.requires_flatten:
                t = t.flatten(start_dim=config.flatten_dims[0], end_dim=config.flatten_dims[1]).to(self.device)
                p = p.flatten(start_dim=config.flatten_dims[0], end_dim=config.flatten_dims[1]).to(self.device)

            metric.update(p, t)

    def _compute_and_log_performance(self, split: str, perf_label: str) -> None:
        """Compute epoch-level metrics from accumulated state and log them.

        Calls ``metric.compute()`` on each metric for the given split to
        obtain the epoch-level aggregated value, logs all metrics via
        :meth:`log_dict`, then resets each metric's internal state ready
        for the next epoch.

        Parameters
        ----------
        split : str
            The data split key (e.g., ``"trn"``, ``"val"``, ``"result_eval"``).
        perf_label : str
            Prefix used when constructing log metric keys
            (e.g., ``"train_perf_"``).
        """

        log_metrics = {}
        for metric_name, metric in self.metrics[split].items():
            log_metrics[perf_label + metric_name] = metric.compute()
            metric.reset()
        self.log_dict(log_metrics, prog_bar=True)

    # ---------
    # Callbacks
    # ---------

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics, then advance the sampler epoch.

        Finalizes accumulated metric state for the ``"trn"`` split and logs
        epoch-level results. Also advances the epoch counter in the
        ``CustomSampler`` if present, to manage any epoch-linked stochasticity.
        """
        if self.performance_metrics is not None:
            self._compute_and_log_performance("trn", "train_perf_")
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch+1)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of each epoch."""
        if self.performance_metrics is not None:
            self._compute_and_log_performance("val", "valid_perf_")

    def on_train_epoch_start(self) -> None:
        """ Reset the model internal states for new training epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_validation_epoch_start(self) -> None:
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics for all result splits at epoch end."""
        if self.performance_metrics is not None:
            for split, label in [
                ("result_train", "result_train_perf_"),
                ("result_valid", "result_valid_perf_"),
                ("result_eval", "result_eval_perf_"),
            ]:
                self._compute_and_log_performance(split, label)

    def on_test_epoch_start(self) -> None:
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        """Update and optionally reset model internal states at the start of each test batch.

        Resets internal states on the first batch of each dataloader. Updates
        recurrent or stateful model internals and hard-sets states to the final
        prediction point in the batch if applicable.

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

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
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
        super().__init__(loss, learning_rate, optimizer, learning_rate_scheduler,
                         performance_metrics, model, model_params, output_scaler)
    # YOU CAN ADD YOUR OWN CUSTOM CALLBACKS OR OVERWRITE THOSE ALREADY DEFINED IN `PLModel`
    # e.g. add on_train_batch_end to add some extra terms to the loss during training

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     super().on_train_batch_start(batch, batch_idx, dataloader_idx)










