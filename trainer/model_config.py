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

from typing import TYPE_CHECKING, Any, Callable

from pytorch_lightning import LightningModule
from torch import argmax
from torch import nn
import torchmetrics as tm

from helpers.fetch_torch_mods import (
    get_lr_scheduler,
    get_optim,
    prepare_functions,
    requires_argmax,
    requires_flatten
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
        super().__init__()

        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics
        self.output_scaler = output_scaler

        self.model = model(**model_params)
        self.loss_functions = prepare_functions(user_args=loss, is_loss=True)

        if self.performance_metrics is not None:
            self.metrics = nn.ModuleDict({
                "trn": self._build_metric_collection(),
                "val": self._build_metric_collection(),
                "result_train": self._build_metric_collection(),
                "result_valid": self._build_metric_collection(),
                "result_eval": self._build_metric_collection(),
            })

        self.save_hyperparameters(ignore=['model'])

    # ----------------------------
    # Initial construction methods
    # ----------------------------

    def _build_metric_collection(self) -> nn.ModuleDict:
        """Build a ModuleDict of individual metrics, each stored with its config."""

        metrics = prepare_functions(self.performance_metrics, is_loss=False)
        return nn.ModuleDict(metrics)

    def configure_optimizers(self) -> dict[str, Any]:
        """Return configured optimizer and optional learning rate scheduler.

        Overrides the method in pl.LightningModule.
        """
        if self.optimizer:
            if isinstance(self.optimizer, dict):
                for optim in self.optimizer:
                    opt_kwargs = self.optimizer[optim]
                    optimizer = get_optim(optim)(
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

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        y_pred = self.model.forward(batch['x'])

        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label="train_loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(batch['y'], y_pred, split="trn", perf_label="train_perf_")

        loss_log_metrics['epoch'] = float(self.current_epoch)
        self.log_dict(loss_log_metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        y_pred = self.model.forward(batch['x'])

        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label="valid_loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(batch['y'], y_pred, split="val", perf_label="valid_perf_")

        loss_log_metrics['epoch'] = float(self.current_epoch)
        self.log_dict(loss_log_metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int):
        loaders = {0: "result_train_", 1: "result_valid_", 2: "result_eval_"}
        splits  = {0: "result_train",  1: "result_valid",  2: "result_eval"}
        current_loader = loaders[dataloader_idx]
        current_split  = splits[dataloader_idx]

        y_pred = self.model.forward(batch['x'])

        _, loss_log_metrics = self._compute_loss(
            y=batch['y'], y_pred=y_pred, loss_label=current_loader + "loss"
        )

        if self.performance_metrics is not None:
            self._update_performance(
                batch['y'], y_pred, split=current_split, perf_label=current_loader + "perf_"
            )

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

        log_metrics: dict[str, float | Tensor] = {}
        losses = []

        for _function in self.loss_functions:

            targets = y.clone()
            predictions = y_pred.clone()
            function = self.loss_functions[_function]

            if function.requires_argmax:
                targets = argmax(targets, dim=1).to(self.device)
                predictions = argmax(predictions, dim=1).to(self.device)
            if function.requires_flatten[0]:
                targets = targets.flatten(start_dim=function.requires_flatten[1][0], end_dim=function.requires_flatten[1][1]).to(self.device)
                predictions = predictions.flatten(start_dim=function.requires_flatten[1][0], end_dim=function.requires_flatten[1][1]).to(self.device)

            loss = function(input=predictions, target=targets)
            losses.append(loss)
            log_metrics[loss_label + '_' + _function] = loss

            if self.output_scaler is not None:
                predictions = self.output_scaler.inverse_transform(predictions.clone().detach().cpu()).to(self.device)
                if not function.requires_argmax:
                    targets = self.output_scaler.inverse_transform(targets.clone().detach().cpu()).to(self.device)
                loss_rescaled = function(input=predictions, target=targets)
                log_metrics[loss_label + '_' + _function] = loss_rescaled

        loss = self._loss_combiner(losses)

        if loss is None:
            logger.error(
                "Something went wrong when trying to compute the loss.",
            )
            e_msg = "The variable 'loss' cannot be None."
            raise TypeError(e_msg)

        return loss, log_metrics

    def _loss_combiner(self, losses):

        return losses[0]

    def _update_performance(self, y: Tensor, y_pred: Tensor, split: str, perf_label: str) -> None:
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
            t = targets.clone()
            p = predictions.clone()

            if metric.requires_argmax:
                t = argmax(t, dim=1).to(self.device)
                p = argmax(p, dim=1).to(self.device)
            if metric.requires_flatten[0]:
                t = t.flatten(start_dim=metric.requires_flatten[1][0], end_dim=metric.requires_flatten[1][1]).to(self.device)
                p = p.flatten(start_dim=metric.requires_flatten[1][0], end_dim=metric.requires_flatten[1][1]).to(self.device)

            metric.update(p, t)

    def _compute_and_log_performance(self, split: str, perf_label: str) -> None:
        log_metrics = {}
        for metric_name, metric in self.metrics[split].items():
            log_metrics[perf_label + metric_name] = metric.compute()
            metric.reset()
        self.log_dict(log_metrics, prog_bar=True)

    # ---------
    # Callbacks
    # ---------

    def on_train_epoch_end(self):
        """ Set the next training epoch number in the CustomSampler.

        This is done to manage potential stochasticity (which is connected to the epoch number).

        """
        if self.performance_metrics is not None:
            self._compute_and_log_performance("trn", "train_perf_")
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch+1)

    def on_validation_epoch_end(self):
        if self.performance_metrics is not None:
            self._compute_and_log_performance("val", "valid_perf_")

    def on_train_epoch_start(self):
        """ Reset the model internal states for new training epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_validation_epoch_start(self):
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_epoch_end(self):
        if self.performance_metrics is not None:
            for split, label in [
                ("result_train", "result_train_perf_"),
                ("result_valid", "result_valid_perf_"),
                ("result_eval", "result_eval_perf_"),
            ]:
                self._compute_and_log_performance(split, label)

    def on_test_epoch_start(self):
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if batch_idx == 0:
            if hasattr(self.model, 'force_reset'):
                self.model.force_reset()
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
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










