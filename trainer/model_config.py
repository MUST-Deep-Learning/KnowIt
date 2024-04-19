"""
-------
PLModel
-------

The "PLModel" class is required in order to be able to use Pytorch Lightning's
Trainer. It initializes a Pytorch model and defines the train, validation, and
test loops as well as the optimizers and any learning rate schedulers.

For more information, see Pytorch Lightning's documentation here:
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

"""  # noqa: INP001, D205, D212, D400, D415

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Constructs a Pytorch Lightning model class."

from typing import Any, Callable

import pytorch_lightning as pl
from torch import Tensor

from helpers.fetch_torch_mods import (
    get_lr_scheduler,
    get_optim,
    prepare_function,
)
from helpers.logger import get_logger

logger = get_logger()


class PLModel(pl.LightningModule):
    """Wrapper class to prepare model for Pytorch Lightning's Trainer.

    The class initializes a Pytorch model and defines the training,
    validation, and test steps over a batch. The optimizer configuration is
    also set inside this class. This is required for Pytorch Lightning's
    Trainer.

    Args:
    ----
        pl.LightningModule (type):      A Pytorch Lightning module.

    """

    def __init__(
        self,
        loss: str | dict[str, Any],
        learning_rate: float,
        optimizer: str | dict[str, Any],
        learning_rate_scheduler: str | dict[str, Any],
        performance_metrics: None | str | dict[str, Any],
        model: type,
        model_params: dict[str, Any],
    ) -> None:
        """PLModel constructor.

        Args:
        ----
            loss (str, dict):           Loss function as given in
                                        torch.nn.functional

            learning_rate (float):      The learning rate to be used during
                                        training.

            optimizer (str, dict):      The optimizer to be used for training
                                        as given in torch.optim. Additional
                                        kwargs can be provided as a dictionary
                                        with the pairs key (str corresponding
                                        to name) and a corresponding value.

            learning_rate_scheduler (dict):
                                        The choice of learning rate scheduler
                                        as given in torch.optim.lr_scheduler.
                                        Additional kwargs can be provided as a
                                        dictionary with the pairs key (str
                                        corresponding to name) and a
                                        corresponding value.

            performance_metrics (dict): The choice of performance metrics as
                                        given in torchmetrics.functional.
                                        Additional kwargs can be provided as a
                                        dictionary with the pairs key (str
                                        corresponding to name) and a
                                        corresponding value.

            model (type):               An unitialized Pytorch model class
                                        defined in ~./archs.

            model_params (dict):        The parameters needed to instantiate
                                        the above Pytorch model class.

        """
        super().__init__()

        self.loss = loss
        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics

        self.model = self._build_model(model, model_params)

        self.save_hyperparameters()

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        x = batch["x"]
        y = batch["y"]

        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(x)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self._compute_loss(
            y=y,
            y_pred=y_pred,
            loss_label="train_loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=y,
                y_pred=y_pred,
                perf_label="train_perf_",
            )

        log_metrics = {
            **loss_log_metrics,
            **perf_log_metrics,
        }
        # The loss and performance is accumulated over an epoch and then
        # averaged.
        self.log_dict(  # type: ignore[type]
            dictionary=log_metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        x = batch["x"]
        y = batch["y"]

        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(x)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self._compute_loss(
            y=y,
            y_pred=y_pred,
            loss_label="valid_loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=y,
                y_pred=y_pred,
                perf_label="valid_perf_",
            )

        log_metrics = {
            **loss_log_metrics,
            **perf_log_metrics,
        }
        # The loss and performance is accumulated over an epoch and then
        # averaged.
        self.log_dict(  # type: ignore[type]
            dictionary=log_metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def test_step(  # type: ignore[return-value]  # noqa: ANN201
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,
    ):
        """Compute loss and optional metrics, log metrics and return the log.

        Overrides the method in pl.LightningModule.
        """
        loaders = {
            0: "result_train_",
            1: "result_valid_",
            2: "result_eval_",
        }
        current_loader = loaders[dataloader_idx]

        x = batch["x"]
        y = batch["y"]

        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(x)

        # compute loss; depends on whether user gave kwargs
        _, loss_log_metrics = self._compute_loss(
            y=y,
            y_pred=y_pred,
            loss_label=current_loader + "loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=y,
                y_pred=y_pred,
                perf_label=current_loader + "perf_",
            )

        log_metrics = {
            **loss_log_metrics,
            **perf_log_metrics,
        }
        # The loss and performance is accumulated over an epoch and then
        # averaged.
        self.log_dict(  # type: ignore[type]
            dictionary=log_metrics,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        return log_metrics

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
            scheduler = get_lr_scheduler(self.lr_scheduler)(
                optimizer=optimizer
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}


    def _build_model(self, model: type, model_params: dict[str, Any]) -> type:
        """Instantiate a Pytorch model with the given model parameters.

        Args:
        ----
            model (type):               An unitialized Pytorch model class
                                        defined in ~./archs.

            model_params (dict):        The parameters needed to instantiate
                                        the above Pytorch model class.

        Returns:
        -------
            (type):                     An Pytorch model object.

        """
        return model(**model_params)


    def _compute_loss(
        self,
        y: float | Tensor,
        y_pred: float | Tensor,
        loss_label: str,
    ) -> tuple[float | Tensor, dict[str, float | Tensor]]:
        """Return the loss and the updated the metrics log.

        Args:
        ----
            y (float | Tensor):         The target value from a set of training
                                        pairs.

            y_pred (float | Tensor):    The model's prediction.

            loss_label (str):           Name to be used for labeling purposes.

        Returns:
        -------
            (tuple):                    The computed loss between y and y_pred
                                        and the dictionary that logs the loss.

        """
        log_metrics: dict[str, float | Tensor] = {}
        loss = None

        # set up loss function once
        if not hasattr(self, "loss_functions"):
            self.loss_functions: dict[
                str,
                Callable[..., float | Tensor],
            ] = prepare_function(user_args=self.loss)

        for _function in self.loss_functions:
            function = self.loss_functions[_function]
            loss = function(input=y_pred, target=y)
            log_metrics[loss_label] = loss

        if loss is None:
            logger.error(
                "Something went wrong when trying to compute the loss.",
            )
            e_msg = "The variable 'loss' cannot be None."
            raise TypeError(e_msg)

        return loss, log_metrics


    def _compute_performance(
        self,
        y: float | Tensor,
        y_pred: float | Tensor,
        perf_label: str,
    ) -> dict[str, float | Tensor]:
        """Return the performance scores(s) and the updated the metrics log.

        Args:
        ----
            y (float | tensor):         The target value from a set of training
                                        pairs.

            y_pred (float | tensor):    The model's prediction.

            perf_label (str):           Name to be used for labeling purposes.

        Returns:
        -------
            (tuple):                    The computed score between y and
                                        y_pred and the dictionary that logs
                                        the performance score.

        """
        log_metrics: dict[str, float | Tensor] = {}
        if self.performance_metrics is None:
            logger.error(
                "Something went wrong when trying to compute the performance\
 metric(s).",
            )
            e_msg = "Performance metrics cannot be of NoneType here."
            raise TypeError(e_msg)

        # set up performance functions once
        if not hasattr(self, "perf_functions"):
            self.perf_functions: dict[
                str,
                Callable[..., float | Tensor],
            ] = prepare_function(user_args=self.performance_metrics)

        for _function in self.perf_functions:
            function = self.perf_functions[_function]
            val = function(preds=y_pred, target=y)
            log_metrics[perf_label + _function] = val

        return log_metrics
