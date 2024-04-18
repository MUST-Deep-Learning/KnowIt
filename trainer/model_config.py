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
    get_loss_function,
    get_lr_scheduler,
    get_optim,
    get_performance_metric,
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
        learning_rate_scheduler: dict[str, Any],
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
        log_metrics = {}

        def _prepare_loss_function() -> (
            dict[
                str,
                Callable[..., float | Tensor]
                | tuple[Callable[..., float | Tensor], dict[str, Any]],
            ]
        ):
            loss_functions = {}
            if isinstance(self.loss, dict):
                for loss_metric in self.loss:
                    loss_kwargs = self.loss[loss_metric]
                    loss = get_loss_function(loss_metric)
                    loss_functions = {loss_metric: (loss, loss_kwargs)}
            else:
                loss = get_loss_function(self.loss)
                loss_functions = {self.loss: loss}

            return loss_functions   #TODO: Fix the error from Pylance here

        if not hasattr(self, "loss_functions"):
            self.loss_functions = _prepare_loss_function()

        for _function in self.loss_functions:
            if type(self.loss_functions[_function]) is tuple:
                function = self.loss_functions[_function][0]
                f_kwargs = self.loss_functions[_function][1]
                loss = function(y_pred, y, **f_kwargs)
            else:
                function = self.loss_functions[_function](y_pred, y)
                log_metrics[loss_label] = loss

        return loss, log_metrics  # type: ignore[return-value]

    def _compute_performance(
        self,
        y: float | Tensor,
        y_pred: float | Tensor,
        perf_label: str,
    ) -> dict[str, float | Tensor]:
        """Return the performance scores(s) and the updated the metrics log.

        Args:
        ----
            log_metrics (dict):         The dictionary that logs the metrics.

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
        log_metrics = {}

        if isinstance(self.performance_metrics, dict):
            for p_metric in self.performance_metrics:
                perf_kwargs = self.performance_metrics[p_metric]
                log_metrics[perf_label + p_metric] = get_performance_metric(
                    p_metric,
                )(y_pred, y, **perf_kwargs)
        elif self.performance_metrics is not None:
            log_metrics[perf_label + self.performance_metrics] = (
                get_performance_metric(self.performance_metrics)(y_pred, y)
            )

        return log_metrics  # type: ignore[return-value]

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201
        """Return the batch loss.

        Overrides the method in pl.LightningModule.
        """
        x = batch["x"]
        y = batch["y"]

        y_pred = self.model.forward(x)  # type: ignore[return-value]
        if type(y_pred) not in (float, Tensor):
            t = type(y_pred)
            emsg = f"The model's forward method gives return type {t}.\
                Expecting float or torch.Tensor."
            raise TypeError(emsg)

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

    def validation_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201
        """Return the batch loss.

        Overrides the method in pl.LightningModule.
        """
        x = batch["x"]
        y = batch["y"]

        y_pred = self.model.forward(x)  # type: ignore[return-value]
        if type(y_pred) not in (float, Tensor):
            t = type(y_pred)
            emsg = f"The model's forward method gives return type {t}.\
                Expecting float or torch.Tensor."
            raise TypeError(emsg)

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
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Return the batch loss.

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

        y_pred = self.model.forward(x)  # type: ignore[return-value]
        if type(y_pred) not in (float, Tensor):
            t = type(y_pred)
            emsg = f"The model's forward method gives return type {t}.\
                Expecting float or torch.Tensor."
            raise TypeError(emsg)

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

    def configure_optimizers(self):
        # get user's optimizer
        if self.optimizer:
            if isinstance(self.optimizer, dict):  # optimizer has kwargs
                for optim in self.optimizer.keys():
                    opt_kwargs = self.optimizer[optim]
                    optimizer = get_optim(optim)(
                        self.model.parameters(), lr=self.lr, **opt_kwargs
                    )
            elif isinstance(self.optimizer, str):  # optimizer has no kwargs
                print(self.model.parameters().__doc__)
                optimizer = get_optim(self.optimizer)(
                    self.model.parameters(), lr=self.lr
                )

        # get user's learning rate scheduler
        if self.lr_scheduler:
            if isinstance(self.lr_scheduler, dict):  # lr schedular has kwargs
                lr_dict = {}
                for sched in self.lr_scheduler.keys():
                    sched_kwargs = self.lr_scheduler[sched]
                    if "monitor" in sched_kwargs:
                        monitor = sched_kwargs.pop("monitor")
                        lr_dict["monitor"] = monitor
                    scheduler = get_lr_scheduler(sched)(
                        optimizer, **sched_kwargs
                    )
                    lr_dict["scheduler"] = scheduler
                return {"optimizer": optimizer, "lr_scheduler": lr_dict}
            elif isinstance(
                self.lr_scheduler, str
            ):  # lr scheduler has no kwargs
                scheduler = get_lr_scheduler(self.lr_scheduler)(optimizer)
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            # no scheduler
            return {"optimizer": optimizer}
