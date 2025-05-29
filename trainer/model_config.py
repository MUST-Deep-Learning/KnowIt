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
__author__ = "randlerabe@gmail.com"
__description__ = "Constructs a Pytorch Lightning model class."

from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl

from helpers.fetch_torch_mods import (
    get_lr_scheduler,
    get_optim,
    prepare_function,
)
from helpers.logger import get_logger

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = get_logger()


class PLModel(pl.LightningModule):
    """Wrapper class to prepare model for Pytorch Lightning's Trainer.

    The class initializes a Pytorch model and defines the training,
    validation, and test steps over a batch. The optimizer configuration is
    also set inside this class. This is required for Pytorch Lightning's
    Trainer.

    Parameters
    ----------
    loss : str, dict[str, Any]
        The loss function to be used during training. The string must match the
        name in Pytorch's functional library. See:
        https://pytorch.org/docs/stable/nn.functional.html#loss-functions

    learning_rate : float
        The learning rate to be used during training.

    optimizer : str | dict[str, Any]
        The optimizer to be used during training. The string must match the
        name in Pytorch's optimizer library. See:
        https://pytorch.org/docs/stable/nn.functional.html#loss-functions

    learning_rate_scheduler : None | str | dict[str, Any], default=None
        The learning rate scheduler to be used during training. If not None, a
        dictionary must be given of the form

            ``{scheduler: scheduler_kwargs}``

        where
            scheduler:  A string that specifies the Pytorch scheduler to be
            used. Must match names found here:
            https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler

            scheduler_kwargs: A dictionary of kwargs required for 'scheduler'.

    performance_metrics : None | str | dict[str, Any], default=None
        Performance metrics to be logged during training. If type=dict, then
        the dictionary must be given of the form

            ``{metric: metric_kwargs}``

        where
            metric: A string that specifies the TORCHMETRICS metric to be
            used. Must match the functional interface names found here:
            https://lightning.ai/docs/torchmetrics/stable/

            metric_kwargs: A dictionary of kwargs required for 'metric'.

    model : Module
        An unitialized Pytorch model class defined in the user's model direc-
        tory.

    model_params : dict[str, Any]
        The parameters needed to instantiate the above Pytorch model class.

    output_scaler : None | object, default=None
        The scaling object to rescale the model outputs to original ranges (if applicable)
        during performance calculations. Must have an appropriate `inverse_transform` function.
        If None, no rescaling is performed. Note, only applicable to logged metrics,
        gradients are still calculated with scaled outputs (if applicable).


    Attributes
    ----------
    loss : str, dict[str, Any]
        Stores the name of the loss function to be used during training and any
        additional kwargs if needed.

    lr : float
        Stores the value of the learning rate to be used during training.

    lr_scheduler : None | str | dict[str, Any]
        Stores the name of the learning rate scheduler to be used during
        training and any additional kwargs if needed.

    optimizer : str | dict[str, Any]
        Stores the name of the optimizer to be used during training and any
        additional kwargs if needed.

    performance_metrics : None | str | dict[str, Any], default=None
        Stores the name of the performance metric(s) to be used during training
        and any additional kwargs if needed.

    output_scaler : None | object, default=None
        The scaling object to rescale the model outputs to original ranges (if applicable)
        during performance calculations. Must have an appropriate `inverse_transform` function.
        If None, no rescaling is performed. Note, only applicable to logged metrics,
        gradients are still calculated with scaled outputs (if applicable).

    model : Module
        An initialized Pytorch model.

    """

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

        self.loss = loss
        self.lr = learning_rate
        self.lr_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.performance_metrics = performance_metrics
        self.output_scaler = output_scaler

        self.model = self._build_model(model, model_params)

        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(batch)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'],
            y_pred=y_pred,
            loss_label="train_loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=batch['y'],
                y_pred=y_pred,
                perf_label="train_perf_",
            )

        log_metrics = {
            **loss_log_metrics,
            **perf_log_metrics,
        }
        log_metrics['epoch'] = float(self.current_epoch)

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

        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(batch)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self._compute_loss(
            y=batch['y'],
            y_pred=y_pred,
            loss_label="valid_loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=batch['y'],
                y_pred=y_pred,
                perf_label="valid_perf_",
            )

        log_metrics = {
            **loss_log_metrics,
            **perf_log_metrics,
        }
        log_metrics['epoch'] = float(self.current_epoch)

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

        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(batch)

        # compute loss; depends on whether user gave kwargs
        _, loss_log_metrics = self._compute_loss(
            y=batch['y'],
            y_pred=y_pred,
            loss_label=current_loader + "loss",
        )

        # compute performance; depends on whether user gave kwargs
        if self.performance_metrics is None:
            perf_log_metrics = {}
        else:
            perf_log_metrics = self._compute_performance(
                y=batch['y'],
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
            scheduler = get_lr_scheduler(self.lr_scheduler)(optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def _build_model(self, model: Module, model_params: dict[str, Any]) -> type:
        """Instantiate a Pytorch model with the given model parameters.

        Parameters
        ----------
        model : Module
            An unitialized Pytorch model class defined in a user's model
            directory.

        model_params : dict[str, Any]
            The parameters needed to instantiate the above Pytorch model
            class.

        Returns
        -------
        type
            A Pytorch model object.

        """
        return model(**model_params)

    def _compute_loss(
        self,
        y: float | Tensor,
        y_pred: float | Tensor,
        loss_label: str,
    ) -> tuple[float | Tensor, dict[str, float | Tensor]]:
        """Return the loss and the metrics log.

        Parameters
        ----------
        y : float | Tensor
            The target value from a set of training pairs.

        y_pred : float | Tensor
            The model's prediction.

        loss_label : str
            Name to be used for labeling purposes.

        Returns
        -------
        tuple
            The computed loss between y and y_pred and the dictionary that
            logs the loss.

        """
        log_metrics: dict[str, float | Tensor] = {}
        loss = None

        # set up loss function once
        if not hasattr(self, "loss_functions"):
            self.loss_functions: dict[
                str,
                Callable[..., float | Tensor],
            ] = prepare_function(user_args=self.loss, is_loss=True)

        for _function in self.loss_functions:
            function = self.loss_functions[_function]
            loss = function(input=y_pred, target=y)
            log_metrics[loss_label] = loss

            if self.output_scaler is not None:
                y_pred = self.output_scaler.inverse_transform(y_pred.clone().detach().cpu()).to(self.device)
                y = self.output_scaler.inverse_transform(y.clone().detach().cpu()).to(self.device)
                loss_rescaled = function(input=y_pred, target=y)
                log_metrics[loss_label] = loss_rescaled


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
        """Return the performance scores(s) and the metrics log.

        Parameters
        ----------
        y : float | tensor
            The target value from a set of training pairs.

        y_pred : float | tensor
            The model's prediction.

        perf_label : str
            Name to be used for labeling purposes.

        Returns
        -------
        tuple
            The computed score between y and y_pred and the dictionary that
            logs the performance score.

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
            ] = prepare_function(
                user_args=self.performance_metrics,
                is_loss=False,
            )

        for _function in self.perf_functions:
            function = self.perf_functions[_function]
            if self.output_scaler is not None:
                y_pred = self.output_scaler.inverse_transform(y_pred.clone().detach().cpu()).to(self.device)
                y = self.output_scaler.inverse_transform(y.clone().detach().cpu()).to(self.device)
            val = function(preds=y_pred, target=y)
            log_metrics[perf_label + _function] = val


        return log_metrics
