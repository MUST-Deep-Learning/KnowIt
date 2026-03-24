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

import pytorch_lightning as pl
from torch import argmax
import time

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

        self.save_hyperparameters(ignore=['model', 'custom_pl_model_kwargs'])

    def on_train_epoch_end(self):
        """ Set the next training epoch number in the CustomSampler.

        This is done to manage potential stochasticity (which is connected to the epoch number).

        """
        if hasattr(self.trainer.train_dataloader.batch_sampler, 'set_epoch'):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch+1)

    def on_train_epoch_start(self):
        """ Reset the model internal states for new training epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_validation_epoch_start(self):
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if batch_idx == 0:
            if hasattr(self.model, 'force_reset'):
                self.model.force_reset()
        else:
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

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        forward = getattr(self.model, "forward")  # noqa: B009
        y_pred = forward(batch['x'])

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
        y_pred = forward(batch['x'])

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
        y_pred = forward(batch['x'])

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
            self.loss_functions = prepare_function(user_args=self.loss, is_loss=True)

        for _function in self.loss_functions:
            function, to_argmax, to_flatten = self.loss_functions[_function]
            if to_argmax:
                y = argmax(y, dim=1).to(self.device)
            if to_flatten[0]:
                y = y.flatten(start_dim=to_flatten[1][0], end_dim=to_flatten[1][1]).to(self.device)
                y_pred = y_pred.flatten(start_dim=to_flatten[1][0], end_dim=to_flatten[1][1]).to(self.device)
            loss = function(input=y_pred, target=y)
            log_metrics[loss_label] = loss

            if self.output_scaler is not None:
                y_pred = self.output_scaler.inverse_transform(y_pred.clone().detach().cpu()).to(self.device)
                if not to_argmax:
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
            self.perf_functions = prepare_function( user_args=self.performance_metrics, is_loss=False)

        for _function in self.perf_functions:
            function, to_argmax, to_flatten = self.perf_functions[_function]
            if self.output_scaler is not None:
                y_pred = self.output_scaler.inverse_transform(y_pred.clone().detach().cpu()).to(self.device)
                y = self.output_scaler.inverse_transform(y.clone().detach().cpu()).to(self.device)
            if to_argmax:
                y = argmax(y, dim=1).to(self.device)
            if to_flatten[0]:
                y = y.flatten(start_dim=to_flatten[1][0], end_dim=to_flatten[1][1]).to(self.device)
                y_pred = y_pred.flatten(start_dim=to_flatten[1][0], end_dim=to_flatten[1][1]).to(self.device)
            val = function(preds=y_pred, target=y)
            log_metrics[perf_label + _function] = val


        return log_metrics

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
            output_scaler,
            custom_pl_model_kwargs=None,

    ) -> None:
        super().__init__(loss, learning_rate, optimizer, learning_rate_scheduler,
                         performance_metrics, model, model_params, output_scaler)

class PLModel_custom_mat(PLModel):

    def __init__(
            self,
            loss,
            learning_rate,
            optimizer,
            learning_rate_scheduler,
            performance_metrics,
            model,
            model_params,
            output_scaler,
            custom_pl_model_kwargs=None,

    ) -> None:
        super().__init__(loss, learning_rate, optimizer, learning_rate_scheduler,
                         performance_metrics, model, model_params, output_scaler)
        self.z_stats = None
        if 'attack_params' not in custom_pl_model_kwargs or 'gen_model' not in custom_pl_model_kwargs:
            logger.error('No attack parameters or gen_model in custom_pl_model_kwargs')
            exit(101)
        attack_params = custom_pl_model_kwargs["attack_params"]
        self.attack_epsilon = attack_params["epsilon"]
        self.attack_steps = attack_params["steps"]
        if 'beta_trades' in custom_pl_model_kwargs:
            self.beta_trades = custom_pl_model_kwargs['beta_trades']
        else:
            self.beta_trades = 1.0
        if 'use_gen_target' in custom_pl_model_kwargs:
            self.use_gen_target = bool(custom_pl_model_kwargs['use_gen_target'])
        else:
            self.use_gen_target = False
        # if "step_size" in attack_params:
        #     self.attack_step_size = attack_params["step_size"]
        # else:
        #     self.attack_step_size = self.attack_epsilon/self.attack_steps
        self.gen_model = custom_pl_model_kwargs['gen_model']
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def calculate_z_stats(self):
        """
        Compute latent statistics over the full training set using the encoder mean mu.

        Supports latent outputs of shape:
            [B, F]     -> one stat per latent dimension
            [B, F, T]  -> one stat per channel, aggregated globally over batch and time

        Returns
        -------
        dict
            {
                "std": [F],
                "min": [F],
                "max": [F],
            }
        """
        self.gen_model.eval()

        z_list = []
        train_loader = self.trainer.train_dataloader

        with torch.no_grad():
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                mu, logvar = self.gen_model.encode(x, y)

                if mu.ndim not in (2, 3):
                    raise ValueError(
                        f"Expected mu to have shape [B, F] or [B, F, T], "
                        f"but got shape {tuple(mu.shape)}"
                    )

                z_list.append(mu.detach())

        if len(z_list) == 0:
            raise ValueError("Training dataloader produced no batches, cannot calculate z stats.")

        z_all = torch.cat(z_list, dim=0)

        if z_all.ndim == 2:
            # z_all: [N, F]
            reduce_dims = 0
        elif z_all.ndim == 3:
            # z_all: [N, F, T] -> aggregate globally over samples and time
            reduce_dims = (0, 2)
        else:
            raise ValueError(
                f"After concatenation, expected z_all to be 2D or 3D, "
                f"but got shape {tuple(z_all.shape)}"
            )

        stats_dict = {
            "std": z_all.std(dim=reduce_dims),
            "min": z_all.amin(dim=reduce_dims),
            "max": z_all.amax(dim=reduce_dims),
        }

        return stats_dict

    def on_train_start(self):
        super().on_train_start()
        if self.z_stats is None:
            self.z_stats = self.calculate_z_stats()

    def get_latent(self, x, y):
        mu, logvar = self.gen_model.encode(x, y)
        return mu

    def gen_man_adv_samples(
            self,
            x_natural,
            y
    ):

        self.model.eval()
        self.gen_model.eval()
        # self.track_bn_stats(False)

        z_natural = self.get_latent(x_natural, y).detach()
        x_natural_dec = torch.clamp(self.gen_model.decode(z_natural, y), 0, 1)
        with torch.no_grad():
            x_natural_dec_logits = F.softmax(self.model(x_natural_dec), dim=1).detach()

        z_std = torch.as_tensor(self.z_stats["std"], device=x_natural.device, dtype=x_natural.dtype)
        z_min = torch.as_tensor(self.z_stats["min"], device=x_natural.device, dtype=x_natural.dtype)
        z_max = torch.as_tensor(self.z_stats["max"], device=x_natural.device, dtype=x_natural.dtype)

        # feature/channel dimension is always dim=1
        num_features = z_natural.shape[1]
        # reshape stats for broadcasting
        if z_natural.ndim == 2:
            # z_natural: [B, F]
            view_shape = (1, num_features)
        else:
            # z_natural: [B, F, T]
            view_shape = (1, num_features, 1)

        z_std = torch.clamp(z_std, min=1e-3)
        z_std = z_std.view(*view_shape)
        z_min = z_min.view(*view_shape)
        z_max = z_max.view(*view_shape)

        epsilon_abs = self.attack_epsilon * z_std
        attack_step_size = torch.mean(epsilon_abs).item()/self.attack_steps

        # random start inside latent epsilon-ball around z_natural
        z_adv = z_natural + (2.0 * torch.rand_like(z_natural) - 1.0) * epsilon_abs
        z_adv = torch.clamp(z_adv, z_min, z_max)

        for _ in range(self.attack_steps):
            z_adv.requires_grad_()

            with torch.enable_grad():
                x_adv = self.gen_model.decode(z_adv, y)
                x_adv = torch.clamp(x_adv, 0, 1)
                logits = self.model(x_adv)
                loss = self.criterion_kl(F.log_softmax(logits, dim=1), x_natural_dec_logits)
                # loss = criterion_ce(logits, y_ce)
                grad = torch.autograd.grad(loss, [z_adv])[0]

            z_adv = z_adv.detach() + attack_step_size * torch.sign(grad.detach())
            # project back into per-feature epsilon box around natural latent
            z_adv = torch.min(torch.max(z_adv, z_natural - epsilon_abs), z_natural + epsilon_abs)
            z_adv = torch.clamp(z_adv, z_min, z_max)

        x_adv = torch.clamp(self.gen_model.decode(z_adv, y), 0, 1)
        # self.track_bn_stats(True)
        return x_adv.detach(), x_natural_dec.detach()

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        forward = getattr(self.model, "forward")  # noqa: B009
        x = batch['x']
        y = batch['y']
        x_adv, x_dec = self.gen_man_adv_samples(x_natural=x, y=y)
        # self.plot_adv_examples(x, x_adv)
        self.model.train()
        self.track_bn_stats(False)
        y_pred_adv = forward(x=x_adv)
        if self.use_gen_target:
            y_pred_dec = forward(x=x_dec)
        self.track_bn_stats(True)
        y_pred = forward(x=x)

        if self.use_gen_target:
            p_pred_target = F.softmax(y_pred_dec, dim=1).clamp(min=1e-12) # to prevents possible NaNs
            p_pred_target = p_pred_target / p_pred_target.sum(dim=1, keepdim=True)
        else:
            p_pred_target = F.softmax(y_pred, dim=1).clamp(min=1e-12)
            p_pred_target = p_pred_target / p_pred_target.sum(dim=1, keepdim=True)

        p_pred_adv = F.log_softmax(y_pred_adv, dim=1)

        if not torch.isfinite(y_pred).all():
            raise RuntimeError("Non-finite y_pred before KL")
        if not torch.isfinite(y_pred_adv).all():
            raise RuntimeError("Non-finite y_pred_adv before KL")
        if self.use_gen_target and not torch.isfinite(y_pred_dec).all():
            raise RuntimeError("Non-finite y_pred_dec before KL")

        loss_natural = F.cross_entropy(y_pred, y)
        loss_adv = self.criterion_kl(p_pred_adv, p_pred_target) * (1/x.shape[0])
        loss = loss_natural + self.beta_trades * loss_adv
        loss_log_metrics = {}
        loss_log_metrics['train_clean_loss'] = loss_natural
        loss_log_metrics['train_adv_loss'] = loss_adv
        loss_log_metrics['train_adv_weighted_loss'] = loss_adv * self.beta_trades
        loss_log_metrics['train_loss'] = loss

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

    def on_save_checkpoint(self, checkpoint):
        # Remove feature encoder weights from the Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict if k.startswith("gen_model.")]
        for k in keys_to_remove:
            del state_dict[k]

    def track_bn_stats(self, track_stats=True):
        """
        If track_stats=False, do not update BN running mean and variance and vice versa.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = track_stats
    def plot_adv_examples(self, x, x_adv):
        """
        Plot the first 2 examples in the batch, with original x in the left column
        and adversarial x_adv in the right column.

        Assumes tensors have shape [B, T, F].
        Each subplot shows all features/components over time for one sample.
        """
        import matplotlib.pyplot as plt

        x = x.detach().cpu()
        x_adv = x_adv.detach().cpu()

        num_examples = min(2, x.shape[0])
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples), squeeze=False)

        for i in range(num_examples):
            # x[i]: [T, F]
            axes[i, 0].plot(x[i])
            axes[i, 0].set_title(f"Sample {i} - original")
            axes[i, 0].set_xlabel("Time")
            axes[i, 0].set_ylabel("Value")

            axes[i, 1].plot(x_adv[i])
            axes[i, 1].set_title(f"Sample {i} - adversarial")
            axes[i, 1].set_xlabel("Time")
            axes[i, 1].set_ylabel("Value")

        plt.tight_layout()
        plt.show()

class PLModel_custom_vae(PLModel):

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
        self._prev_batch_end_time = None
        self._train_step_start_time = None

        if 'feature_encoder' in custom_pl_model_kwargs:
            feature_encoder = custom_pl_model_kwargs['feature_encoder']
            feature_layers = ["network.0", "network.1", "network.2"]
            self.feature_encoder = FeatureHookExtractor(feature_encoder, feature_layers)
            self.feature_encoder.eval()
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        else:
            self.feature_encoder = None

        if 'feature_loss_weight' in custom_pl_model_kwargs:
            self.feat_loss_weight = custom_pl_model_kwargs['feature_loss_weight']
        else:
            self.feat_loss_weight = 1.0

        if 'diff_loss_weight' in custom_pl_model_kwargs:
            self.diff_loss_weight = custom_pl_model_kwargs['diff_loss_weight']
        else:
            self.diff_loss_weight = 0.0
        self.criterion = torch.nn.MSELoss()

    def calculate_loss(self, x, y, prefix):

        log_metrics = {}
        if (self.feature_encoder is not None) and (self.feat_loss_weight > 0.0):
            with torch.no_grad():
                x_feats = self.feature_encoder(x)
            y_feats = self.feature_encoder(y)
            # calc MSE in feature space
            feat_loss = 0.0
            for xf, yf in zip(x_feats, y_feats):
                feat_loss = feat_loss + self.criterion(xf, yf)
            feat_loss = feat_loss / len(x_feats)
        else:
            feat_loss = torch.tensor(0.0)
        log_metrics[prefix + '_feat_loss'] = feat_loss

        if self.diff_loss_weight > 0.0:
            dx = x[:, 1:, :] - x[:, :-1, :]
            dy = y[:, 1:, :] - y[:, :-1, :]
            diff_loss = self.criterion(dx, dy)
        else:
            diff_loss = torch.tensor(0.0)
        input_loss = self.criterion(x, y)
        log_metrics[prefix + '_input_loss'] = input_loss
        log_metrics[prefix + '_diff_loss'] = diff_loss
        rec_loss = (
                input_loss
                + self.feat_loss_weight * feat_loss
                + self.diff_loss_weight * diff_loss
        )
        log_metrics[prefix + '_rec_loss'] = rec_loss
        return rec_loss, log_metrics

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        forward = getattr(self.model, "forward")  # noqa: B009
        x = batch['x']
        y = batch['y']
        y_pred = forward(x=x,y=y)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self.calculate_loss(
            x=x,
            y=y_pred,
            prefix="train",
        )
        if float(self.model.get_betakl()) > 0.0:
            kl_w, kl = self.model.kl_loss()
            loss_log_metrics['train_kl_weighted_loss'] = kl_w
            loss_log_metrics['train_kl_loss'] = kl
            loss = loss + kl_w
        loss_log_metrics['train_loss'] = loss


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
        log_metrics['beta'] = float(self.model.get_betakl())
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
        x = batch['x']
        y = batch['y']
        y_pred = forward(x=x,y=y)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self.calculate_loss(
            x=x,
            y=y_pred,
            prefix="valid",
        )

        if float(self.model.get_betakl()) > 0.0:
            kl_w, kl = self.model.kl_loss()
            loss = loss + kl_w
            loss_log_metrics['valid_kl_weighted_loss'] = kl_w
            loss_log_metrics['valid_kl_loss'] = kl
        loss_log_metrics['valid_loss'] = loss
        loss_log_metrics['valid_kl_weighted_loss'] = kl_w
        loss_log_metrics['valid_kl_loss'] = kl
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
            **perf_log_metrics
        }
        log_metrics['epoch'] = float(self.current_epoch)
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
        x = batch['x']
        y = batch['y']
        y_pred = forward(x=x,y=y)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self.calculate_loss(
            x=x,
            y=y_pred,
            prefix="current_loader",
        )
        kl_w, kl = self.model.kl_loss()
        loss = loss + kl_w
        loss_log_metrics[current_loader+'loss'] = loss
        loss_log_metrics[current_loader+'kl_weighted_loss'] = kl_w
        loss_log_metrics[current_loader+'kl_loss'] = kl
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
    def on_train_epoch_start(self):
        """ Reset the model internal states for new training epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()
        if hasattr(self.feature_encoder, 'force_reset'):
            self.feature_encoder.force_reset()
        if hasattr(self.model, "apply_kl_warmup"):
            self.model.apply_kl_warmup(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        # Remove feature encoder weights from the Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict if k.startswith("feature_encoder.")]
        for k in keys_to_remove:
            del state_dict[k]

class FeatureHookExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_names: list[str]):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features = {}
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):
        name_to_module = dict(self.model.named_modules())

        for layer_name in self.layer_names:
            if layer_name not in name_to_module:
                raise ValueError(f"Layer '{layer_name}' not found in model. "
                                 f"Available names: {list(name_to_module.keys())}")

            module = name_to_module[layer_name]
            hook = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook(module, inputs, output):
            self._features[layer_name] = output
        return hook

    def forward(self, x):
        self._features = {}
        _ = self.model(x)
        return [self._features[name] for name in self.layer_names]

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []



