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
from .manifold_adversarial_training import gen_man_adv_samples
from momentfm import MOMENTPipeline

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
            output_scaler=None,
            custom_pl_model_kwargs=None,
    ) -> None:
        super().__init__(loss, learning_rate, optimizer, learning_rate_scheduler,
                         performance_metrics, model, model_params, output_scaler)
        self._prev_batch_end_time = None
        self._train_step_start_time = None
        self.feature_encoder = None
        if 'use_feature_encoder' in custom_pl_model_kwargs:
            if custom_pl_model_kwargs['use_feature_encoder']:
                self.feature_encoder = MOMENTPipeline.from_pretrained(
                    "AutonLab/MOMENT-1-small",
                    model_kwargs={'task_name': 'embedding'},
                    # We are loading the model in `embedding` mode to learn representations
                    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
                )
                self.feature_encoder.init()
                for p in self.feature_encoder.parameters():
                    p.requires_grad = False
                self.feature_encoder.eval()
                num_params = sum(p.numel() for p in self.feature_encoder.encoder.parameters())
                print(f"Number of parameters in feature encoder: {num_params}")
        if 'feature_loss_weight' in custom_pl_model_kwargs:
            self.feat_loss_weight = custom_pl_model_kwargs['feature_loss_weight']
        else:
            self.feat_loss_weight = 1.0
        self.criterion = torch.nn.MSELoss()

    def calculate_loss(self, x, y, prefix):

        log_metrics = {}
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        if self.feature_encoder is not None:

            x_moment, x_mask = self.left_pad_to_512_with_mask(x)
            y_moment, y_mask = self.left_pad_to_512_with_mask(y)
            with torch.no_grad():
                x_feat = self.feature_encoder(x_enc=x_moment, input_mask=x_mask).embeddings
            y_feat = self.feature_encoder(x_enc=y_moment, input_mask=y_mask).embeddings
            # calc MSE in feature space
            print('X-feat:')
            print(x_feat.shape)
            print('Y-feat:')
            print(y_feat.shape)
            feat_loss = self.criterion(x_feat, y_feat)
            print(f'Feat loss: {feat_loss.item()}')
        else:
            feat_loss = torch.tensor(0.0)
        log_metrics[prefix + '_feat_loss'] = feat_loss

        input_loss = self.criterion(x, y)
        log_metrics[prefix + '_input_loss'] = input_loss
        rec_loss = input_loss + self.feat_loss_weight * feat_loss
        log_metrics[prefix + '_rec_loss'] = rec_loss
        return rec_loss, log_metrics

    def left_pad_to_512_with_mask(self, x):
        """
        Left-pad a batch of time series to length 512 and create a MOMENT input mask.

        Args:
            x: torch.Tensor of shape [B, C, T]

        Returns:
            x_pad: torch.Tensor of shape [B, C, 512]
            input_mask: torch.Tensor of shape [B, 512]
        """
        b, c, t = x.shape

        if t > 512:
            raise ValueError(f"Expected sequence length <= 512, got {t}")

        pad_len = 512 - t

        if pad_len == 0:
            input_mask = torch.ones((b, 512), device=x.device, dtype=x.dtype)
            return x, input_mask

        x_pad = torch.nn.functional.pad(x, (pad_len, 0), mode="constant", value=0.0)

        input_mask = torch.cat(
            [
                torch.zeros((b, pad_len), device=x.device, dtype=x.dtype),
                torch.ones((b, t), device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )

        return x_pad, input_mask

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
        kl_w, kl = self.model.kl_loss()
        loss = loss + kl_w
        loss_log_metrics['train_loss'] = loss
        loss_log_metrics['train_kl_weighted_loss'] = kl_w
        loss_log_metrics['train_kl_loss'] = kl

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

        kl_w, kl = self.model.kl_loss()
        loss = loss + kl_w
        loss_log_metrics['valid_loss'] = loss
        loss_log_metrics['valid_kl_weighted_loss'] = kl_w
        loss_log_metrics['valid_kl_loss'] = kl
        # compute performance; depends on whether user gave kwargs
        # if self.performance_metrics is None:
        #     perf_log_metrics = {}
        # else:
        #     perf_log_metrics = self._compute_performance(
        #         y=batch['y'],
        #         y_pred=y_pred,
        #         perf_label="valid_perf_",
        #     )
        # valid_real_gen_mean_psd_diff = self.measure_gen_performance(x=x, y=y)

        log_metrics = {
            **loss_log_metrics,
            # 'valid_real_gen_psd_diff': valid_real_gen_mean_psd_diff
        }
        log_metrics['epoch'] = float(self.current_epoch)
        # log_metrics['beta'] = float(self.model.get_betakl())
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
        if hasattr(self.model, "apply_kl_warmup"):
            self.model.apply_kl_warmup(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        # Remove feature encoder weights from the Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict if k.startswith("feature_encoder.")]
        for k in keys_to_remove:
            del state_dict[k]


class PLModel_custom_old_feat_encoder(PLModel):

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
            self.feature_encoder = custom_pl_model_kwargs['feature_encoder']
            self.feature_encoder.train()
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        else:
            self.feature_encoder = None

        if 'feature_loss_weight' in custom_pl_model_kwargs:
            self.feat_loss_weight = custom_pl_model_kwargs['feature_loss_weight']
        else:
            self.feat_loss_weight = 1.0
        self.criterion = torch.nn.MSELoss()

    def calculate_loss(self, x, y, prefix):

        log_metrics = {}
        if self.feature_encoder is not None:
            self.feature_encoder.force_reset()
            with torch.no_grad():
                x_feat = self.feature_encoder(x)
            self.feature_encoder.force_reset()
            y_feat = self.feature_encoder(y)
            # calc MSE in feature space
            feat_loss = self.criterion(x_feat, y_feat)
        else:
            feat_loss = torch.tensor(0.0)
        log_metrics[prefix + '_feat_loss'] = feat_loss

        input_loss = self.criterion(x, y)
        log_metrics[prefix + '_input_loss'] = input_loss
        rec_loss = input_loss + self.feat_loss_weight * feat_loss
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
        kl_w, kl = self.model.kl_loss()
        loss = loss + kl_w
        loss_log_metrics['train_loss'] = loss
        loss_log_metrics['train_kl_weighted_loss'] = kl_w
        loss_log_metrics['train_kl_loss'] = kl

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

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     super().on_train_batch_start(batch, batch_idx, dataloader_idx)
    #
    #     now = time.perf_counter()
    #     if self._prev_batch_end_time is not None:
    #         idle_time = now - self._prev_batch_end_time
    #         self.log("train_batch_idle_time", idle_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    #         print(f"Train batch idle time {idle_time}")
    #     self._train_step_start_time = now
    #
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     now = time.perf_counter()
    #
    #     if self._train_step_start_time is not None:
    #         step_time = now - self._train_step_start_time
    #         self.log("train_batch_step_time", step_time, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    #         print(f"Train batch step time {step_time}")
    #     self._prev_batch_end_time = now



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

        kl_w, kl = self.model.kl_loss()
        loss = loss + kl_w
        loss_log_metrics['valid_loss'] = loss
        loss_log_metrics['valid_kl_weighted_loss'] = kl_w
        loss_log_metrics['valid_kl_loss'] = kl
        # compute performance; depends on whether user gave kwargs
        # if self.performance_metrics is None:
        #     perf_log_metrics = {}
        # else:
        #     perf_log_metrics = self._compute_performance(
        #         y=batch['y'],
        #         y_pred=y_pred,
        #         perf_label="valid_perf_",
        #     )
        # valid_real_gen_mean_psd_diff = self.measure_gen_performance(x=x, y=y)

        log_metrics = {
            **loss_log_metrics,
            # 'valid_real_gen_psd_diff': valid_real_gen_mean_psd_diff
        }
        log_metrics['epoch'] = float(self.current_epoch)
        # log_metrics['beta'] = float(self.model.get_betakl())
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
    def on_validation_epoch_start(self):
        """ Reset the model internal states for new validation epoch."""
        if hasattr(self.model, 'force_reset'):
            self.model.force_reset()
        if hasattr(self.feature_encoder, 'force_reset'):
            self.feature_encoder.force_reset()
    def on_validation_epoch_end(self) -> None:
        print('lmao')

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if batch_idx == 0:
            if hasattr(self.model, 'force_reset'):
                self.model.force_reset()
            if hasattr(self.feature_encoder, 'force_reset'):
                self.feature_encoder.force_reset()
        else:
            if hasattr(self.model, 'update_states'):
                self.model.update_states(batch['ist_idx'][0], batch['x'].device)
            if hasattr(self.model, 'hard_set_states'):
                self.model.hard_set_states(batch['ist_idx'][-1])
            if hasattr(self.feature_encoder, 'update_states'):
                self.feature_encoder.update_states(batch['ist_idx'][0], batch['x'].device)
            if hasattr(self.feature_encoder, 'hard_set_states'):
                self.feature_encoder.hard_set_states(batch['ist_idx'][-1])
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])
        if hasattr(self.feature_encoder, 'update_states'):
            self.feature_encoder.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.feature_encoder, 'hard_set_states'):
            self.feature_encoder.hard_set_states(batch['ist_idx'][-1])
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])
        if hasattr(self.feature_encoder, 'update_states'):
            self.feature_encoder.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.feature_encoder, 'hard_set_states'):
            self.feature_encoder.hard_set_states(batch['ist_idx'][-1])
    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """ Update model internal states if applicable. Also hard sets the internal states to the
        final prediction point in the batch in case of variable length inputs."""
        if hasattr(self.model, 'update_states'):
            self.model.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.model, 'hard_set_states'):
            self.model.hard_set_states(batch['ist_idx'][-1])
        if hasattr(self.feature_encoder, 'update_states'):
            self.feature_encoder.update_states(batch['ist_idx'][0], batch['x'].device)
        if hasattr(self.feature_encoder, 'hard_set_states'):
            self.feature_encoder.hard_set_states(batch['ist_idx'][-1])

    def on_save_checkpoint(self, checkpoint):
        # Remove feature encoder weights from the Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict if k.startswith("feature_encoder.")]
        for k in keys_to_remove:
            del state_dict[k]

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
            gen_model_path,
            attack_params

    ) -> None:
        super().__init__(loss, learning_rate, optimizer, learning_rate_scheduler,
                         performance_metrics, model, model_params, output_scaler)

        self.z_stats = None
        self.attack_epsilon = attack_params["epsilon"]
        self.attack_steps = attack_params["steps"]
        self.attack_step_size = attack_params["step_size"] #make sure to check whether this actually takes the std into account
        if self.attack_step_size is None:
            self.attack_step_size = self.attack_epsilon/self.attack_steps
        self.gen_model = 'ay lmao'# TT to write gen_model loading using KnowIT functions

    def calculate_z_stats(self):
        """
        Compute latent statistics over the full training set using the encoder mean mu.

        Returns
        -------
        dict
            {
                "std": [latent_dim],
                "min": [latent_dim],
                "max": [latent_dim],
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
                z_list.append(mu.detach())

        z_all = torch.cat(z_list, dim=0)  # [N, latent_dim]

        stats_dict = {
            "std": z_all.std(dim=0),
            "min": z_all.min(dim=0).values,
            "max": z_all.max(dim=0).values,
        }

        return stats_dict

    def on_train_start(self):
        super().on_train_start()
        if self.z_stats is None:
            self.z_stats = self.calculate_z_stats()

    def training_step(self, batch: dict[str, Any], batch_idx: int):  # type: ignore[return-value]  # noqa: ANN201, ARG002
        """Compute loss and optional metrics, log metrics, and return the loss.

        Overrides the method in pl.LightningModule.
        """
        forward = getattr(self.model, "forward")  # noqa: B009
        x = batch['x']
        y = batch['y']
        x_adv = gen_man_adv_samples(
            model=self.model,
            gen_model=self.gen_model,
            x_natural=x,
            y=y,
            step_size=self.attack_step_size,
            epsilon=self.attack_epsilon,
            perturb_steps=self.attack_steps,
            stats_dict=self.z_stats,
        )
        y_pred = forward(x=x_adv,y=y)

        # compute loss; depends on whether user gave kwargs
        loss, loss_log_metrics = self._compute_loss(
            y=y,
            y_pred=y_pred,
            loss_label="train_loss",
        )
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










