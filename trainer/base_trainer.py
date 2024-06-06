"""
-----------
BaseTrainer
-----------

The "BaseTrainer" is an abstract class that functions as the interface bet-
ween the context class ``KITrainer'' and any of the concrete trainer state
objects.

The "BaseTrainer" class stores the user's parameters and defines a set of
abstract methods to be inherited by the trainer state objects.

Note that some kwargs in the constructor require additional parameters (for
example, the learning rate scheduler). In this case, the parameter is always
provided as a dictionary with the keys as strings specifying the additional
parameters.

For example, if one wants to use the 'ReduceLROnPlateau' scheduler
from Pytorch, then one can specify it as a string

    lr_scheduler = 'ReduceLROnPlateau'

which will use the default values for the scheduler. However, if one wants to
use use custom values or the scheduler requires additional kwargs, then the
scheduler should be passed as a dictionary such as

    lr_scheduler = {
        'ReduceLROnPlateau':{
            'factor': 0.2,
            'patience': 5,
            'threshold': 0.001
        }
    }
The same idea holds for any other kwarg in "BaseTrainer" that might need addit-
ional parameters.
"""  # noqa: INP001, D205, D212, D400, D415

from __future__ import annotations  # required for Python versions <3.9

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the abstract BaseTrainer class."

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data.dataloader import DataLoader
    from torch.nn import Module

    from trainer.trainer import KITrainer

from abc import ABC, abstractmethod

import torch
from pytorch_lightning import seed_everything

from helpers.logger import get_logger

logger = get_logger()


class BaseTrainer(ABC):
    """Abstract class to interface between the context class "KITrainer" and a
    trainer state.

    "BaseTrainer" will initialize necessary and optional kwargs to be used by
    any of the KnowIt Trainer states. It also defines abstract methods that are
    to be defined in each state object.

    Inherits:
    --------
        ABC: type
            Used to define abstract class.

    """  # noqa: D205

    def __init__(
        self,
        model: Module,
        model_params: dict[str, Any],
        out_dir: str,
        device: str,
        loss_fn: str | dict[str, Any],
        optim: str | dict[str, Any],
        max_epochs: int,
        learning_rate: float,
        lr_scheduler: None | str | dict[str, Any] = None,
        performance_metrics: None | str | dict[str, Any] = None,
        early_stopping_args: None | dict[str, Any] = None,
        ckpt_mode: str = "min",
        *,
        return_final: bool = False,
        mute_logger: bool = False,
        seed: None | int = 123,
    ) -> None:
        """BaseTrainer constructor.

        Parameters
        ----------
            model: Module
                The Pytorch model architecture defined by the user in their
                models directory.

            model_params: dict[str, Any]
                The parameters required to initialize model.

            out_dir (str):
                The directory to save the model's checkpoint file.

            device: str
                The device on which training is to be performed (cpu or gpu).

            loss_fn: str | dict
                The loss function to be used during training. The string must
                match the name in Pytorch's functional library. See:
                https://pytorch.org/docs/stable/nn.functional.html#loss-functions

            optim: str | dict
                The optimizer to be used during training. The string must
                match the name in Pytorch's optimizer library. See:
                https://pytorch.org/docs/stable/optim.html#algorithms

            max_epochs: int
                The number of training iterations, where a single iteration is
                over the entire training set.

            learning_rate: float
                The learning rate to be used during parameter updates. It
                controls the size of the updates.

            lr_scheduler: None | str | dict, default=None
                The learning rate scheduler to be used during training. If not
                None, a dictionary must be given of the form
                    ``{scheduler: scheduler_kwargs}``
                where:
                    scheduler:
                        A string that specifies the Pytorch scheduler to be
                        used. Must match names found here:
                        https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler

                    scheduler_kwargs:
                        A dictionary of kwargs required for 'scheduler'.

            performance_metrics: None | str | dict, default=None
                Performance metrics to be logged during training. If
                type=dict, then the dictionary must be given of the form
                    {metric: metric_kwargs},
                where
                    metric: A string that specifies the TORCHMETRICS metric to
                    be used. Must match the functional interface names found
                    here: https://lightning.ai/docs/torchmetrics/stable/

                    metric_kwargs:  A dictionary of kwargs required for
                    'metric'.

            early_stopping_args: None | dict, default=None
                Sets the Pytorch Lightning's EarlyStopping callback. If not
                None, a dictionary must be given with string keywords
                corresponding to an argument in EarlyStopping and the
                corresponding value. See:
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping

            ckpt_mode: str, default='min'
                Sets the condition for when a model checkpoint should be saved
                or overwritten during training.

            return_final: bool, default=False
                If True, checkpoint file is saved at the end of the last
                epoch. If False, checkpoint file is saved based on ckpt_mode.

            mute_logger: bool, default=False
                If True, the trainer will not log any metrics or save any
                checkpoints during training.

            seed: None | int, default=123
                If int, sets the random seed value for reproducibility. If
                None, a new random seed is used for each training run.

        """
        self.out_dir = out_dir
        self.mute_logger = mute_logger
        self.seed = seed
        self.early_stopping_args = early_stopping_args
        self.return_final = return_final
        self.ckpt_mode = ckpt_mode

        # seed everything
        if seed:
            seed_everything(seed, workers=True)

        # model setup kwargs
        self.pl_model_kwargs: dict[str, Any] = {
            "model": model,
            "model_params": model_params,
            "loss": loss_fn,
            "performance_metrics": performance_metrics,
            "optimizer": optim,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": lr_scheduler,
        }

        # PL trainer setup kwargs
        self.trainer_kwargs: dict[str, Any] = {
            "max_epochs": max_epochs,
            "detect_anomaly": True,
        }

        # device(s) to use
        self.trainer_kwargs["accelerator"] = device
        if device == "gpu":
            try:
                torch.set_float32_matmul_precision("high")
            except Warning:
                logger.warning(
                    """Your GPU does not have tensor cores. Internal
                    computations will proceed using default float32 datatype.
                    """,
                )

    @property
    def context(self) -> KITrainer:  # noqa: D102
        return self._context

    @context.setter
    def context(self, context: KITrainer) -> None:
        self._context = context

    @abstractmethod
    def fit_model(  # noqa: D102
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        pass

    @abstractmethod
    def evaluate_model(  # noqa: D102
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        pass

    @abstractmethod
    def _prepare_pl_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_pl_trainer(
        self,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def _save_model_state(self) -> ModelCheckpoint | None:
        pass
