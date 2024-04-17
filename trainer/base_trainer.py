"""
-----------
BaseTrainer
-----------

The "BaseTrainer" is an abstract class that functions as the interface bet-
ween the context class ``KITrainer'' and any of the concrete trainer state
objects.

The "BaseTrainer" class stores the user's parameters and defines a set of
abstract methods to be used by the trainer state objects.
"""  # noqa: INP001, D205, D212, D400, D415

from __future__ import annotations  # required for Python versions <3.9

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the abstract BaseTrainer class."

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import ModelCheckpoint

    from trainer.trainer import KITrainer

from abc import ABC, abstractmethod

import torch
from pytorch_lightning import seed_everything

from helpers.logger import get_logger

logger = get_logger()

class BaseTrainer(ABC):
    """Abstract class to interface between the context class "KITrainer" and a
    concrete state object.

    "BaseTrainer" will initialize necessary and optional kwargs to be used by
    any of the KnowIt Trainer states. It also defines abstract methods that are
    to be defined in each state object.

    Args:
    ----
        ABC (abc.ABC):          Used to define abstract class.

    """  # noqa: D205

    def __init__(
        self,
        model: type,
        model_params: dict[str, Any],
        out_dir: str,
        device: str,
        loss_fn: str | dict[str, Any],
        optim: str | dict[str, Any],
        max_epochs: int,
        learning_rate: float,
        lr_scheduler: None | dict[str, Any] = None,
        performance_metrics: None | str | dict[str, Any] = None,
        early_stopping_args: None | dict[str, Any] = None,
        ckpt_mode: str = "min",
        *,
        return_final: bool = False,
        mute_logger: bool = False,
        seed: None | int = 123,
    ) -> None:
        """BaseTrainer constructor.

        Args:
        ----
            model (type):           The Pytorch model architecture define by
                                    the user in Knowits ./archs subdirectory.

            model_params (dict):    The parameters required to initialize
                                    model.

            out_dir (str):          The directory to save the model's check-
                                    point file.

            device (str):           The device on which training is to be per-
                                    formed (cpu or gpu).

            loss_fn (str | dict):   The loss function to be used during train-
                                    ing. The string must match Pytorch's
                                    functional library. See:
                                    https://pytorch.org/docs/stable/nn.functional.html#loss-functions

            optim (str | dict):     The optimizer to be used during training.
                                    The string must match Pytorch's optimizer
                                    library. See:
                                    https://pytorch.org/docs/stable/nn.functional.html#loss-functions

            max_epochs (int):       The number of training iterations, where a
                                    single iteration is over the entire train-
                                    ing set.

            learning_rate (float):  The learning rate to be used during
                                    parameter updates. It controls the size of
                                    the updates.

            lr_scheduler (dict | None):
                                    The learning rate scheduler to be used
                                    during training. If not None, a dictionary
                                    must be given of the form
                                        {scheduler: scheduler_kwargs},
                                    where
                                        scheduler:      A string that specifies
                                                        the Pytorch scheduler
                                                        to be used. Must match
                                                        names found here:
                                                        https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler

                                        scheduler_
                                        kwargs:         A dictionary of kwargs
                                                        required for
                                                        'scheduler'.
                                    Default: None

            performance_metrics (str | dict | None):
                                    Performance metrics to be logged during
                                    training. If type=dict, then the dictionary
                                    must be given of the form
                                        {metric: metric_kwargs},
                                    where
                                        metric:         A string that specifies
                                                        the TORCHMETRICS metric
                                                        to be used. Must match
                                                        the functional inter-
                                                        face names found here:
                                                        https://lightning.ai/docs/torchmetrics/stable/

                                        metric_
                                        kwargs:         A dictionary of kwargs
                                                        required for 'metric'.
                                    Default: None.

            early_stopping_args (None | dict):
                                    Sets the Pytorch Lightning's EarlyStopping
                                    callback. If not None, a dictionary must be
                                    given with keywords corresponding to an
                                    argument in EarlyStopping and the corres-
                                    ponding value. See:
                                    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping
                                    Default: None

            ckpt_mode (str):        Sets the condition for when a model check-
                                    point should be saved or overwritten during
                                    training.
                                    Default: 'min'.

            return_final (bool):    If True, checkpoint file is saved at the
                                    end of the last epoch. If False, checkpoint
                                    file is saved based on ckpt_mode.
                                    Default: False.

            mute_logger (bool):     If True, the trainer will not log any
                                    metrics or save any checkpoints during
                                    training.
                                    Default: False.

            seed (None | int):      If int, sets the random seed value for
                                    reproducibility. If None, a new random seed
                                    is used for each training run.
                                    Default: 123.

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
    def fit_model(self, dataloaders: tuple[type, type, type]) -> None:
        """Fit model to the training data and monitor metrics on val set.

        Args:
        ----
            dataloaders (tuple):    The train dataloader and validation
                                    dataloader. The ordering of the tuple
                                    must be given is (train, val).

        """

    @abstractmethod
    def evaluate_model(self, dataloaders: tuple[type, type, type]) -> None:
        """Evaluate the trained model's performance on a tuple of data sets.

        NOTE: If the concatenated strings for metrics become long, Pytorch
        Lightning will print the evaluation results on two seperate lines in
        the terminal.

        Args:
        ----
            dataloaders (tuple):        A tuple consisting of three Pytorch
                                        dataloaders (train, val, eval).

        """

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
