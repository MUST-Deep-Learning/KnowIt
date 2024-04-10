"""-----BaseTrainer-----.

The ``BaseTrainer'' class is the parent (root) class that is to be
inherited by KITrainer.

The function of the ``BaseTrainer'' class is to store the user's parameters and
appropriately prepares Pytorch Lightning's trainer module based on the user's
needs.

To complete!!!

"""

from __future__ import annotations  # required for Python versions <3.9
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.trainer import KITrainer

__author__ = "randlerabe@gmail.com"
__description__ = (
    "Contains the base class that prepares the Pytorch Lightning trainer."
)

from abc import ABC, abstractmethod

from typing import Callable, Literal, Tuple

import torch
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from helpers.logger import get_logger
from trainer.model_config import PLModel

logger = get_logger()


class BaseTrainer(ABC):


    def __init__(
        self,
        model: type,
        model_params: dict,
        out_dir: str,
        device: str,
        loss_fn: str | dict,
        optim: str | dict,
        max_epochs: int,
        learning_rate: float,
        lr_scheduler: dict | None = None,
        performance_metrics: dict | None = None,
        clip_gradients: dict | None = None,
        *,
        return_final: bool = False,
        mute_logger: bool = False,
        seed: int | bool = False,
        early_stopping_args: None | dict = None,
        deterministic: bool | Literal["warn"] | None = None,
        ckpt_mode: str = "min",
    ) -> None:
        # set output directory
        self.out_dir = out_dir

        # turn off logger during hp tuning
        self.mute_logger = mute_logger

        # save global seed
        self.seed = seed

        # model kwargs
        self.pl_model_kwargs = {
            "model": model,
            "model_params": model_params,
            "loss": loss_fn,
            "performance_metrics": performance_metrics,
            "optimizer": optim,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": lr_scheduler,
        }

        self.early_stopping_args = early_stopping_args
        self.trainer_kwargs = {
            "max_epochs": max_epochs,
            "deterministic": deterministic,
            "detect_anomaly": True,
        }

        if clip_gradients:
            self.trainer_kwargs["gradient_clip_val"] = clip_gradients["value"]
            self.trainer_kwargs["gradient_clip_algorithm"] = clip_gradients[
                "algorithm"
            ]
        else:
            self.trainer_kwargs["gradient_clip_val"] = None
            self.trainer_kwargs["gradient_clip_algorithm"] = "norm"

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

        # misc
        self.return_final = return_final
        self.ckpt_mode = ckpt_mode

    @property
    def context(self) -> KITrainer:
        return self._context

    @context.setter
    def context(self, context: KITrainer) -> None:
        self._context = context
        
    @abstractmethod
    def fit_model(self, dataloaders):
        pass
    
    @abstractmethod
    def evaluate_model(self, dataloaders):
        pass
    
    @abstractmethod
    def prepare_pl_model(self, to_ckpt):
        pass
    
    @abstractmethod
    def _prepare_pl_trainer(
        self,
    ) -> type:
        """Calls Pytorch Lightning's trainer using the user's parameters."""

        pass
    
    @abstractmethod
    def _save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.

        """
        pass
