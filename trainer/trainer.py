"""
---------------
KITrainer
---------------

The "KITrainer" is a context class that that interacts with the overall
Knowit architecture script (which, in turn, interacts with all of KnowIt's sub-
modules).

The user is able to use the KnowIt Trainer in different ways, depending on the
training task. As such, the trainer submodule is built similar to a `State'
pattern.

Thus, there are three parts to the submodule: the context class (KITrainer)
that Knowit directly interacts with, an abstract class (BaseTrainer) that
interfaces the context class with a concrete trainer state, and a set of
trainer state classes.

The three possible concrete states are:
    - STATE 1 (NEW): Train a new model from scratch.
    - STATE 2 (CONTINUE): Continue training an existing model from checkpoint.
    - STATE 3 (EVAL): Load a trained model and evaluate it on a dataset.

KnowIt's Trainer module is built using Pytorch Lightning. See here:
https://lightning.ai/pytorch-lightning
"""  # noqa: INP001, D205, D212, D400, D415

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the KITrainer context class."

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data.dataloader import DataLoader

from helpers.logger import get_logger

logger = get_logger()


class KITrainer:
    """Context class that interacts with KnowIt.

    The class interacts with the overall KnowIt architecture script. Based on
    the user's training task, it will point to the correct Trainer state.

    Args:
    ----
        _state (None | type):   A concrete state object that corresponds to one
                                of the possible states for the trainer.

        ckpt_file (None|str):   A string that points to a Pytorch checkpoint
                                file. Required for certain trainer states.
                                default: None

    Kwargs:
    ------
        **kwargs (any):         Kwargs required to initialize the BaseTrainer
                                abstract class.

    """

    _state: None | type[Any] = None

    def __init__(
        self,
        state: type[Any],
        base_trainer_kwargs: dict[str, Any],
        optional_pl_kwargs: dict[str, Any],
        ckpt_file: None | str = None,
    ) -> None:
        self._set_state(
            state=state,
            base_trainer_kwargs=base_trainer_kwargs,
            optional_pl_kwargs=optional_pl_kwargs,
            ckpt_file=ckpt_file,
        )

    def _set_state(
        self,
        state: type[Any],
        base_trainer_kwargs: dict[str, Any],
        ckpt_file: None | str,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        if ckpt_file:
            self._state = state(
                to_ckpt=ckpt_file,
                base_kwargs=base_trainer_kwargs,
                optional_pl_kwargs=optional_pl_kwargs,
            )
        else:
            self._state = state(
                base_kwargs=base_trainer_kwargs,
                optional_pl_kwargs=optional_pl_kwargs,
            )

        if not self._state:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)
        
        self._state.context = self

    def fit_and_eval(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Fit model to training data and evaluate on eval data.

        Args:
        ----
            dataloaders (tuple):    The Pytorch dataloaders that has been set
                                    up in KnowIt's datamodule.

        """
        if self._state is None:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)

        self._state.fit_model(dataloaders=dataloaders)
        self._state.evaluate_model(dataloaders=dataloaders)

    def eval(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Evaluate a trained model from checkpoint on a user's data.

        Args:
        ----
            dataloaders (tuple):    The Pytorch dataloaders that has been set
                                    up in KnowIt's datamodule.

        """
        if self._state is None:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)

        self._state.evaluate_model(dataloaders=dataloaders)
