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
"""

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

    Parameters
    ----------
    state : type[Any]
        A concrete state that corresponds to one of the possible states for
        the trainer.

    base_trainer_kwargs : dict[str, Any]
        The kwargs required in the BaseTrainer submodule.

    optional_pl_kwargs : dict[str, Any]
        An additional kwargs that a user would like to provide Pytorch
        Lightning's Trainer.

    ckpt_file : None | str, default=None
        A string that points to a Pytorch checkpoint file. Required for
        certain trainer states.

    train_flag: str, default='train'
        An additional flag to indicate whether the Trainer is in a train
        state or an evaluate only state.

    Attributes
    ----------
    _state: None | type[Any], default=None
        The current state that the Trainer is initialized in.
    """

    _state: None | type[Any] = None

    def __init__(
        self,
        state: type[Any],
        base_trainer_kwargs: dict[str, Any],
        optional_pl_kwargs: dict[str, Any],
        ckpt_file: None | str = None,
        *,
        train_flag: str = "train",
    ) -> None:
        self._set_state(
            state=state,
            base_trainer_kwargs=base_trainer_kwargs,
            optional_pl_kwargs=optional_pl_kwargs,
            ckpt_file=ckpt_file,
            train_flag=train_flag,
        )

    def _set_state(
        self,
        state: type[Any],
        base_trainer_kwargs: dict[str, Any],
        ckpt_file: None | str,
        optional_pl_kwargs: dict[str, Any],
        *,
        train_flag: str,
    ) -> None:
        if train_flag == "train":
            # train from scratch
            self._state = state(
                base_kwargs=base_trainer_kwargs,
                optional_pl_kwargs=optional_pl_kwargs,
            )
        elif train_flag == "train_from_ckpt":
            # train from checkpoint
            self._state = state(
                to_ckpt=ckpt_file,
                base_kwargs=base_trainer_kwargs,
                optional_pl_kwargs=optional_pl_kwargs,
            )
        elif train_flag == "evaluate_only":
            # evaluate model
            self._state = state(
                to_ckpt=ckpt_file,
                base_trainer_kwargs=base_trainer_kwargs,
            )

        if not self._state:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)

        self._state.context = self

    def fit(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Fit model to training data.

        Parameters
        ----------
        dataloaders tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The Pytorch dataloaders that has been set up in KnowIt's
            datamodule. The triplet corresponds to the train, val, and eval
            dataloaders.

        """
        if self._state is None:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)

        self._state.fit_model(dataloaders=dataloaders)

    def evaluate_fitted_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Evaluate a trained model from checkpoint on a user's data.

        Parameters
        ----------
        dataloaders tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The Pytorch dataloaders that has been set up in KnowIt's
            datamodule. The triplet corresponds to the train, val, and eval
            dataloaders.

        """
        if self._state is None:
            emsg = "Trainer state cannot be set to None."
            raise TypeError(emsg)

        self._state.evaluate_model(dataloaders=dataloaders)
