"""
-------------
TrainerStates
-------------

There are three states that a user can initialize KnowIt's Trainer in. They are
as follows:

STATE 1 (NEW): Train a new model from scratch.

    Class: TrainNew
        Fits a model to a training set and evaluates the model on a valid-
        ation set and evaluation set. During training, metrics are logged
        and a checkpoint file is saved (only if 'mute_logger' is False) in
        the user's model output directory.

STATE 2 (CONTINUE): Continue training an existing model from checkpoint.

    Class: ContinueTraining
        Initializes a pretrained model from a checkpoint file. Fits the
        initialized model to a training set and evaluates the model on a
        validation set and evaluation set. During training, metrics are
        logged and a checkpoint file is saved (only if 'mute_logger' is
        False) in the user's model output directory.

STATE 3 (EVAL): Continue training an existing model from checkpoint.

    Class: EvaluateOnly
        Initializes a pretrained model from a checkpoint file. Evaluates
        the model on a validation set and evaluation set.

In the case that the above states are inadequate for a user's task, the module
also contains an example template class "CustomTrainer" that a user can edit.

"""  # noqa: D205, D400

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import Trainer as PLTrainer  
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from helpers.logger import get_logger
from trainer.base_trainer import BaseTrainer
from trainer.model_config import PLModel

from pytorch_lightning.loggers import WandbLogger

logger = get_logger()


class TrainNew(BaseTrainer):
    """Fit a model to a training dataset and evaluate on val/eval sets.

    Parameters
    ----------
    base_kwargs : dict[str, Any]
        The user's input parameters (to be stored in the parent class).

    optional_pl_kwargs : dict[str, Any]
        Additional kwargs to be provided to Pytorch Lightning's Trainer (such
        as gradient clipping, etc). See Pytorch Lightning's documentation for
        more information.

    Attributes
    ----------
    pl_model : type
        The Pytorch Lightning model initialized with a user's Pytorch model.

    trainer : type
        The Pytorch Lightning trainer initialized with pl_model and any
        additional user kwargs (see optional_pl_kwargs).
    """

    def __init__(
        self,
        base_kwargs: dict[str, Any],
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**base_kwargs)

        self._prepare_pl_model()

        self._prepare_pl_trainer(optional_pl_kwargs=optional_pl_kwargs)

    def fit_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Fit model to the training data and monitor metrics on val set.

        Parameters
        ----------
        dataloaders : tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The triplet containing the train dataloader and validation
            dataloader. The ordering of the tuple must be given as
            (train, val, eval).

        """
        train_dataloader = dataloaders[0]
        val_dataloader = dataloaders[1]

        self.trainer.fit(
            model=self.pl_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Evaluate the trained model's performance on a tuple of data sets.

        Parameters
        ----------
        dataloaders : tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The triplet containing the train dataloader and validation
            dataloader. The ordering of the tuple must be given as
            (train, val, eval).

        NOTE: If the concatenated strings for metrics become long, Pytorch
        Lightning will print the evaluation results on two seperate lines in
        the terminal.

        """
        if self.return_final:
            set_ckpt_path = self.out_dir + "/last.ckpt"
        else:
            set_ckpt_path = "best"

        logger.info(
            "Testing model on the current training run's best checkpoint.",
        )
        self.trainer.test(ckpt_path=set_ckpt_path, dataloaders=dataloaders)

    def _prepare_pl_model(self) -> None:
        self.pl_model = PLModel(**self.pl_model_kwargs)

    def _prepare_pl_trainer(
        self,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        # training logger - save results in current model's folder
        if self.logger_status == "off":
            self.trainer_kwargs["logger"] = None
            self.trainer_kwargs["default_root_dir"] = None
            ckpt_callback = None
        elif self.logger_status == "w&b_only":
            self.trainer_kwargs["logger"] = WandbLogger(log_model=False)
            self.trainer_kwargs["default_root_dir"] = None
            ckpt_callback = None
        elif self.logger_status == "w&b_on":
            ckpt_callback = self._save_model_state()
            self.trainer_kwargs["default_root_dir"] = self.out_dir
            self.trainer_kwargs["logger"] = [
                pl_loggers.CSVLogger(save_dir=self.out_dir),
                WandbLogger(log_model=False),
            ]
        elif not self.logger_status:
            ckpt_callback = self._save_model_state()
            self.trainer_kwargs["default_root_dir"] = self.out_dir
            self.trainer_kwargs["logger"] = [
                pl_loggers.CSVLogger(save_dir=self.out_dir),
            ]

        # set up EarlyStopping if enabled
        if isinstance(self.early_stopping_args, dict):
            try:
                early_stopping = EarlyStopping(**self.early_stopping_args)
                logger.info("Early stopping is enabled.")
            except Warning:
                logger.warning(
                    "Unable to add Early Stopping. If Early Stopping should be\
                        enabled, it must be passed as a dict with kwargs.",
                )
                early_stopping = None
        else:
            early_stopping = None

        callbacks = [
            c for c in [ckpt_callback, early_stopping] if c is not None
        ]
        self.trainer_kwargs["callbacks"] = callbacks

        self.trainer = PLTrainer(
            **self.trainer_kwargs,
            **optional_pl_kwargs,
        )

    def _save_model_state(self) -> ModelCheckpoint:
        # determine if last model or best model needs to be returned
        if self.return_final:
            set_top_k = 0
            set_save_last = True
        else:
            set_top_k = 1
            set_save_last = False

        to_monitor = "valid_loss"
        if isinstance(self.pl_model_kwargs["performance_metrics"], dict):
            met = list(self.pl_model_kwargs["performance_metrics"].keys())
            met = met[0]
            to_monitor = "valid_perf_" + met
        elif isinstance(self.pl_model_kwargs["performance_metrics"], str):
            to_monitor = (
                "valid_perf_" + self.pl_model_kwargs["performance_metrics"]
            )

        return ModelCheckpoint(
            dirpath=self.out_dir,
            monitor=to_monitor,
            filename="bestmodel-{epoch}-{" + to_monitor + ":.2f}",
            save_top_k=set_top_k,
            save_last=set_save_last,
            mode=self.ckpt_mode,
        )


class ContinueTraining(BaseTrainer):
    """Fit a pretrained model to a training set and evaluate on val/eval sets.

    Parameters
    ----------
     to_ckpt : None | str
        Path to the model checkpoint file.

    base_kwargs : dict[str, Any]
        The user's input parameters (to be stored in the parent class).

    optional_pl_kwargs : dict[str, Any]
        Additional kwargs to be provided to Pytorch Lightning's Trainer (such
        as gradient clipping, etc). See Pytorch Lightning's documentation for
        more information.

    Attributes
    ----------
    pl_model : type
        The Pytorch Lightning model initialized with a user's Pytorch model.

    trainer : type
        The Pytorch Lightning trainer initialized with pl_model and any
        additional user kwargs (see optional_pl_kwargs).
    """

    def __init__(
        self,
        to_ckpt: str,
        base_kwargs: dict[str, Any],
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**base_kwargs)
        self.ckpt_file = to_ckpt

        self._prepare_pl_model()

        self._prepare_pl_trainer(optional_pl_kwargs=optional_pl_kwargs)

    def fit_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Fit model to the training data and monitor metrics on val set.

        Parameters
        ----------
        dataloaders : tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The triplet containing the train dataloader and validation
            dataloader. The ordering of the tuple must be given as
            (train, val, eval).

        """
        train_dataloader = dataloaders[0]
        val_dataloader = dataloaders[1]

        logger.info("Resuming model training from checkpoint.")
        self.trainer.fit(
            model=self.pl_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=self.ckpt_file,
        )

    def evaluate_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Evaluate the trained model's performance on a tuple of data sets.

        Parameters
        ----------
        dataloaders : tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The triplet containing the train dataloader and validation
            dataloader. The ordering of the tuple must be given as
            (train, val, eval).

        NOTE: If the concatenated strings for metrics become long, Pytorch
        Lightning will print the evaluation results on two seperate lines in
        the terminal.

        """
        if self.return_final:
            set_ckpt_path = self.out_dir + "/last.ckpt"
        else:
            set_ckpt_path = "best"

        logger.info(
            "Testing model on the current training run's best checkpoint.",
        )
        self.trainer.test(ckpt_path=set_ckpt_path, dataloaders=dataloaders)

    def _prepare_pl_model(self) -> None:
        self.pl_model = PLModel.load_from_checkpoint(  # type: ignore  # noqa: PGH003
            checkpoint_path=self.ckpt_file,
        )

    def _prepare_pl_trainer(
        self,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        # training logger - save results in current model's folder
        if self.logger_status:
            self.trainer_kwargs["logger"] = None
            self.trainer_kwargs["default_root_dir"] = None
            ckpt_callback = None
        else:
            ckpt_callback = self._save_model_state()
            self.trainer_kwargs["default_root_dir"] = self.out_dir
            self.trainer_kwargs["logger"] = [
                pl_loggers.CSVLogger(save_dir=self.out_dir),
                WandbLogger(),
            ]

        # set up EarlyStopping if enabled
        if isinstance(self.early_stopping_args, dict):
            try:
                early_stopping = EarlyStopping(**self.early_stopping_args)
                logger.info("Early stopping is enabled.")
            except Warning:
                logger.warning(
                    "Unable to add Early Stopping. If Early Stopping should be\
                        enabled, it must be passed as a dict with kwargs.",
                )
                early_stopping = None
        else:
            early_stopping = None

        callbacks = [
            c for c in [ckpt_callback, early_stopping] if c is not None
        ]
        self.trainer_kwargs["callbacks"] = callbacks

        self.trainer = PLTrainer(
            **self.trainer_kwargs,
            **optional_pl_kwargs,
        )

    def _save_model_state(self) -> ModelCheckpoint:
        # determine if last model or best model needs to be returned
        if self.return_final:
            set_top_k = 0
            set_save_last = True
        else:
            set_top_k = 1
            set_save_last = False

        to_monitor = "valid_loss"
        if isinstance(self.pl_model_kwargs["performance_metrics"], dict):
            met = list(self.pl_model_kwargs["performance_metrics"].keys())
            met = met[0]
            to_monitor = "valid_perf_" + met
        elif isinstance(self.pl_model_kwargs["performance_metrics"], str):
            to_monitor = (
                "valid_perf_" + self.pl_model_kwargs["performance_metrics"]
            )

        return ModelCheckpoint(
            dirpath=self.out_dir,
            monitor=to_monitor,
            filename="bestmodel-{epoch}-{" + to_monitor + ":.2f}",
            save_top_k=set_top_k,
            save_last=set_save_last,
            mode=self.ckpt_mode,
        )


class EvaluateOnly(BaseTrainer):
    """Evaluate a trained model on a dataset.

    Parameters
    ----------
     to_ckpt : None | str
        Path to the model checkpoint file.

    base_trainer_kwargs: dict
        A dictionary to initialize Pytorch Lightning's Trainer module.

    Attributes
    ----------
    ckpt_file: str
        Path to model checkpoint file.

    pl_model : type
        The Pytorch Lightning model initialized with a user's Pytorch model.

    trainer : type
        The Pytorch Lightning trainer initialized with pl_model and any addit-
        ional kwargs.
    """

    def __init__(
        self,
        to_ckpt: str,
        base_trainer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**base_trainer_kwargs)
        self.ckpt_file = to_ckpt

        self._prepare_pl_model()

        self._prepare_pl_trainer(optional_pl_kwargs={})

    def fit_model(  # noqa: D102
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        pass

    def evaluate_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        """Evaluate the trained model's performance on a tuple of data sets.

        Parameters
        ----------
        dataloaders : tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]
            The triplet containing the train dataloader and validation
            dataloader. The ordering of the tuple must be given as
            (train, val, eval).

        NOTE: If the concatenated strings for metrics become long, Pytorch
        Lightning will print the evaluation results on two seperate lines in
        the terminal.

        """
        logger.info("Testing on model loaded from checkpoint.")
        self.trainer.test(model=self.pl_model, dataloaders=dataloaders)

    def _prepare_pl_model(self) -> None:
        self.pl_model = PLModel.load_from_checkpoint(  # type: ignore  # noqa: PGH003
            checkpoint_path=self.ckpt_file,
            **self.pl_model_kwargs,
        )

    def _prepare_pl_trainer(
        self,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:

        self.trainer_kwargs["default_root_dir"] = self.out_dir
        self.trainer_kwargs["logger"] = [
            pl_loggers.CSVLogger(save_dir=self.out_dir, version="evaluation"),
        ]
        self.trainer = PLTrainer(
            **self.trainer_kwargs,
            **optional_pl_kwargs, # optional_pl_kwargs={}
        )

    def _save_model_state(self) -> None:
        pass


class CustomTrainer(BaseTrainer):
    """A template KnowIt trainer state.

    The template can be edited by a user to create a custom trainer state.

    Parameters
    ----------
    base_kwargs : dict[str, Any]
        The user's input parameters (to be stored in the parent class).

    optional_pl_kwargs : dict[str, Any]
        Additional kwargs to be provided to Pytorch Lightning's Trainer (such
        as gradient clipping, etc). See Pytorch Lightning's documentation for
        more information.

    """

    def __init__(
        self,
        base_kwargs: dict[str, Any],
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        # configure
        super().__init__(**base_kwargs)

        self._prepare_pl_model()

        self._prepare_pl_trainer(optional_pl_kwargs=optional_pl_kwargs)

    def fit_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        # configure
        self.trainer.fit(
            model=self.pl_model,
            train_dataloaders=dataloaders[0],
            val_dataloaders=dataloaders[1],
        )

    def evaluate_model(
        self,
        dataloaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    ) -> None:
        # configure
        self.trainer.test(ckpt_path="best", dataloaders=dataloaders)

    def _prepare_pl_model(self) -> None:
        # configure
        self.pl_model = PLModel(**self.pl_model_kwargs)

    def _prepare_pl_trainer(
        self,
        optional_pl_kwargs: dict[str, Any],
    ) -> None:
        # configure
        self.trainer = PLTrainer(
            **self.trainer_kwargs,
            **optional_pl_kwargs,
        )

    def custom_method(self):
        # configure
        pass

    def _save_model_state(self) -> ModelCheckpoint:
        # configure
        return ModelCheckpoint(
            dirpath=self.out_dir,
        )
