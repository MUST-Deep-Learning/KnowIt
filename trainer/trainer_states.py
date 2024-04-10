
from typing import Callable, Literal, Tuple

from trainer.base_trainer import BaseTrainer

import torch
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from helpers.logger import get_logger
from trainer.model_config import PLModel


logger = get_logger()

class TrainNew(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prepare_pl_model()
        
        self.trainer = self._prepare_pl_trainer()    
    
    def prepare_pl_model(self):
       
        # build a PL model from arguments (untrained)
        self.pl_model = PLModel(**self.pl_model_kwargs)
    
    def fit_model(self, dataloaders):
        """Uses Pytorch Lightning to fit the model to the train data

        Args:
        ----
            dataloaders (tuple): The train dataloader and validation dataloader. The ordering of the tuple is (train, val).

        """
        train_dataloader = dataloaders[0]
        val_dataloader = dataloaders[1]

        # fit trainer object to data
        
        self.trainer.fit(
            model=self.pl_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate_model(self, dataloaders):
        """Evaluates the model's performance on a evaluation set.

        NOTE If the concatenated strings for metrics become long, Pytorch Lightning will print
        the evaluation results on two seperate lines in the terminal.

        Args:
        ----
            eval_dataloader (Pytorch dataloader)    : The evaluation dataloader.

        """
        # the path to the best model ckpt or the last model ckpt
        if self.return_final:
            set_ckpt_path = self.out_dir + "/last.ckpt"
        else:
            set_ckpt_path = "best"

        
        logger.info(
            "Testing model on the current training run's best checkpoint."
        )
        self.trainer.test(ckpt_path=set_ckpt_path, dataloaders=dataloaders)

    def _prepare_pl_trainer(
        self,
    ) -> type:
        """Calls Pytorch Lightning's trainer using the user's parameters."""

        # training logger - save results in current model's folder
        if self.mute_logger:
            self.trainer_kwargs["logger"] = None
            self.trainer_kwargs["default_root_dir"] = None
        else:
            ckpt_callback = self._save_model_state()
            self.trainer_kwargs["default_root_dir"] = self.out_dir
            self.trainer_kwargs["logger"] = pl_loggers.CSVLogger(
                save_dir=self.out_dir,
            )

        # Early stopping
        if isinstance(self.early_stopping_args, dict):
            try:
                early_stopping = EarlyStopping(**self.early_stopping_args)
                logger.info("Early stopping is enabled.")
            except Warning:
                logger.warning(
                    "Unable to add Early Stopping. If Early Stopping should be enabled, it must be passed as a dict with kwargs."
                )
                early_stopping = None
        else:
            early_stopping = None

        callbacks = [c for c in [ckpt_callback, early_stopping] if c != None]
        self.trainer_kwargs["callbacks"] = callbacks

        # set seed
        if self.seed:
            seed_everything(self.seed, workers=True)

        # Pytorch Lightning trainer object
        trainer = PLTrainer(
            **self.trainer_kwargs,
        )

        return trainer
    
    def _save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.

        """
        # determine if last model or best model needs to be returned
        if self.return_final:
            set_top_k = 0
            set_save_last = True
        else:
            set_top_k = 1
            set_save_last = False

        to_monitor = "valid_loss"
        if self.pl_model_kwargs["performance_metrics"]:
            try:
                met = list(self.pl_model_kwargs["performance_matrics"].keys())
                met = met[0]
                to_monitor = "valid_perf_" + met
            except:
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
    def __init__(self, ckpt_file, **kwargs):
        super().__init__(**kwargs)
        
        self.ckpt_file = ckpt_file
        
        self.prepare_pl_model(
            to_ckpt=ckpt_file,
        )
        
        self.trainer = self._prepare_pl_trainer()
    
    def prepare_pl_model(self, to_ckpt):
        
        
        self.pl_model = PLModel.load_from_checkpoint(
            checkpoint_path=to_ckpt,
        )
    
    def fit_model(self, dataloaders):
        """Uses Pytorch Lightning to fit the model to the train data

        Args:
        ----
            dataloaders (tuple): The train dataloader and validation dataloader. The ordering of the tuple is (train, val).

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

    def evaluate_model(self, dataloaders):
        """Evaluates the model's performance on a evaluation set.

        NOTE If the concatenated strings for metrics become long, Pytorch Lightning will print
        the evaluation results on two seperate lines in the terminal.

        Args:
        ----
            eval_dataloader (Pytorch dataloader)    : The evaluation dataloader.

        """
        # the path to the best model ckpt or the last model ckpt
        if self.return_final:
            set_ckpt_path = self.out_dir + "/last.ckpt"
        else:
            set_ckpt_path = "best"

        
        logger.info(
            "Testing model on the current training run's best checkpoint."
        )
        self.trainer.test(ckpt_path=set_ckpt_path, dataloaders=dataloaders)

            
    def _prepare_pl_trainer(
        self,
    ) -> type:
        """Calls Pytorch Lightning's trainer using the user's parameters."""

        # training logger - save results in current model's folder
        if self.mute_logger:
            self.trainer_kwargs["logger"] = None
            self.trainer_kwargs["default_root_dir"] = None
        else:
            ckpt_callback = self._save_model_state()
            self.trainer_kwargs["default_root_dir"] = self.out_dir
            self.trainer_kwargs["logger"] = pl_loggers.CSVLogger(
                save_dir=self.out_dir,
            )

        # Early stopping
        if isinstance(self.early_stopping_args, dict):
            try:
                early_stopping = EarlyStopping(**self.early_stopping_args)
                logger.info("Early stopping is enabled.")
            except Warning:
                logger.warning(
                    "Unable to add Early Stopping. If Early Stopping should be enabled, it must be passed as a dict with kwargs."
                )
                early_stopping = None
        else:
            early_stopping = None

        callbacks = [c for c in [ckpt_callback, early_stopping] if c != None]
        self.trainer_kwargs["callbacks"] = callbacks

        # set seed
        if self.seed:
            seed_everything(self.seed, workers=True)

        
        trainer = PLTrainer(
            **self.trainer_kwargs
        )
        
        return trainer
    
    def _save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.

        """
        # determine if last model or best model needs to be returned
        if self.return_final:
            set_top_k = 0
            set_save_last = True
        else:
            set_top_k = 1
            set_save_last = False

        to_monitor = "valid_loss"
        if self.pl_model_kwargs["performance_metrics"]:
            try:
                met = list(self.pl_model_kwargs["performance_matrics"].keys())
                met = met[0]
                to_monitor = "valid_perf_" + met
            except:
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
    def __init__(self, ckpt_file):
        
        self.ckpt_file = ckpt_file
        
        self.prepare_pl_model(
            to_ckpt=ckpt_file,
        )
        
        self.trainer = self._prepare_pl_trainer()
    
    def prepare_pl_model(self, to_ckpt):

        self.pl_model = PLModel.load_from_checkpoint(
            checkpoint_path=to_ckpt,
        )
    
    def fit_model(self, dataloaders):
        """Uses Pytorch Lightning to fit the model to the train data

        Args:
        ----
            dataloaders (tuple): The train dataloader and validation dataloader. The ordering of the tuple is (train, val).

        """
        pass

    def evaluate_model(self, dataloaders):
        """Evaluates the model's performance on a evaluation set.

        NOTE If the concatenated strings for metrics become long, Pytorch Lightning will print
        the evaluation results on two seperate lines in the terminal.

        Args:
        ----
            eval_dataloader (Pytorch dataloader)    : The evaluation dataloader.

        """
            
        
        logger.info("Testing on model loaded from checkpoint.")
        self.trainer.test(model=self.pl_model, dataloaders=dataloaders)
            
    def _prepare_pl_trainer(
        self,
    ) -> type:
        """Calls Pytorch Lightning's trainer using the user's parameters."""
        
        trainer = PLTrainer()

        return trainer
    
    def _save_model_state(self):
        """Saves the best model to the user's project output directory as a checkpoint.
        Files are named as datetime strings.

        """
        pass



























