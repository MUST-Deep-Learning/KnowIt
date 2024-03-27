__author__ = "randlerabe@gmail.com"
__description__ = (
    "Contains the base class that prepares the Pytorch Lightning trainer."
)

from typing import Callable, Literal, Optional, Tuple, Union

import torch
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from helpers.logger import get_logger

logger = get_logger()


class BaseTrainer:
    """A wrapper class that handles user parameters and directs the instructions
    to Pytorch Lightning.

    Args (compulsary)
    - experiment_name: str.                                The name of the experiment.
    - train_device: str.                                   The choice of training device ('cpu', 'gpu', etc).
    - loss_fn: str or dict.                                The choice of loss function (see Pytorch's torch.nn.functional documentation)
    - optim: str or dict:                                  The choice of optimizer (see Pytorch's torch.nn.optim documentation)
    - max_epochs: int.                                     The number of epochs that the model should train on.
    - learning_rate: float.                                The learning rate that the chosen optimizer should use.
    - model: Class.                                        The Pytorch model architecture define by the user in Knowits ./archs subdir.
    - model_params: dict.                                  A dictionary with values needed to init the above Pytorch model.

    Args (optional)
    - learning_rate_scheduler: dict. Default: {}.          A dictionary that specifies the learning rate scheduler
                                                            and any needed kwargs.
    - performance_metrics: None or dict. Default: None.    Specifies any performance metrics on the validation
                                                            set during training.
    - early_stopping: bool or dict. Default: False.        Specifies early stopping conditions.
    - gradient_clip_val: float: Default: 0.0.              Clips exploding gradients according to the chosen
                                                            gradient_clip_algorithm.
    - gradient_clip_algorithm: str. Default: 'norm'.       Specifies how the gradient_clip_val should be applied.
    - set_seed: int or bool. Default: False.               A global seed applied by Pytorch Lightning for reproducibility.
    - deterministic: bool, str, or None. Default: None.    Pytorch Lightning attempts to further reduce randomness
                                                            during training. This may incur a performance hit.
    - safe_mode: bool. Default: False.                     If set to True, aborts the model training if the experiment name already
                                                            exists in the user's project output folder.
    """

    def __init__(
        self,
        out_dir: str,
        train_device: str,
        loss_fn: Union[str, dict],
        optim: Union[str, dict],
        max_epochs: int,
        learning_rate: float,
        learning_rate_scheduler: dict = {},
        performance_metrics: Optional[dict] = None,
        early_stopping: Union[bool, dict] = False,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
        train_precision: str = "32-true",
        set_seed: Union[int, bool] = False,
        deterministic: Union[bool, Literal["warn"], None] = None,
        num_devices: Union[str, int] = "auto",
        return_final: bool = False,
        mute_logger: bool = False,
        model_selection_mode: str = "min",
    ):
        # set output directory
        self.out_dir = out_dir

        # turn off logger during hp tuning
        self.mute_logger = mute_logger

        # save global seed
        self.set_seed = set_seed

        # device(s) to use
        self.train_device = train_device
        self.num_devices = num_devices
        if train_device == "gpu":
            try:
                torch.set_float32_matmul_precision("high")
            except:
                logger.warning(
                    "Tried to utilize Tensor Cores, but none found."
                )

        # model hyperparameters
        self.loss_fn = loss_fn
        self.optim = optim
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.performance_metrics = performance_metrics
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # misc
        self.return_final = return_final
        self.model_selection_mode = model_selection_mode
        self.deterministic = deterministic
        self.precision = train_precision

    def _build_PL_trainer(
        self, state: int, save_state: Callable[[None], Tuple[str, type]]
    ) -> type:
        """Calls Pytorch Lightning's trainer using the user's parameters."""

        # training logger - save results in current model's folder
        if self.mute_logger:
            csv_logger = None
            ckpt_path, ckpt_callback = None, None
        else:
            ckpt_path, ckpt_callback = save_state()
            csv_logger = pl_loggers.CSVLogger(save_dir=ckpt_path)

        # Early stopping
        try:
            early_stopping = EarlyStopping(**self.early_stopping[True])
            logger.info("Early stopping is enabled.")
        except:
            logger.info(
                "Early stopping is not enabled. If Early Stopping should be enabled, it must be passed as a dict with kwargs."
            )
            early_stopping = None

        callbacks = [c for c in [ckpt_callback, early_stopping] if c != None]

        # set seed
        if self.set_seed:
            seed_everything(self.set_seed, workers=True)

        # Pytorch Lightning trainer object
        # if train_flag == True and from_ckpt_flag == False:
        if state == 1:
            trainer = PLTrainer(
                max_epochs=self.max_epochs,
                accelerator=self.train_device,
                logger=csv_logger,
                devices=self.num_devices,
                callbacks=callbacks,
                detect_anomaly=True,
                precision=self.precision,
                default_root_dir=ckpt_path,
                deterministic=self.deterministic,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        # elif train_flag == True and from_ckpt_flag == True:
        elif state == 2:
            trainer = PLTrainer(
                max_epochs=self.max_epochs,
                default_root_dir=ckpt_path,
                callbacks=callbacks,
                detect_anomaly=True,
                logger=csv_logger,
            )

        return trainer
