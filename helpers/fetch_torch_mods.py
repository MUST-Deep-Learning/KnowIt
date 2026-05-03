from __future__ import annotations  # noqa: D100
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com"
__description__ = "Helper functions used when loading and configuring Pytorch modules."

# standard library imports
import os

# external imports
import torch
from pandas import read_csv

# internal imports
from helpers.logger import get_logger

logger = get_logger()


def get_optim(optimizer: str) -> type[torch.optim.Optimizer]:
    """Return the user's choice of optimizer class.

    Parameters
    ----------
    optimizer : str
        The name of the optimizer as specified in ``torch.optim``
        (e.g., ``"Adam"``, ``"SGD"``).

    Returns
    -------
    type[torch.optim.Optimizer]
        The uninitialised PyTorch optimizer class.
    """
    return getattr(torch.optim, optimizer)


def get_lr_scheduler(
    scheduler: str,
) -> type[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Return the user's choice of learning-rate scheduler class.

    Parameters
    ----------
    scheduler : str
        The name of the scheduler as specified in
        ``torch.optim.lr_scheduler``
        (e.g., ``"StepLR"``, ``"CosineAnnealingLR"``).

    Returns
    -------
    type[torch.optim.lr_scheduler.LRScheduler] | type[torch.optim.lr_scheduler.ReduceLROnPlateau]
        The uninitialised scheduler class.
    """
    return getattr(torch.optim.lr_scheduler, scheduler)


def get_model_score(model_dir: str) -> tuple:
    """Scan a directory for a checkpoint, load it, and extract the best score.

    If no best score is found in the checkpoint, the function assumes that the
    final-epoch model was saved and retrieves the score from the
    ``/lightning_logs/version_0/metrics.csv`` file.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the model checkpoint file.

    Returns
    -------
    tuple
        A ``(best_model_score, monitor, epoch)`` tuple where:

        - ``best_model_score`` (*float*) — the best score achieved.
        - ``monitor`` (*str*) — the monitored metric name.
        - ``epoch`` (*int*) — the epoch at which the best score was achieved.

    Notes
    -----
    Assumes exactly one ``.ckpt`` file exists in ``model_dir``.
    """
    ckpt_name = [f for f in os.listdir(model_dir) if "ckpt" in f]
    ckpt_name = next(iter(ckpt_name))
    ckpt_path = os.path.join(model_dir, ckpt_name)
    ckpt = torch.load(f=ckpt_path)
    keys = list(ckpt["callbacks"].keys())

    callback_key = None
    for key in keys:
        if key.startswith("ModelCheckpoint"):
            callback_key = key
    if callback_key is None:
        logger.error(
            "No ModelCheckpoint callback key found in the checkpoint file. "
            "Cannot get model score. Aborting."
        )
        exit(101)

    best_score = ckpt["callbacks"][callback_key]["best_model_score"]
    metric = ckpt["callbacks"][callback_key]["monitor"]
    epoch = ckpt["epoch"]
    if best_score is not None:
        best_score = best_score.item()
    else:
        lc_info = os.path.join(
            model_dir, "lightning_logs", "version_0", "metrics.csv"
        )
        metrics_df = read_csv(lc_info)
        valid_loss_df = metrics_df[metric]
        best_score = valid_loss_df.loc[valid_loss_df.last_valid_index()]

    return best_score, metric, epoch
