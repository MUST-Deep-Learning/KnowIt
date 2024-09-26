"""
-------------
KIInterpreter
-------------

The "KIInterpreter" class is the parent (root) class that is to be inherited
by all other model interpretability modules.

The function of the "KIInterpreter" class is to store the datamodule and
initialize the Pytorch model for use by its descendant classes. As such, it is
a direct link to Knowit's other modules. It is agnostic to the user's choice of
interpretability method.

"""  # noqa: D205, D400

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the Knowit interpreter class."

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.nn import Module

import torch
from torch import nn

from helpers.logger import get_logger

logger = get_logger()


class KIInterpreter:
    """Root class to be inherited by all interpretability classes.

    The class stores the user's parameters and initializes a trained PyTorch
    model from checkpoint.

    Parameters
    ----------
    model : type
        The PyTorch model architecture class.

    model_params : dict
        The dictionary needed to initialize the model.

    datamodule : type
        The Knowit datamodule for the experiment.

    device : str
        The device on which to run the model.

    path_to_ckpt : str
        The path to a trained model's checkpoint file.

    Attributes
    ----------
    model : Module
        The initialized PyTorch model loaded with weights from the checkpoint.

    datamodule : type
        The Knowit datamodule for the experiment.

    device : torch.device
        The device on which the model is run.
    """

    def __init__(
        self,
        model: Module,
        model_params: dict[str, Any],
        datamodule: type,
        device: str,
        path_to_ckpt: str,
    ) -> None:
        self.model = self._load_model_from_ckpt(
            model=model,
            model_params=model_params,
            ckpt_path=path_to_ckpt,
        )
        mods = [
            str(m) for m in self.model.modules() if not\
                isinstance(m, nn.Sequential)
        ]
        for m in mods:
            if "lstm" in m.lower() or "rnn" in m.lower():
                torch.backends.cudnn.enabled = False # see Captum's FAQ
                break


        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.model.to(self.device)
        self.datamodule = datamodule

    def _load_model_from_ckpt(
        self,
        model: Module,
        model_params: dict[str, Any],
        ckpt_path: str,
    ) -> Module:
        """Load a PyTorch model from a checkpoint file.

        Parameters
        ----------
        model : Module
            The PyTorch model class to be initialized.
        model_params : dict[str, Any]
            The parameters for initializing the model.
        ckpt_path : str
            The path to the checkpoint file.

        Returns
        -------
        Module
            The model loaded with weights from the checkpoint.
        """
        logger.info("Initializing Pytorch model using checkpoint file.")

        # init Pytorch model with user params
        model = model(**model_params)

        # obtain model weights from ckpt
        ckpt = torch.load(f=ckpt_path)
        model_weights = ckpt["state_dict"]

        # For each key, PL saves it as "model.model." instead of "model." as
        # expected by Pytorch.
        # The below method is implemented in PL's own example, see:
        # https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        for key in list(model_weights):
            model_weights[key.replace("model.", "", 1)] = model_weights.pop(
                key,
            )

        # load model weights and set to eval mode
        model.load_state_dict(model_weights)
        model.eval()

        return model
