"""
-------------
KIInterpreter
-------------

The "KIInterpreter" class is the parent (root) class that is to be
inherited by all other model interpretability modules.

The function of the "KIInterpreter" class is to store the datamodule and
initialize the Pytorch model for use by its descendant classes. As such, it is
a direct link to Knowit's other modules. It is agnostic to the user's choice of
interpretability method.

"""# noqa: INP001, D205, D212, D400, D415

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the Knowit interpreter class."

from typing import Any

import torch

from helpers.logger import get_logger

logger = get_logger()


class KIInterpreter:
    """Root class to be inherited by all interpretability classes.

    The class stores the user's parameters and initializes a trained Pytorch
    model from checkpoint.

    """

    def __init__(
        self,
        model: type,
        model_params: dict[str, Any],
        datamodule: type,
        device: str,
        path_to_ckpt: str,
    ) -> None:
        """KIInterpreter constructor.

        Args:
        ----
            model (type):           The Pytorch model architecture class.

            model_params (dict):    The dictionary needed to intialize model.

            datamodule (type):      The Knowit datamodule for the experiment.

            device (str):           The device on which to run the model.

            path_to_ckpt (str):     The path to a trained model's checkpoint
                                    file.

        """
        self.model = self._load_model_from_ckpt(
            model=model,
            model_params=model_params,
            ckpt_path=path_to_ckpt,
        )
        self.datamodule = datamodule
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.model.to(self.device)

    def _load_model_from_ckpt(
        self,
        model: type,
        model_params: dict[str, Any],
        ckpt_path: str,
    ) -> type[Any]:
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

__all__ = [
    "KIInterpreter",
]