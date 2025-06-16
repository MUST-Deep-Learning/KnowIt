"""
------------------
FeatureAttribution
------------------

The "FeatureAttribution" class is a child class which inherits from
"KIInterpreter".

The function of the "FeatureAttribution" class is to serve the user's choice
of feature attribution method (a descendant class) by extracting the necessary
information from Knowit's datamodule and returning it in the expected form for
Captum.

"""  # noqa: D205, D400

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com"
__description__ = "Contains the class for performing feature attribution."

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

from helpers.logger import get_logger
from interpret.interpreter import KIInterpreter

logger = get_logger()


class FeatureAttribution(KIInterpreter):
    """Provide methods for Captum's feature attribution modules.

    Parameters
    ----------
    model : Module
        The Pytorch model architecture class.

    model_params : dict
        The dictionary needed to initialize the model.

    path_to_ckpt : str
        The path to a trained model's checkpoint file.

    datamodule : type
        The Knowit datamodule for the experiment.

    i_data : str
        The user's choice of dataset to perform feature attribution.
        Choices: 'train', 'valid', 'eval'.

    device : str
        The hardware device to be used for feature extraction (either cpu or
        gpu).

    Attributes
    ----------
    model : Module
        The initialized PyTorch model loaded with weights from the checkpoint.

    datamodule : type
        The Knowit datamodule for the experiment.

    device : torch.device
        The device on which the model is run.

    i_data : str
        The user's choice of dataset to perform feature attribution.
    """

    def __init__(
        self,
        model: Module,
        model_params: dict[str, Any],
        path_to_ckpt: str,
        datamodule: type,
        i_data: str,
        device: str,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            path_to_ckpt=path_to_ckpt,
            datamodule=datamodule,
            device=device,
        )

        self.i_data = i_data

    def _fetch_points_from_datamodule(
        self,
        point_ids: int | list[int] | tuple[int, int],
        *,
        is_baseline: bool = False,
    ) -> dict:
        """Fetch data points from the datamodule based on provided point IDs.

        Parameters
        ----------
        point_ids : int | list[int] | tuple[int, int]
            The IDs of the data points to fetch. Can be a single integer, a
            list of integers, or a tuple specifying a range (start, end).
        is_baseline : bool, default=False
            If True, fetches baseline points from the training set.

        Returns
        -------
        dict
            A dictionary containing the data points corresponding to the provided
            IDs at key 'x'.

        Raises
        ------
        ValueError
            If the provided point IDs are invalid or out of range.

        Notes
        -----
        If `is_baseline` is True, the method fetches data points from the
        training set. Otherwise, it fetches data points from the dataset
        specified by `self.i_data`.
        """

        if is_baseline:
            set_tag = "train"
        else:
            set_tag = self.i_data
        return self.datamodule.fetch_input_points_manually(set_tag, point_ids)
