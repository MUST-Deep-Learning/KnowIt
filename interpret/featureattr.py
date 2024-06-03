"""
------------------
FeatureAttribution
------------------

The ``FeatureAttribution'' class is a child class which inherits from
``KIInterpreter''.

The function of the ``FeatureAttribution'' class is to serve the user's choice
of feature attribution method (a descendant class) by extracting the necessary
information from Knowit's datamodule and returning it in the expected form for
Captum.

"""# noqa: INP001, D205, D212, D400, D415

from __future__ import annotations  # required for Python versions <3.9

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the class for performing feature attribution."

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import tensor

from helpers.logger import get_logger
from interpret.interpreter import KIInterpreter

logger = get_logger()


class FeatureAttribution(KIInterpreter):
    """Provide methods for Captum's feature attribution modules.

    Args:
    ----
        model (type):           The Pytorch model architecture defined in
                                ./archs

        model_params (dict):    The dictionary needed to intialize model.

        path_to_ckpt (str):     The path to a trained model's checkpoint file.

        datamodule (type)       The Knowit datamodule for the experiment.

        i_data (str)            The user's choice of dataset to perform feature
                                attribution. Choices: 'train', 'valid', 'eval'.

    """

    def __init__(
        self,
        model: type,
        model_params: dict,
        path_to_ckpt: str,
        datamodule: object,
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
        self, point_ids: int | tuple, *, is_baseline: bool = False,
    ) -> tensor:
        """Return the corresponding data points from Knowit's Datamodule.

        Args:
        ----
            point_ids (Union[int, tuple]):      A single prediction point or a
                                                range specified by a tuple.

            is_baseline (bool):                 A flag to determine if user is
                                                sampling points for generating
                                                baselines.

        Returns:
        -------
            tensor (tensor):                    A Pytorch tensor of shape
                                                (number_of_points_ids,
                                                in_chunk,
                                                in_components).

        """
        if is_baseline:
            data_loader = self.datamodule.get_dataloader(
                "train",
            )  # only sample baselines from training set
        else:
            data_loader = self.datamodule.get_dataloader(self.i_data)

        if isinstance(point_ids, tuple):
            ids = list(range(point_ids[0], point_ids[1]))
        else:
            ids = point_ids

        try:
            tensor = data_loader.dataset.__getitem__(idx=ids)["x"]
        except ValueError:
            logger.exception(
                'Invalid: ids %s not in choice "%s" (which has range %s)',
                str(point_ids),
                str(self.i_data),
                str(getattr(self.datamodule, self.i_data + "_set_size")),
            )
            sys.exit()

        return tensor
