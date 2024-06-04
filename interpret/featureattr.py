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

"""  # noqa: INP001, D205, D212, D400, D415

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Contains the class for performing feature attribution."

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

from helpers.logger import get_logger
from interpret.interpreter import KIInterpreter

logger = get_logger()


class FeatureAttribution(KIInterpreter):
    """Provide methods for Captum's feature attribution modules."""

    def __init__(
        self,
        model: type,
        model_params: dict[str, Any],
        path_to_ckpt: str,
        datamodule: type,
        i_data: str,
        device: str,
    ) -> None:
        """FeatureAttribution constructor.

        Args:
        ----
            model (type):           The Pytorch model architecture class.

            model_params (dict):    The dictionary needed to intialize model.

            path_to_ckpt (str):     The path to a trained model's checkpoint
                                    file.

            datamodule (type):      The Knowit datamodule for the experiment.

            i_data (str):           The user's choice of dataset to perform
                                    feature attribution. Choices: 'train',
                                    'valid', 'eval'.

            device (str):           The hardware device to be used for feature
                                    extraction (either cpu or gpu).

        """
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
    ) -> Tensor:
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
