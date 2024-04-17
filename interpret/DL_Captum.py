"""
--------
DeepLift
--------

DeepLift is a feature attribution method.

For each of the model's output features, feature attribution assigns a value
to each input feature that is based on its contribution to the model's output.

The method is implemented through the Captum library.

For more information on the method, see:
https://arxiv.org/abs/1704.02685

and

https://captum.ai/api/deep_lift.html

"""# noqa: INP001, D205, D212, D400, D415

from __future__ import annotations

__author__ = "randlerabe@gmail.com"
__description__ = "Implements Captum's DeepLift attribution method."

import numpy as np
import torch
from captum.attr import DeepLift

from helpers.logger import get_logger
from interpret.featureattr import FeatureAttribution

logger = get_logger()


class DeepL(FeatureAttribution):
    """Implement the DeepLift feature attribution method.

    Args:
    ----
        model (type):           The Pytorch model architecture defined in
                                ./archs.

        model_params (dict):    The dictionary needed to intialize model.

        datamodule (type):      The Knowit datamodule for the experiment.

        path_to_ckpt (str):     The path to a trained model's checkpoint file.

        i_data (str):           The user's choice of dataset to perform feature
                                attribution. Choices: 'train', 'valid', 'eval'.

        device (str):           On which hardware device to generate
                                attributions.

        seed (int):             The seed to  be used by Numpy for random
                                sampling of baselines and reproducibility.

        multiply_by
        _inputs (bool):         If True, perform local attributions. If False,
                                perform global attributions. For more inform-
                                ation, see Captum's documentation.

    """

    def __init__(
        self,
        model: type,
        model_params: dict,
        datamodule: object,
        path_to_ckpt: str,
        i_data: str,
        device: str,
        seed: int,
        *,
        multiply_by_inputs: bool = True,
    ) -> None:

        super().__init__(
            model=model,
            model_params=model_params,
            datamodule=datamodule,
            path_to_ckpt=path_to_ckpt,
            i_data=i_data,
            device=device,
        )

        self.dl = DeepLift(self.model, multiply_by_inputs=multiply_by_inputs)
        self.seed = seed

    def generate_baseline_from_data(self, num_baselines: int) -> torch.tensor:
        """Return a (single) baseline.

        Randomly samples a distribution of baselines from the training data and
        then averages over the sample to obtain a single baseline.

        Args:
        ----
            num_baselines (int):            The total number of baselines to
                                            sample.

        Returns:
        -------
            (torch.tensor):                 A torch tensor of shape
                                            (1, in_chunk, in_components)

        """
        logger.info("Generating baselines.")

        num_samples = self.datamodule.train_set_size
        if num_samples < num_baselines:
            logger.warning(
                "Not enough prediction points for %s baselines. Using %s.",
                str(num_samples),
                str(num_baselines),
            )
            num_baselines = num_samples

        rng = np.random.default_rng(seed=self.seed)
        baselines = rng.choice(
            np.arange(num_samples),
            size=num_baselines,
            replace=False,
        )
        baselines = list(baselines)

        baselines = self._fetch_points_from_datamodule(
            baselines,
            is_baseline=True,
        )

        return torch.mean(baselines, 0, keepdim=True)

    def interpret(
        self,
        pred_point_id: int | tuple,
        num_baselines: int = 1000,
    ) -> dict[int | tuple, dict[str, torch.tensor]]:
        """Return attribution matrices and deltas.

        Generates attribution matrices for a single prediction point or a range
        of prediction points (also referred to as explicands).

        NOTE: The output stores the information from a tensor of size

        (out_chunk, out_components, prediction_points, in_chunk, in_components)

        inside a dictionary data structure. For time series data, this can grow
        rapidly, which may therefore obscure model interpretability.

        Args:
        ----
            pred_point_id (int | tuple):    The prediction point or range of
                                            prediction points that will be used
                                            to generate attribution matrices.

            num_baselines (int):            Specifies the size of the baseline
                                            distribution.

        Returns:
        -------
            results (dict):                 For a regression model with output
                                            shape (out_chunk, out_components),
                                            returns a dictionary as follows:
                                                * Dict Key: a tuple (m, n) with
                                                    m in range(out_chunk) and n
                                                    in range(out_components).

                                                * Dict Element: a torch tensor
                                                    with shape:
                                                        > (prediction_points,
                                                        in_chunk, in_components
                                                        )
                                                        if pred_point_id is a
                                                        tuple

                                                        > (in_chunk,
                                                        in_components) if
                                                        pred_point_id is int.

                                            For a classification model with
                                            output shape (classes,), returns a
                                            dictionary as follows:
                                                * Dict Key: an class value from
                                                    classes
                                                * Dict Element: a torch tensor
                                                    with shape:
                                                        > (prediction_points,
                                                        in_chunk,
                                                        in_components)
                                                        if pred_point_id is a
                                                        tuple

                                                        > (in_chunk,
                                                        in_components) if
                                                        pred_point_id is int.

        """
        # extract explicands using ids
        input_tensor = super()._fetch_points_from_datamodule(pred_point_id)

        # Captum requires batch dimension = number of explicands
        if not isinstance(pred_point_id, tuple):
            input_tensor = torch.unsqueeze(
                input_tensor,
                0,
            )

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True  # required by Captum

        # generate a baseline (baseline is computed as an average over a sample
        # distribution)
        baseline = self.generate_baseline_from_data(
            num_baselines=num_baselines,
        )
        baseline = baseline.to(self.device)

        # determine model output type
        if hasattr(self.datamodule, "class_set"):
            logger.info(
                "Preparing attribution matrices for classification task.",
            )
            is_classification = True
        else:
            logger.info("Preparing attribution matrices for regression task.")
            is_classification = False
            out_shape = self.datamodule.out_shape

        # compute attribution matrices for each output component and each input
        # explicand
        results = {}
        if is_classification:
            for key in self.datamodule.class_set:
                target = self.datamodule.class_set[key]
                attributions, delta = self.dl.attribute(
                    inputs=input_tensor,
                    baselines=baseline,
                    target=target,
                    return_convergence_delta=True,
                )
                attributions = torch.squeeze(attributions, 0)

                results[target] = {
                    "attributions": attributions,
                    "delta": delta,
                }
        else:
            for out_chunk in range(out_shape[0]):
                for out_component in range(out_shape[1]):
                    target = (out_chunk, out_component)
                    attributions, delta = self.dl.attribute(
                        inputs=input_tensor,
                        baselines=baseline,
                        target=target,
                        return_convergence_delta=True,
                    )
                    attributions = torch.squeeze(attributions, 0)

                    results[target] = {
                        "attributions": attributions,
                        "delta": delta,
                    }

        return results
