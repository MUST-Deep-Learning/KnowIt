"""
------------
DeepLiftShap
------------

DeepLiftShap is a feature attribution method.

For each of the model's output features, feature attribution assigns a value
to each input feature that is based on its contribution to the model's output.

The method is implemented through the Captum library.

For more information on the method, see:
https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

and

https://captum.ai/api/deep_lift_shap.html

"""# noqa: INP001, D205, D212, D400, D415

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = "randlerabe@gmail.com, tiantheunissen@gmail.com"
__description__ = "Implements Captum's DeepLiftShap attribution method."

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

import numpy as np
import torch
from captum.attr import DeepLiftShap
from collections import defaultdict

from helpers.logger import get_logger
from interpret.featureattr import FeatureAttribution

logger = get_logger()


class DLS(FeatureAttribution):
    """Implement the DeepLiftShap feature attribution method.

    Parameters
    ----------
    model : Module
        The Pytorch model architecture class.

    model_params : dict[str, Any]
        The dictionary needed to initialize the model.

    datamodule : type
        The Knowit datamodule for the experiment.

    path_to_ckpt : str
        The path to a trained model's checkpoint file.

    i_data : str
        The user's choice of dataset to perform feature attribution.
        Choices: 'train', 'valid', 'eval'.

    device : str
        On which hardware device to generate attributions.

    seed : int
        The seed to be used by Numpy for random sampling of baselines
        and reproducibility.

    multiply_by_inputs : bool, default=True
        If True, perform local attributions. If False, perform global
        attributions. For more information, see Captum's documentation.

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

    seed : int
        The seed used by Numpy for random sampling of baselines.

    dls : DeepLiftShap
        The DeepLiftShap instance from Captum for feature attribution.
    """

    def __init__(
        self,
        model: Module,
        model_params: dict[str, Any],
        datamodule: type,
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

        self.dls = DeepLiftShap(
            self.model,
            multiply_by_inputs=multiply_by_inputs
        )
        self.seed = seed

    def generate_baseline_from_data(self, num_baselines: int) -> tuple:
        """Return a number of baselines and (optionally) their corresponding internal state.

        Randomly samples a distribution of baselines from the training data.
        If possible, also generates internal baselines by passing the
        baselines through the model and obtaining the internal states.
        Parameters
        ----------
        num_baselines : int
            The total number of baselines to sample.

        Returns
        -------
        Tuple[Tensor, Optional[Tensor | None]]
            A tuple containing:
            - A torch tensor of shape (num_baselines, in_chunk, in_components) representing the baselines.
            - The internal states baselines (if available), otherwise None.

        Notes
        -----
        - If the number of available samples in the training data is less than `num_baselines`,
          the method will use all available samples and log a warning.
        - The internal state baseline is generated only if the model has the `update_states`, `force_reset`,
          and `get_internal_states` methods.
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

        x_baselines = self._fetch_points_from_datamodule(
            baselines,
            is_baseline=True,
        )['x']
        x_baselines = x_baselines.to(self.device)

        internal_state_baselines = None
        if (hasattr(self.model, 'update_states') and
                hasattr(self.model, 'force_reset') and
                hasattr(self.model, 'get_internal_states')):
                # manually reset internal states to have the same number of dimensions as number of baselines
                self.model.force_reset()
                self.model.update_states(torch.zeros(x_baselines.shape[0]), x_baselines.device)
                # pass baselines through model to generate internal states
                _ = self.model(x_baselines)
                # get baseline internal states
                internal_state_baselines = self.model.get_internal_states()
                # manually reset internal states back to default dimensionality
                self.model.force_reset()
                self.model.update_states(torch.zeros(1), x_baselines.device)
                self.model.force_reset()

        return x_baselines, internal_state_baselines

    def interpret(
        self,
        pred_point_id: int | tuple[int, int],
        num_baselines: int = 1000,
    ) -> dict[int | tuple[int, int], dict[str, Tensor]]:
        """Return attribution matrices and deltas.

        Generates attribution matrices for a single prediction point or a range
        of prediction points (also referred to as explicands).

        Parameters
        ----------
        pred_point_id : int | tuple
            The prediction point or range of prediction points that will be
            used to generate attribution matrices.

        num_baselines : int, default=1000
            Specifies the size of the baseline distribution.

        Returns
        -------
        results : dict[int | tuple[int, int], dict[str, Tensor]]
            For a regression model with output shape
            (out_chunk, out_components),
            returns a dictionary as follows:
                * Dict Key: a tuple (m, n) with m in range(out_chunk) and
                n in range(out_components).
                * Dict Element: a torch tensor with shape:
                    > (prediction_points, in_chunk, in_components) if
                    pred_point_id is a tuple.
                    > (in_chunk, in_components) if pred_point_id is int.

            For a classification model with output shape (classes,), returns a
            dictionary as follows:
                * Dict Key: a class value from classes.
                * Dict Element: a torch tensor with shape:
                    > (prediction_points, in_chunk, in_components) if
                    pred_point_id is a tuple.
                    > (in_chunk, in_components) if pred_point_id is int.

        Notes
        -----
        The output stores the information from a tensor of size
        (out_chunk, out_components, prediction_points, in_chunk, in_components)
        inside a dictionary data structure. For time series data, this can grow
        rapidly, which may therefore obscure model interpretability.
        """

        def _extract_attributions(custom_batch: dict, baseline: Tensor,
                                  target: tuple, internal_baseline: list | None):
            """ Extract the feature attributions one-by-one and concatenate at the end. """

            # separate prediction points for sequential processing
            pseudo_batches = []
            for p in range(custom_batch['x'].shape[0]):
                new_batch = {'x': torch.unsqueeze(custom_batch['x'][p], 0),
                             'y': torch.unsqueeze(custom_batch['y'][p], 0),
                             's_id': custom_batch['s_id'][p],
                             'ist_idx': np.expand_dims(custom_batch['ist_idx'][p], 0)}
                pseudo_batches.append(new_batch)

            # manually reset internal states back to default dimensionality
            if hasattr(self.model, 'update_states') and hasattr(self.model, 'force_reset'):
                self.model.force_reset()
                self.model.update_states(torch.zeros(1), pseudo_batches[0]['x'].device)
                self.model.force_reset()

            full_attributions = defaultdict(list)
            full_delta = []
            for pp in range(len(pseudo_batches)):

                # if possibility of statefulness, update internal state to current prediction point
                if hasattr(self.model, 'update_states') and hasattr(self.model, 'force_reset'):
                    self.model.force_reset()
                    for qpp in range(0, pp):
                        self.model.update_states(torch.from_numpy(pseudo_batches[qpp]['ist_idx']),
                                                 pseudo_batches[qpp]['x'].device)
                        _ = self.model(pseudo_batches[qpp]['x'])

                if hasattr(self.model, 'get_internal_states'):
                    # if model has the possibility of being stateful construct input including hidden states
                    inputs = [pseudo_batches[pp]['x']]
                    for i in self.model.get_internal_states():
                        if type(i) is list or type(i) is tuple:
                            inputs.extend(i)
                        else:
                            inputs.append(i)
                    baselines = [baseline]
                    for i in internal_baseline:
                        if type(i) is list or type(i) is tuple:
                            baselines.extend(i)
                        else:
                            baselines.append(i)
                    attributions, delta = self.dls.attribute(
                        inputs=tuple(inputs),
                        baselines=tuple(baselines),
                        target=target,
                        return_convergence_delta=True,
                    )
                else:
                    # assume model is not stateful and simply attribute the input
                    attributions, delta = self.dls.attribute(
                        inputs=tuple([pseudo_batches[pp]['x']]),
                        baselines=tuple([baseline]),
                        target=target,
                        return_convergence_delta=True,
                    )

                for a in range(len(attributions)):
                    full_attributions[a].append(attributions[a])
                full_delta.append(delta)

            attributions = {}
            for a in full_attributions:
                attributions[a] = torch.cat(full_attributions[a])
            delta = torch.cat(full_delta)

            return attributions, delta

        # extract explicands using ids
        custom_batch = super()._fetch_points_from_datamodule(pred_point_id)

        custom_batch['x'] = custom_batch['x'].to(self.device)
        custom_batch['x'].requires_grad = True  # required by Captum

        # generate baseline distribution
        baselines, internal_baselines = self.generate_baseline_from_data(
            num_baselines=num_baselines,
        )

        # determine model output type
        if hasattr(self.datamodule, "class_set") and self.datamodule.class_set is not None:
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
                attributions, delta = _extract_attributions(custom_batch, baselines, target, internal_baselines)

                results[target] = {
                    "attributions": attributions,
                    "delta": delta,
                }
        else:
            for out_chunk in range(out_shape[0]):
                for out_component in range(out_shape[1]):
                    target = (out_chunk, out_component)
                    attributions, delta = _extract_attributions(custom_batch, baselines, target, internal_baselines)

                    results[target] = {
                        "attributions": attributions,
                        "delta": delta,
                    }

        return results
