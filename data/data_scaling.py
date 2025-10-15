"""
----------
DataScaler
----------

This module takes a provided ``DataExtractor`` object (see ``KnowIt.Basedataset``) and train selection matrix
and fits two scalers to the data, one for the input of the model and one for the output.
This is done based on the criteria presented by the 'method' and 'tag' arguments.
These scalers are used in ``CustomDataset`` to scale the features when examples
are sampled from the dataloader.

The 'tag' argument defines what features will be scaled.
    -   if tag='in_only' then only input features to the model are scaled.
    -   if tag='full' then the input and output features to the model are scaled.
    -   if tag=None (default) then no features will be scaled.

The 'method' argument defines the type of scaler to use.
    -   if method='z-norm' (default) then the mean and standard deviation of each feature-timedelay pair is used to standardize the features.
    -   if method='zero-one' then each feature-timedelay pair is linearly scaled to (0, 1) separately.
    -   if method=None then no features will be scaled.

In practice, when either tag=None or method=None, a ``NoScale`` scaler is used, that
does not change any feature values.

Scalers are returned with the ``DataScaler.get_scalers`` function.

A scaler always expects their inputs to be an array of shape=[n_time_delays, n_components].
This applies to both inputs and outputs.

"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataScaler, ZScale, LinScale, and NoScale classes for KnowIt.'

# external imports
from numpy import (nanmean, nanvar, nanmin, nanmax, array, unique, sqrt, minimum, maximum, logical_and)
import torch

# internal imports
from data.base_dataset import DataExtractor
from helpers.logger import get_logger

logger = get_logger()


class NoScale:

    """A dummy scaler performing no scaling to data features.
    """

    def __init__(self) -> None:
        pass

    def fit(self, data: array) -> None:
        """ Does nothing. """
        pass

    def transform(self, data: array) -> array:
        """ Transforms nothing. Just returns the provided parameters."""
        return data

    def inverse_transform(self, data: array) -> array:
        """ Inverts nothing. Just returns the provided parameters."""
        return data


class DataScaler:
    """The DataScaler module is used by CustomDataset to scale the raw data for model training.

    This method initializes the DataScaler object and fits the scalers to the train set data,
    as defined by the train selection matrix, based on the provided scaling method and tag.
    The tag determines whether the scaler should be applied to input only, or to both input and output data.

    Parameters
    ----------
    data_extractor : DataExtractor
        The data extractor object to read data from disk.
    train_selection : array, shape=[n_train_prediction_points, 3]
        The selection matrix corresponding to the train set.
    method : str | None
        The scaling method to use for fitting the scalers. Currently, supports 'z-norm', 'zero-one', or None.
    tag : str | None
        A string indicating the type of scaling to apply. It can be:
        - 'in_only': Scale only the input features using the specified method. Output features are not scaled.
        - 'full': Scale both input features and output features using the specified method.
        - None: No scaling is applied to either input features or output features.
        - Any other value will result in an error.
    load_level : str, default='instance'
        What level to load values from disk with.
        If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
        If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.

    Attributes
    ----------
    x_scaler : ZScale | LinScale | NoScale, default=NoScale
        The scaler fitted to the input features.
    y_scaler : ZScale | LinScale | NoScale, default=NoScale
        The scaler fitted to the output labels.

    Raises
    -------
    ValueError
        If an unknown scaling tag is provided.
    """
    x_scaler = NoScale()
    y_scaler = NoScale()

    def __init__(self, data_extractor: DataExtractor, train_selection: array, method: str | None, tag: str | None,
                 x_map: array, y_map: array, load_level: str = 'instance') -> None:

        if tag is not None:
            if tag == 'in_only':
                self.x_scaler = self._fit_scaler(data_extractor, train_selection, x_map, method, load_level)
                self.y_scaler = self._fit_scaler(None, None, None, None, load_level)
            elif tag == 'full':
                self.x_scaler = self._fit_scaler(data_extractor, train_selection, x_map, method, load_level)
                self.y_scaler = self._fit_scaler(data_extractor, train_selection, y_map, method, load_level)
            else:
                logger.error('Unknown scaling tag %s', tag)
                exit(101)
        else:
            self.x_scaler = self._fit_scaler(None, None, None, None, load_level)
            self.y_scaler = self._fit_scaler(None, None, None, None, load_level)

    def get_scalers(self) -> tuple:
        """Returns the fitted scalers.

        This method retrieves the scalers that have been fitted to the data. It is used to obtain
        the scaling objects for both the input features (x_scaler) and the target data (y_scaler).

        Returns
        -------
        tuple
            - x_scaler : ZScale | LinScale | NoScale
                The scaler fitted to the input features.
            - y_scaler : ZScale | LinScale | NoScale
                The scaler fitted to the target data.
        """
        return self.x_scaler, self.y_scaler

    @staticmethod
    def _fit_scaler(data_extractor: DataExtractor | None, train_selection: array | None,
                    s_map: array | None, method: str | None, load_level: str = 'instance') -> ZScale | LinScale | NoScale:
        """Fit the appropriate scaler based on the specified method.

        Parameters
        ----------
        data_extractor : DataExtractor | None
            The data extractor object to read data from disk.
        train_selection : array | None, shape=[n_train_prediction_points, 3]
            The selection matrix corresponding to the train set.
        s_map : array
            Mapping for relevant components. This could be the x_map, or y_map found in PreparedDataset.
        method : str
            The scaling method to use. Options include:
            - 'z-norm': Standardizes data to have zero mean and unit variance using ZScale.
            - 'zero-one': Scales data to the range [0, 1] using LinScale.
            - None: No scaling is applied using NoScale.
        load_level : str, default='instance'
            What level to load values from disk with.
            If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
            If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.

        Returns
        -------
        scaler : object
            The fitted scaler object based on the chosen method.

        Raises
        ------
        ValueError
            If an unknown scaler method is provided.

        """

        if method == 'z-norm':
            scaler = ZScale()
            scaler.fit(data_extractor, train_selection, s_map, load_level)
        elif method == 'zero-one':
            scaler = LinScale()
            scaler.fit(data_extractor, train_selection, s_map, load_level)
        elif method is None:
            scaler = NoScale()
            scaler.fit(data_extractor)
        else:
            logger.error('Unknown scaler method %s.', method)
            exit(101)

        return scaler


class ZScale:
    """Performs a basic per component standardization.

    For each component value c, the transformation is defined as:
        (c - native_mean) / native_std

    where native_mean and native_std are the mean and std of the corresponding component as measured on the train set.

    Attributes
    ----------
    native_mean : array, shape=[n_components,]
        The mean component value across prediction points in the train set.
    native_std : array, shape=[n_components,]
        The standard deviation of values across prediction points in the train set.
    """
    native_mean = None
    native_std = None

    def __init__(self) -> None:
        pass

    def fit(self, data_extractor: DataExtractor, train_selection: array,
            s_map: array, load_level: str = 'instance') -> None:
        """Records the mean and std across prediction points in the train set.

        Parameters
        ----------
        data_extractor : DataExtractor
            The data extractor object to read data from disk.
        train_selection : array, shape=[n_train_prediction_points, 3]
            The selection matrix corresponding to the train set.
        s_map : array
            Mapping for relevant components. This could be the x_map, or y_map found in PreparedDataset.
        load_level : str, default='instance'
            What level to load values from disk with.
            If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
            If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.
        """

        def _rec_mean_std(vals, mu_m, v_m, m):
            """
            Recursively update the mean and std across prediction points.
            We use the method described here http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
            by Matt Hancock.
            """
            mu_n = nanmean(vals, axis=0)
            n = vals.shape[0]
            v_n = nanvar(vals, axis=0)
            if mu_m is None:
                return mu_n, n, v_n
            else:
                mean = (m / (m + n)) * mu_m + (n / (m + n)) * mu_n
                variance = ((m / (m + n)) * v_m +
                            (n / (m + n)) * v_n +
                            ((m * n) / ((m + n) * (m + n))) * (mu_m - mu_n) * (mu_m - mu_n))
                count = m + n
                return mean, variance, count

        mean = None
        count = None
        variance = None
        instances = unique(train_selection[:, 0], axis=0)
        for i in instances:
            slices = unique(train_selection[train_selection[:, 0] == i, 1])
            if load_level == 'instance':
                instance_vals = data_extractor.instance(i)
            for s in slices:
                t = train_selection[logical_and(train_selection[:, 0] == i, train_selection[:, 1] == s), 2]
                if load_level == 'instance':
                    vals = instance_vals[instance_vals['slice'] == s]
                    vals = vals.drop(columns=['slice'])
                    vals = vals.to_numpy()
                else:
                    vals = data_extractor.slice(i, s).to_numpy()
                vals = vals[t, :]
                vals = vals[:, s_map]
                mean, variance, count = _rec_mean_std(vals, mean, variance, count)

        self.native_mean = mean
        self.native_std = sqrt(variance)

    def transform(self, data: array) -> array:
        """ Performs Z-Normalization.

        Parameters
        ----------
        data : array, shape=[n_components,]
            The data to be transformed with Z-Normalization.
        """
        if self.native_mean is None or self.native_std is None:
            logger.error('ZScale transform not fitted yet.')
            exit(101)

        return (data - self.native_mean) / self.native_std

    def inverse_transform(self, data: array) -> array:
        """ Performs inverse Z-Normalization.

        Parameters
        ----------
        data : array, shape=[n_components,]
            The data to be inversely transformed with Z-Normalization.
        """
        if self.native_mean is None or self.native_std is None:
            logger.error('ZScale transform not fitted yet.')
            exit(101)

        return (data * self.native_std) + self.native_mean


class LinScale:
    """Performs a basic per component linear scaling.
    Can be scaled to any range, but default is (0, 1).

    For each component value c the transformation is defined as:
        (target_max - target_min) * (c - native_min) / (native_max - native_min) + target_min

    where native_min and native_mix are the min and max values of the corresponding component
    as measured on the train set, and target_min and target_max is 0 and 1 respectively.

    Parameters
    ----------
    target_min : float, default=0
        The desired transformed minimum feature values.
    target_max : float, default=1
        The desired transformed maximum feature values.

    Attributes
    ----------
    native_min : array, shape=[n_components,]
        The minimum native feature values across prediction points.
    native_max : array, shape=[n_components,]
        The maximum native feature values across prediction points.
    target_min : float, default=0
        The transformed minimum feature values.
    target_max : float, default=1
        The transformed maximum feature values.

    """
    native_min = None
    native_max = None
    target_min = 0.
    target_max = 1.

    def __init__(self, target_min: float = 0, target_max: float = 1) -> None:
        self.target_min = target_min
        self.target_max = target_max

    def fit(self, data_extractor: DataExtractor, train_selection: array,
            s_map: array, load_level: str = 'instance') -> None:
        """Records the min and max across prediction points in the train set.

        Parameters
        ----------
        data_extractor : DataExtractor
            The data extractor object to read data from disk.
        train_selection : array, shape=[n_train_prediction_points, 3]
            The selection matrix corresponding to the train set.
        s_map : array
            Mapping for relevant components. This could be the x_map, or y_map found in PreparedDataset.
        load_level : str, default='instance'
            What level to load values from disk with.
            If load_level='instance' an instance at a time will be loaded. This is memory heavy, but faster.
            If load_level='slice' a slice at a time will be loaded. This is lighter on memory, but slower.
        """

        min = None
        max = None
        instances = unique(train_selection[:, 0], axis=0)
        for i in instances:
            slices = unique(train_selection[train_selection[:, 0] == i, 1])
            if load_level == 'instance':
                instance_vals = data_extractor.instance(i)
            for s in slices:
                t = train_selection[logical_and(train_selection[:, 0] == i, train_selection[:, 1] == s), 2]
                if load_level == 'instance':
                    vals = instance_vals[instance_vals['slice'] == s]
                    vals = vals.drop(columns=['slice'])
                    vals = vals.to_numpy()
                else:
                    vals = data_extractor.slice(i, s).to_numpy()
                vals = vals[t, :]
                vals = vals[:, s_map]
                if min is None:
                    min = nanmin(vals, axis=0)
                    max = nanmax(vals, axis=0)
                else:
                    min = minimum(min, nanmin(vals, axis=0))
                    max = maximum(max, nanmax(vals, axis=0))

        self.native_min = min
        self.native_max = max

    def transform(self, data: array) -> array:
        """Transforms features, linearly, from expected ranges to desired range.

        Parameters
        ----------
        data : array, shape=[n_components,]
            The data to be transformed with Linear scaling.
        """
        if self.native_min is None or self.native_max is None:
            logger.error('LinScale transform not fitted yet.')
            exit(101)

        # TODO: TEMP WORKARAOUND, NEED BETTER SOLUTION
        if torch.is_tensor(data):
            data = torch.as_numpy(data)

        return ((self.target_max - self.target_min) *
                ((data - self.native_min) /
                 (self.native_max - self.native_min)) + self.target_min)

    def inverse_transform(self, data: array) -> array:
        """Inversely transforms features, linearly, back to native ranges.

        Parameters
        ----------
        data : array, shape=[n_components,]
            The data to be inversely transformed with Linear scaling.
        """
        if self.native_min is None or self.native_max is None:
            logger.error('LinScale transform not fitted yet.')
            exit(101)

        # TODO: TEMP WORKARAOUND, NEED BETTER SOLUTION
        if torch.is_tensor(data):
            native_max = torch.as_tensor(self.native_max)
            native_min = torch.as_tensor(self.native_min)
        else:
            native_max = self.native_max
            native_min = self.native_min

        return ((native_max - native_min) *
                ((data - self.target_min) /
                 (self.target_max - self.target_min)) + native_min)
