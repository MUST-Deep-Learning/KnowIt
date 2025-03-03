"""
----------
DataScaler
----------
This module takes the extracted training set from ``PreparedDataset`` and fits two scalers to the data,
one for the input of the model and one for the output. This is done based on the criteria presented by the
'method' and 'tag' arguments. These scalers are used in PreparedDataset to scale the features upon further
extractions of the datasets.

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
"""

from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataScaler class for KnowIt.'

# external imports
from numpy import (nanmean, nanvar, nanmin, nanmax, array, unique, sqrt, minimum, maximum, logical_and)

# internal imports
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
        """ Performs nothing. Just returns the provided parameters."""
        return data

    def inverse_transform(self, data: array) -> array:
        """ Inverts nothing. Just returns the provided parameters."""
        return data


class DataScaler:
    """The DataScaler module is used by PreparedDataset to scale the raw data for model training.

    This method initializes the DataScaler object and fits the scalers for the input ('x') and output ('y')
    data based on the provided scaling method and tag. The tag determines whether the scaler should be applied
    to input only, or to both input and output data.

    Parameters
    ----------
    train_set : dict
        A dictionary containing the training set data with keys 'x' for input features
        and 'y' for output features.
    method : str | None
        The scaling method to use for fitting the scalers. Currently, supports 'z-norm', 'zero-one', or None.
    tag : str | None
        A string indicating the type of scaling to apply. It can be:
        - 'in_only': Scale only the input features using the specified method. Output features are not scaled.
        - 'full': Scale both input features and output features using the specified method.
        - None: No scaling is applied to either input features or output features.
        - Any other value will result in an error.
    load_level : str, default='instance'
            What level to load values from disc with.
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

    def __init__(self, data: dict, train_selection: array, method: str | None, tag: str | None,
                 x_map: array, y_map: array, load_level: str = 'instance') -> None:

        if tag:
            # required_lookups = unique(train_selection[:, :2], axis=0)
            if tag == 'in_only':
                self.x_scaler = self._fit_scaler(data, train_selection, x_map, method, load_level)
                self.y_scaler = self._fit_scaler(data, train_selection, y_map, None, load_level)
            elif tag == 'full':
                self.x_scaler = self._fit_scaler(data, train_selection, x_map, method, load_level)
                self.y_scaler = self._fit_scaler(data, train_selection, y_map, method, load_level)
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
    def _fit_scaler(data: dict, train_selection: array, s_map: array, method: str | None, load_level: str = 'instance') -> ZScale | LinScale | NoScale:
        """Fit the appropriate scaler based on the specified method.

        This method selects and fits a scaler to the provided data based on the chosen scaling method.
        The method expects the data to be in the format data[sample][feature1]...[featureN].

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data to fit the scaler on. It should be an array where the first dimension
            refers to samples (i.e. prediction points).
        method : str
            The scaling method to use. Options include:
            - 'z-norm': Standardizes data to have zero mean and unit variance using ZScale.
            - 'zero-one': Scales data to the range [0, 1] using LinScale.
            - None: No scaling is applied using NoScale.

        Returns
        -------
        scaler : object
            The fitted scaler object based on the chosen method.

        Raises
        ------
        ValueError
            If an unknown scaler method is provided.

        """
        # expects data[sample][feature1]...[featureN]
        if method == 'z-norm':
            scaler = ZScale()
            scaler.fit(data, train_selection, s_map, load_level)
        elif method == 'zero-one':
            scaler = LinScale()
            scaler.fit(data, train_selection, s_map, load_level)
        elif method is None:
            scaler = NoScale()
            scaler.fit(data)
        else:
            logger.error('Unknown scaler method %s.', method)
            exit(101)

        return scaler


class ZScale:
    """Performs a basic per feature standardization across samples (i.e. prediction points)
    assuming the first axis in the data represents samples.

    For each feature f the transformation is defined as:
        (f - native_mean) / native_std

    Attributes
    ----------
    native_mean : array, shape=[n_time_delays, n_components]
        The mean feature value across prediction points.
    native_std : array, shape=[n_time_delays, n_components]
        The standard deviation of feature values across prediction points.
    """
    native_mean = None
    native_std = None

    def __init__(self) -> None:
        pass

    def fit(self, data: dict, train_selection: array, s_map: array, load_level: str = 'slice') -> None:
        """Records the mean and std across prediction points.

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data for which the mean and std across prediction points are recorded.
        """

        def _rec_mean_std(vals, mu_m, v_m, m):
            """ Recursively update the mean and std across prediction points.
            See: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html"""
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
                instance_vals = data.instance(i)
            for s in slices:
                t = train_selection[logical_and(train_selection[:, 0] == i, train_selection[:, 1] == s), 2]
                if load_level == 'instance':
                    vals = instance_vals[instance_vals['slice'] == s]
                    vals = vals.drop(columns=['slice'])
                    vals = vals.to_numpy()
                else:
                    vals = data.slice(i, s).to_numpy()
                vals = vals[t, :]
                vals = vals[:, s_map]
                mean, variance, count = _rec_mean_std(vals, mean, variance, count)


        self.native_mean = mean
        self.native_std = sqrt(variance)

    def transform(self, data: array) -> array:
        """ Performs Z-Normalization.

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
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
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data to be inversely transformed with Z-Normalization.
        """
        if self.native_mean is None or self.native_std is None:
            logger.error('ZScale transform not fitted yet.')
            exit(101)

        return (data * self.native_std) + self.native_mean


class LinScale:
    """Performs a basic per feature linear scaling across samples assuming the
    first axis in the data represents samples. Can be scaled to any range,
    but default is (0, 1).

    For each feature f the transformation is defined as:
        (target_max - target_min) * (f - native_min) / (native_max - native_min) + target_min

    Parameters
    ----------
    target_min : float, default=0
        The desired transformed minimum feature values.
    target_max : float, default=1
        The desired transformed maximum feature values.

    Attributes
    ----------
    native_min : array, shape=[n_time_delays, n_components]
        The minimum native feature values across prediction points.
    native_max : array, shape=[n_time_delays, n_components]
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

    def fit(self, data: dict, train_selection: array, s_map: array, load_level: str = 'instance') -> None:
        """ Records the max and min across prediction points.

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data for which the max and min across prediction points are recorded.
        """

        min = None
        max = None
        instances = unique(train_selection[:, 0], axis=0)
        for i in instances:
            slices = unique(train_selection[train_selection[:, 0] == i, 1])
            if load_level == 'instance':
                instance_vals = data.instance(i)
            for s in slices:
                t = train_selection[logical_and(train_selection[:, 0] == i, train_selection[:, 1] == s), 2]
                if load_level == 'instance':
                    vals = instance_vals[instance_vals['slice'] == s]
                    vals = vals.drop(columns=['slice'])
                    vals = vals.to_numpy()
                else:
                    vals = data.slice(i, s).to_numpy()
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
        """Tranforms features, linearly, from expected ranges to desired range.

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data to be transformed with Linear scaling.
        """
        if not self.native_min or not self.native_max:
            logger.error('LinScale transform not fitted yet.')
            exit(101)

        return ((self.target_max - self.target_min) *
                ((data - self.native_min) /
                 (self.native_max - self.native_min)) + self.target_min)

    def inverse_transform(self, data: array) -> array:
        """Inversely tranforms features, linearly, back to native ranges.

        Parameters
        ----------
        data : array, shape=[n_prediction points, n_time_delays, n_components]
            The data to be inversely transformed with Linear scaling.
        """
        if self.native_min is None or self.native_max is None:
            logger.error('LinScale transform not fitted yet.')
            exit(101)

        return ((self.native_max - self.native_min) *
                ((data - self.target_min) /
                 (self.target_max - self.target_min)) + self.native_min)
