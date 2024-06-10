"""
---------------
DataScaler
---------------
This module takes the extracted training set from PreparedDataset and fits two scalers to the data,
one for the input of the model and one for the output. This is done based on the criteria presented by the
'method' and 'tag' arguments. These scalers are used in PreparedDataset to scale the features upon further
extractions of the datasets.

The 'tag' argument defines what features will be scaled.
    -   if tag='in_only' then only input features to the model is scaled.
    -   if tag='full' then the input and output features to the model are scaled.
    -   if tag=None (default) then no features will be scaled.

The 'method' argument defines the type of scaler to use.
    -   if method='z-norm' (default) then the mean and standard deviation of each feature-timedelay pair
            is used to standardize the features.
    -   if method='zero-one' then each feature-timedelay pair is linearly scaled to (0, 1) separately.
    -   if method=None then no features will be scaled.

In practice, when either tag=None or method=None, a 'NoScale' scaler is used, that
does not change any feature values.

Scalers are returned with the DataScaler.get_scalers() function.

"""

from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataScaler class for KnowIt.'

# external imports
from numpy import (mean, std, min, max, array)

# internal imports
from helpers.logger import get_logger

logger = get_logger()


class DataScaler:

    """

        The DataScaler module is used by PreparedDataset to scale the raw data for model training.

    """

    def __init__(self, train_set: dict, method: str, tag: str):

        """
            Instantiate a DataScaler object and run main operations.

            This method initializes the DataScaler object and fits the scalers for the input ('x') and output ('y')
            data based on the provided scaling method and tag. The tag determines whether the scaler should be applied
            to input only, or to both input and output data.

            Args:
            -----
            train_set : dict
                A dictionary containing the training set data with keys 'x' for input features
                and 'y' for output features.
            method : str
                The scaling method to use for fitting the scalers. Currently, supports 'z-norm' and 'zero-one'.
            tag : str
                A string indicating the type of scaling to apply. It can be:
                - 'in_only': Scale only the input features using the specified method. Output features are not scaled.
                - 'full': Scale both input features and output features using the specified method.
                - Any other value will result in an error.

            Attributes:
            -----------
            x_scaler : Scaler or None
                The scaler fitted to the input features.
            y_scaler : Scaler or None
                The scaler fitted to the output labels.

            Raises:
            -------
            ValueError
                If an unknown scaling tag is provided.

        """

        if tag:
            if tag == 'in_only':
                self.x_scaler = self._fit_scaler(train_set['x'], method)
                self.y_scaler = self._fit_scaler(train_set['y'], None)
            elif tag == 'full':
                self.x_scaler = self._fit_scaler(train_set['x'], method)
                self.y_scaler = self._fit_scaler(train_set['y'], method)
            else:
                logger.error('Unknown scaling tag %s', tag)
                exit(101)
        else:
            self.x_scaler = self._fit_scaler(None, None)
            self.y_scaler = self._fit_scaler(None, None)

    def get_scalers(self):
        """ Returns the fitted scalers. """
        return self.x_scaler, self.y_scaler

    @staticmethod
    def _fit_scaler(data: array, method: str | None):

        """
            Fit the appropriate scaler based on the specified method.

            This method selects and fits a scaler to the provided data based on the chosen scaling method.
            The method expects the data to be in the format data[sample][feature1]...[featureN].

            Args:
            -----
            data : array
                The data to fit the scaler on. It should be an array where each row represents a sample
                and each column represents a feature.
            method : str
                The scaling method to use. Options include:
                - 'z-norm': Standardizes data to have zero mean and unit variance using ZScale.
                - 'zero-one': Scales data to the range [0, 1] using LinScale.
                - None: No scaling is applied using NoScale.
                - Any other value will result in an error.

            Returns:
            --------
            scaler : object
                The fitted scaler object based on the chosen method.

            Raises:
            -------
            ValueError
                If an unknown scaler method is provided.

        """

        # expects data[sample][feature1]...[featureN]
        if method == 'z-norm':
            scaler = ZScale()
            scaler.fit(data)
        elif method == 'zero-one':
            scaler = LinScale()
            scaler.fit(data)
        elif method is None:
            scaler = NoScale()
            scaler.fit(data)
        else:
            logger.error('Unknown scaler method %s.', method)
            exit(101)

        return scaler


class ZScale:

    """ Performs a basic per feature standardization across samples assuming the
    first axis in the data represents samples. """

    def __init__(self):
        self.native_mean = None
        self.native_std = None

    def fit(self, data: array):
        """ Records the mean and std across samples. """
        self.native_mean = mean(data, axis=0)
        self.native_std = std(data, axis=0)

    def transform(self, data: array):
        """ Performs Z-Normalization. """
        return (data - self.native_mean) / self.native_std

    def inverse_transform(self, data: array):
        """ Inverts Z-Normalization. """
        return (data * self.native_std) + self.native_mean


class LinScale:

    """ Performs a basic per feature linear scaling across samples assuming the
    first axis in the data represents samples. Can be scaled to any range,
    but default is (0, 1). """

    def __init__(self, target_min: float = 0, target_max: float = 1):
        self.native_min = None
        self.native_max = None
        self.target_min = target_min
        self.target_max = target_max

    def fit(self, data: array):
        """ Records the max and min across samples. """
        self.native_min = min(data, axis=0)
        self.native_max = max(data, axis=0)

    def transform(self, data: array):
        """ Performs linear scaling. """
        return ((self.target_max - self.target_min) *
                ((data - self.native_min) /
                 (self.native_max - self.native_min)) + self.target_min)

    def inverse_transform(self, data: array):
        """ Inverts linear scaling. """
        return ((self.native_max - self.native_min) *
                ((data - self.target_min) /
                 (self.target_max - self.target_min)) + self.native_min)


class NoScale:

    """ A dummy scaler performing no changes to data features."""

    def __init__(self):
        pass

    def fit(self, data: array):
        """ Does nothing. """
        pass

    def transform(self, data: array):
        """ Performs nothing. """
        return data

    def inverse_transform(self, data: array):
        """ Inverts nothing. """
        return data
