__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the DataScaler class for Knowit.'

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
    -   if method='z-norm' (default) then the mean and standard deviation of each feature-timestep pair
            is used to standardize the features.
    -   if method='zero-one' then each feature-timestep pair is linearly scaled to (0, 1) separately.
    -   if method=None then no features will be scaled.  

In practice, when either tag=None or method=None, a 'NoScale' scaler is used, that
does not change any feature values.

Scalers are returned with the DataScaler.get_scalers() function.

"""

# external imports
from numpy import (mean, std, min, max, array)

# internal imports
from helpers.logger import get_logger

logger = get_logger()


class DataScaler:

    def __init__(self, train_set: dict, method: str, tag: str):

        """ Instantiates a DataScaler object and runs main operations. """

        if tag:
            if tag == 'in_only':
                self.x_scaler = self.__fit_scaler(train_set['x'], method)
                self.y_scaler = self.__fit_scaler(train_set['y'], None)
            elif tag == 'full':
                self.x_scaler = self.__fit_scaler(train_set['x'], method)
                self.y_scaler = self.__fit_scaler(train_set['y'], method)
            else:
                logger.error('Unknown scaling tag %s', tag)
                exit(101)
        else:
            self.x_scaler = self.__fit_scaler(None, None)
            self.y_scaler = self.__fit_scaler(None, None)

    def get_scalers(self):
        """ Returns the fitted scalers. """
        return self.x_scaler, self.y_scaler

    @staticmethod
    def __fit_scaler(data: array, method: str):

        """ Fits the appropriate scaler based on method. """

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
    first column in the data represents samples."""

    def __init__(self):
        self.native_mean = None
        self.native_std = None

    def fit(self, data):
        self.native_mean = mean(data, axis=0)
        self.native_std = std(data, axis=0)

    def transform(self, data):
        return (data - self.native_mean) / self.native_std

    def inverse_transform(self, data):
        return (data * self.native_std) + self.native_mean


class LinScale:

    """ Performs a basic per feature linear scaling across samples assuming the
    first column in the data represents samples. Can be scaled to any range,
    but default is (0, 1). """

    def __init__(self, target_min=0, target_max=1):
        self.native_min = None
        self.native_max = None
        self.target_min = target_min
        self.target_max = target_max

    def fit(self, data):
        self.native_min = min(data, axis=0)
        self.native_max = max(data, axis=0)

    def transform(self, data):
        return ((self.target_max - self.target_min) *
                ((data - self.native_min) /
                 (self.native_max - self.native_min)) + self.target_min)

    def inverse_transform(self, data):
        return ((self.native_max - self.native_min) *
                ((data - self.target_min) /
                 (self.target_max - self.target_min)) + self.native_min)


class NoScale:

    """ A dummy scaler performing no changes to data features."""

    def __init__(self):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data