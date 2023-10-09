import numpy as np

from helpers.logger import get_logger

logger = get_logger()


class DataScaler:

    def __init__(self, train_set, method, tag):

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
        return self.x_scaler, self.y_scaler

    @staticmethod
    def __fit_scaler(data, method):
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

    def __init__(self):
        self.native_mean = None
        self.native_std = None

    def fit(self, data):
        self.native_mean = np.mean(data, axis=0)
        self.native_std = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.native_mean) / self.native_std

    def inverse_transform(self, data):
        return (data * self.native_std) + self.native_mean


class LinScale:

    def __init__(self, target_min=0, target_max=1):
        self.native_min = None
        self.native_max = None
        self.target_min = target_min
        self.target_max = target_max

    def fit(self, data):
        self.native_min = np.min(data, axis=0)
        self.native_max = np.max(data, axis=0)

    def transform(self, data):
        return ((self.target_max - self.target_min) *
                ((data - self.native_min) /
                 (self.native_max - self.native_min)) + self.target_min)

    def inverse_transform(self, data):
        return ((self.native_max - self.native_min) *
                ((data - self.target_min) /
                 (self.target_max - self.target_min)) + self.native_min)


class NoScale:

    def __init__(self):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data