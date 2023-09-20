__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the methods to scale the prepared data in various ways.'

import numpy as np

from helpers.logger import get_logger

logger = get_logger()


def get_scaler(the_data, selection, instances, in_chunk, out_chunk, task, mode='data_feature', method='z-norm'):

    if mode == 'data_feature':
        x_data, y_data = get_raw_data(the_data, selection['train'], instances, in_chunk, out_chunk)
    elif mode == 'model_feature':
        x_data, y_data = get_model_data(the_data, selection['train'], instances, in_chunk, out_chunk)
    else:
        logger.error('Unknown data scaling mode %s.', mode)
        exit(101)

    x_scaler = fit_scaler(x_data, method)
    if task == 'regression':
        y_scaler = fit_scaler(y_data, method)
    elif task == 'classification':
        y_scaler = fit_scaler(y_data, 'none')
    else:
        logger.error('Unknown task for scaling targets %s.', task)
        exit(101)

    return x_scaler, y_scaler


def fit_scaler(data, method):

    # expects data[sample][feature1]...[featureN]

    if method == 'z-norm':
        scaler = Zscale()
        scaler.fit(data)
    elif method == 'zero-one':
        scaler = Linscale()
        scaler.fit(data)
    elif method == 'none':
        scaler = Noscale()
        scaler.fit(data)
    else:
        logger.error('Unknown scaler method %s.', method)
        exit(101)

    return scaler


def get_model_data(the_data, selected, instances, in_chunk, out_chunk):

    ic_back = in_chunk[0]
    ic_forward = in_chunk[1]

    oc_back = out_chunk[0]
    oc_forward = out_chunk[1]

    x_vals = []
    y_vals = []
    for s in selected:
        current_instance = instances[s[0]]
        current_slice = s[1]
        timestep = s[2]
        y = the_data[current_instance][current_slice]['y'][timestep + oc_back:timestep + oc_forward + 1, :]
        x = the_data[current_instance][current_slice]['x'][timestep + ic_back:timestep + ic_forward + 1, :]
        x_vals.append(x)
        y_vals.append(y)

    # [sample][timesteps][features]
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    return x_vals, y_vals


def get_raw_data(the_data, selected, instances, in_chunk, out_chunk):

    ic_back = in_chunk[0]
    ic_forward = in_chunk[1]
    oc_back = out_chunk[0]
    oc_forward = out_chunk[1]

    full_x_selected = set()
    full_y_selected = set()
    for s in selected:
        current_slice = s[1]
        timestep = s[2]
        x_t = np.arange(timestep + ic_back, timestep + ic_forward + 1)
        y_t = np.arange(timestep + oc_back, timestep + oc_forward + 1)
        for t in x_t:
            full_x_selected.add((s[0], current_slice, t))
        for t in y_t:
            full_y_selected.add((s[0], current_slice, t))

    x_vals = []
    y_vals = []
    for x in full_x_selected:
        x_vals.append(the_data[instances[x[0]]][x[1]]['x'][x[2]])
    for y in full_y_selected:
        y_vals.append(the_data[instances[y[0]]][y[1]]['y'][y[2]])

    # [timestep][features]
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    return x_vals, y_vals


class Zscale:

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


class Linscale:

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


class Noscale:

    def __init__(self):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data