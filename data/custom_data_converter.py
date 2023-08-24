__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the methods to convert raw data to a dataset option.'

import pandas as pd
import numpy as np

from helpers.read_configs import load_from_path
from helpers.logger import get_logger

logger = get_logger()


def extract_df(path, meta=None, fill_nans=True):

    # load dataframe
    data_df = load_from_path(path)

    # find meta data
    data_meta = meta
    if not data_meta:
        data_meta = data_df.attrs
    if not data_meta:
        logger.error('Error obtaining meta data for raw data.')
        exit(101)

    # check dataframe format
    data_df = check_df(data_df, data_meta)

    # split dataframe by instance if applicable
    the_data = {}
    if 'instance' in list(data_df.columns):
        for i, df in data_df.groupby(data_df['instance']):
            the_data[i] = df.drop('instance', axis=1)
    else:
        the_data[0] = data_df

    the_data = clean_data(the_data, data_meta, fill_nans)
    the_data = compile_data(the_data, data_meta)

    data_meta['the_data'] = the_data

    return data_meta


def compile_data(the_data, data_meta):

    for i in the_data:
        for s in range(len(the_data[i])):
            d = the_data[i][s]
            t = d.index
            x = d[data_meta['input_components']]
            y = d[data_meta['target_components']]
            #TODO: WIP


            ping = 0

    return the_data


def clean_data(the_data, data_meta, fill_nans):

    for i in the_data:

        df = the_data[i]

        # order and drop duplicates
        len_check = len(df)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        len_check_dropped = len(df)
        if len_check != len_check_dropped:
            logger.warning('Instance %s has ' + str(len_check - len_check_dropped) +
                           ' duplicates, dropping!', str(i))

        # split into contiguous blocks
        d_slices = split_on_timedelta(df, data_meta['time_delta'])

        # handle missing values
        new_d_slices = []
        for d in d_slices:
            if d[data_meta['input_components']].isnull().values.any():
                if fill_nans:
                    logger.warning('Filling missing values with linear interpolation.')
                    new_d = d[data_meta['input_components']].interpolate(method='linear')
                    new_d = new_d.dropna()
                    # new_d[data_meta['target_components']] = d[data_meta['target_components']]
                    new_d.loc[new_d.index, data_meta['target_components']] = d.loc[new_d.index, data_meta['target_components']]
                    new_d_slices.append(new_d)
                else:
                    logger.warning('Splitting on missing values.')
                    new_d = split_on_nan(d, data_meta['input_components'])
                    new_d_slices.extend(new_d)
            else:
                new_d_slices.append(d)

        the_data[i] = new_d_slices

    return the_data


def split_on_timedelta(df, timedelta):
    deltas = df.index.to_series().diff()
    mask = deltas > timedelta
    heads = list(df.columns)
    df['split_group'] = mask.cumsum()
    unique_groups = df['split_group'].value_counts(sort=False).to_numpy()
    num_unique_groups = len(unique_groups)
    d_slices = []
    for u in range(num_unique_groups):
        new_seq = df.loc[df['split_group'] == u, heads]
        d_slices.append(new_seq)
    d = 0
    while d < len(d_slices) - 1:
        end = d_slices[d].index[-1]
        start = d_slices[d + 1].index[0]
        if start - timedelta == end:
            d_slices[d] = pd.concat([d_slices[d], d_slices[d + 1]])
            d_slices.pop(d + 1)
        else:
            d += 1
    return d_slices


def split_on_nan(d, components):
    d = d.copy()
    heads = list(d.columns)
    d_slices = []
    d['nan_group'] = d[components].isnull().any(axis=1).cumsum()
    if d['nan_group'][0] == 1:
        d['nan_group'] = d['nan_group'] - 1
    unique_groups = d['nan_group'].value_counts(sort=False).to_numpy()
    num_unique_groups = len(unique_groups)
    for u in range(num_unique_groups):
        potential_series = d.loc[d['nan_group'] == u, heads]
        if not potential_series.isnull().values.any():
            d_slices.append(potential_series)
        elif potential_series.shape[0] != 1:
            potential_series = potential_series.drop(index=potential_series.index[0], axis=0)
            d_slices.append(potential_series)

    return d_slices


def check_df(df, meta):

    # check all required present components
    components = set(list(df.columns))
    input_components = set(meta['input_components'])
    target_components = set(meta['target_components'])

    if len(input_components - components) > 0:
        logger.error('Some input components missing from dataframe.')
        exit(101)
    if len(target_components - components) > 0:
        logger.error('Some target components missing from dataframe.')
        exit(101)

    # check unknown components
    unknown_components = components - input_components.union(target_components)
    if len(unknown_components) > 0 and unknown_components != set(['instance']):
        logger.warning('Some unknown components in dataframe ignored: %s.',
                       str(unknown_components))
        df = df.drop(unknown_components, axis=1)
        components = set(list(df.columns))

    # check if dataframe is timeindexed
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error('Dataframe not time indexed. Skipping')
        exit(101)

    # check for all-nan columns
    if (np.array([df[c].isnull().values.all() for c in components])).any():
        logger.error('Found all-nan column in dataframe. Skipping')
        exit(101)

    # check if instance column has any NaNs
    if 'instance' in list(df.columns):
        if df['instance'].isnull().values.any():
            logger.error('Some instance IDs are nan.')
            exit(101)

    # check that inputs and targets are numeric
    df[meta['input_components']].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
    df[meta['target_components']].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())

    return df


extract_df('/home/tian/postdoc_work/knowit/dummy_raw_data/dummy_zero/dummy_zero.pickle')
