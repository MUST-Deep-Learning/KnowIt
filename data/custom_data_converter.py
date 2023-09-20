__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the methods to convert raw data to a dataset option.'

"""

A BaseDataset object can be initialized with a pd.DataFrame containing raw, unclean data,
or a path to one. This script contains the methods that convert such a dataframe into the required 
dataset option.

--------------
Expected input
--------------

Whether a dataframe `df` or path to one is provided, we expect it to meet some conditions:
1.  df.attrs must contain the required (as defined in BaseDataset.required_base_meta) meta data.
    Otherwise it can be provided in the form of a 'meta' dictionary in convert_df and convert_df_from_path.
2.  The headers of df must contain the components defined in the metadata from point 1.
3.  df must be time indexed.
4.  df must not contain any all-NaN columns.
5.  Values in the columns corresponding to the input components (as defined in the metadata) must be numeric.
    If the task is 'regression', this also applies to the target components.

"""

import pandas as pd
import numpy as np
from collections import defaultdict

from helpers.read_configs import load_from_path
from helpers.logger import get_logger

logger = get_logger()


def convert_df_from_path(path, required_meta, fill_nans, meta=None):
    """load the given dataframe and pass it along"""
    data_df = load_from_path(path)
    return convert_df(data_df, required_meta, fill_nans, meta)


def convert_df(data_df, required_meta, fill_nans, meta=None):
    """
    Convert a DataFrame into a dataset dictionary with metadata.

    Parameters:
    data_df (pandas.DataFrame): The DataFrame containing the raw data.
    required_meta (dict[str]): A dictionary of required metadata.
    fill_nans (bool): If True, fill missing values with linear interpolation during data cleaning.
    meta (dict, optional): A dictionary containing metadata for the dataset. If not provided,
                          the function will attempt to extract metadata from the DataFrame's attributes.

    Returns:
    dict: A dictionary containing the dataset and its metadata.

    The function performs the following steps:
    1. Extract or use provided metadata for the dataset.
    2. Check the format of the DataFrame to ensure it matches the expected format.
    3. Split the DataFrame into instances if an 'instance' column is present.
    4. Clean the data by filling or splitting missing values and other necessary operations.
    5. Compile the cleaned data into a structured format.
    6. Construct a dataset dictionary with metadata and the cleaned data.

    """

    # find meta data
    data_meta = meta
    if not data_meta:
        data_meta = data_df.attrs
    if not data_meta:
        logger.error('Error obtaining meta data for raw data.')
        exit(101)

    # check dataframe format
    data_df = check_df(data_df, data_meta, required_meta)

    # split dataframe by instance if applicable
    the_data = {}
    if 'instance' in list(data_df.columns):
        for i, df in data_df.groupby(data_df['instance']):
            the_data[i] = df.drop('instance', axis=1)
    else:
        the_data[0] = data_df

    the_data = clean_data(the_data, data_meta, fill_nans)
    the_data = compile_data(the_data, data_meta)

    # construct dataset
    dataset_dict = {}
    dataset_dict.update(data_meta)
    dataset_dict['the_data'] = the_data

    return dataset_dict


def compile_data(the_data, data_meta):

    """Format the clean dataframe into the knowit the_data structure."""

    # check that the defined io chunks are welldefined
    x0 = data_meta['in_chunk'][0]
    x1 = data_meta['in_chunk'][1]
    y0 = data_meta['out_chunk'][0]
    y1 = data_meta['out_chunk'][1]
    if x0 > x1:
        logger.error('in_chunk undefined range %s', str(data_meta['in_chunk']))
        exit(101)
    if y0 > y1:
        logger.error('out_chunk undefined range %s', str(data_meta['out_chunk']))
        exit(101)

    back_chunk = min(x0, y0)
    forward_chunk = max(x1, y1)

    chunk_size = sum([abs(back_chunk), abs(forward_chunk)])

    to_remove = defaultdict(list)
    for i in the_data:
        for s in range(len(the_data[i])):

            d = the_data[i][s]
            t = d.index.to_numpy()
            x = d[data_meta['input_components']].to_numpy()
            y = d[data_meta['target_components']].to_numpy()
            y = y.astype(float)
            if len(t) > chunk_size:
                # kill targets that don't have corresponding inputs
                if back_chunk < 0:
                    y[:abs(back_chunk), :] = np.nan
                if forward_chunk > 0:
                    y[-forward_chunk:, :] = np.nan
            else:
                to_remove[i].append(s)
            the_data[i][s] = {'t': t, 'x': x, 'y': y}

    # remove slices that are too short for even 1 prediction
    for i in to_remove:
        to_remove[i].reverse()
        for s in to_remove[i]:
            del the_data[i][s]

    # remove instances that don't have any appropriate slices
    instances = list(the_data.keys())
    for i in instances:
        if len(the_data[i]) == 0:
            del the_data[i]

    return the_data


def clean_data(the_data, data_meta, fill_nans):
    """
    Clean and preprocess the data within each instance in 'the_data' dictionary.

    Parameters:
    the_data (dict): A dictionary containing instances of data to be cleaned.
    data_meta (dict): Metadata describing the structure of the data.
    fill_nans (bool): If True, fill missing values with linear interpolation; otherwise, split on missing values.

    Returns:
    dict: A dictionary containing cleaned and preprocessed data instances.

    The function performs the following steps for each instance in 'the_data':
    1. Order the data and remove duplicate time steps within the instance.
    2. Split the instance into contiguous blocks based on the specified time delta.
    3. Handle missing values within each block:
       - If 'fill_nans' is True, missing values are filled using linear interpolation.
       - If 'fill_nans' is False, the block is split into smaller blocks at missing value boundaries.
    4. Keep the cleaned data instances and remove instances with no valid data.

    """

    instances_to_keep = []
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

        if len(new_d_slices) > 0:
            the_data[i] = new_d_slices
            instances_to_keep.append(i)
        else:
            logger.warning('Removing instance %s.', str(i))

    the_data = dict((i, the_data[i]) for i in instances_to_keep)

    return the_data


def split_on_timedelta(df, timedelta):
    """
    Split a DataFrame 'df' into contiguous blocks based on a specified time delta.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be split.
    timedelta (str or pandas.Timedelta): The time duration used to split the DataFrame.

    Returns:
    list of pandas.DataFrame: A list of DataFrames, where each DataFrame represents a contiguous time block.

    The function splits the input DataFrame 'df' into contiguous blocks based on the 'timedelta' parameter.
    It creates a new column 'split_group' in the DataFrame to label each block. Blocks are identified based on
    gaps in time greater than the specified 'timedelta'. The resulting list contains DataFrames representing
    each contiguous time block.

    """

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
    """
    Split a DataFrame 'd' into contiguous blocks based on missing values in specified 'components'.

    Parameters:
    d (pandas.DataFrame): The DataFrame to be split.
    components (list of str): A list of column names in 'd' to consider for missing value-based splitting.

    Returns:
    list of pandas.DataFrame: A list of DataFrames, where each DataFrame represents a contiguous block
                              with no missing values in the specified components.

    The function creates contiguous blocks within the input DataFrame 'd' based on missing values in the
    specified 'components'. It adds a 'nan_group' column to label each block. Blocks are identified based
    on the presence of rows without missing values in the specified components. The resulting list contains
    DataFrames representing each contiguous block.

    """

    d = d.copy()
    heads = list(d.columns)
    d_slices = []

    if d.notna().all(axis=1).any():
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
    else:
        logger.warning('No non-nan rows to split on. Ignoring slice.')

    return d_slices


def check_df(df, meta, required_meta):
    """performs various checks on the provided dataframe and metadata"""

    # check that all meta arguments are present
    if not set(required_meta).issubset(set(list(meta.keys()))):
        logger.error('Not all meta arguments provided.')
        exit(101)

    # check meta type format
    for m in meta:
        good = False
        for m_type in required_meta[m]:
            if isinstance(meta[m], m_type):
                good = True
                break
        if not good:
            logger.error('Provided %s should be of type %s.', m, str(required_meta[m]))
            exit(101)


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
    if 'instance' in unknown_components:
        unknown_components.remove('instance')
    if len(unknown_components) > 0:
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
    if meta['task'] == 'regression':
        df[meta['target_components']].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())

    return df