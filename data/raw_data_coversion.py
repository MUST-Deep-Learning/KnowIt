__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the RawDataConverter class for Knowit.'

"""
--------------------
RawDataConverter
--------------------
This module takes a given dataframe and converts it to a known datastructure for Knowit.
The dataframe needs to comply with a set of conditions to be properly converted:
1. Must be time indexed. (with a pandas.Timedelta or datetime.timedelta, not strings)
2. Must contain the required meta data (as defined in BaseDataset.__required_base_meta).
    in the dataframe.attrs dictionary. Meta data can alternatively be passed with the 'meta' argument.
     - name (str)
     - components (list)
     - time_delta (Timedelta, timedelta)
3. Must contain no all-NaN columns.
4. Must contain columns corresponding to the components defined in the meta data.
5. If instances are desired, they must be defined in the meta data as instances(list)
    And a corresponding column 'instance' must be present in the dataframe. 
    This column cannot have any NaNs.
    
The resulting datastructure can be returned with the RawDataConverter.get_new_data function.
The format of the resulting data structure is defined at the top of the base_dataset.py script.

--------------------
Handling NaNs
--------------------
RawDataConverter is instantiated with two special variables to handle possible NaNs:
 - nan_filler (str, None): 
    - None (default): No NaNs will be filled. 
    - split:          Slices will be split on NaNs.
    - ''method'' from pandas.DataFrame.interpolate used to interpolate NaNs in both directions.
        (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)
 - nan_filled_components (str, None)
    - None (default): All columns with float datatype will be flagged for NaN-handling.
    - A list of column headers to flag for NaN-handling.
    
--------
Note!
--------

- Duplicate time steps within an instance are dropped after ordering chronologically.
- Time series are split into slices based on the time_delta presented in the meta data.

"""

# external imports
from pandas import DatetimeIndex, concat, DataFrame
from numpy import array

# internal imports
from helpers.logger import get_logger

logger = get_logger()


class RawDataConverter:

    def __init__(self, df: DataFrame, required_meta: dict,
                 nan_filler: str = None,
                 nan_filled_components: str = None,
                 meta: dict = None):
        """
        Instantiate the RawDataConverter module with the given arguments.

        Parameters:
            df (DataFrame): The DataFrame containing the raw data to be processed.
            required_meta (dict): A dictionary specifying the required metadata for processing.
            nan_filler (str, optional): A string representing how to handle missing values
                in the data (default is None).
            nan_filled_components (str, optional): A string specifying which components to treat
                for missing values (default is None).
            meta (dict, optional): Alternative metadata to be associated
                with the raw data (default is None).

        Calls internal methods to check metadata, DataFrame, split data by instances, and compile the data.
        """

        self.df = df
        self.required_meta = required_meta
        self.nan_filler = nan_filler
        self.nan_filled_components = nan_filled_components
        self.meta = meta
        self.the_data = None
        self.instances = None

        self.__check_meta()
        self.__check_df()
        self.__split_by_instance()
        self.__compile_data()

    def get_new_data(self):
        """ Return the converted data structure. """
        new_data = {}
        new_data.update(self.meta)
        new_data['instances'] = self.instances
        new_data['base_nan_filler'] = self.nan_filler
        new_data['nan_filled_components'] = self.nan_filled_components
        new_data['the_data'] = self.the_data
        return new_data

    def __compile_data(self):
        """ For each instance and slice, drop duplicates, split on time delta,
            and handling missing values"""

        instance_to_keep = []
        for i in self.the_data:
            i_slices = self.the_data[i][self.meta['components']]

            # drop duplicates
            i_slices, num_dropped = self.__drop_duplicates(i_slices)
            if num_dropped > 0:
                logger.warning('Instance %s has ' + str(num_dropped) +
                               ' duplicates, dropping!', str(i))

            # split into contiguous blocks
            i_slices = self.__split_on_timedelta(i_slices, self.meta['time_delta'])

            # handle missing values
            i_slices = self.__handle_nans(i, i_slices)

            if len(i_slices) > 0:
                # compile into known format
                self.the_data[i] = []
                for s in range(len(i_slices)):
                    t = i_slices[s].index.to_numpy()
                    d = i_slices[s][self.meta['components']].to_numpy()
                    self.the_data[i].append({'t': t, 'd': d})
                instance_to_keep.append(i)
            else:
                logger.warning('Found no appropriate slices in instance %s. Dropping instance.', str(i))

        if len(instance_to_keep) > 0:
            self.the_data = dict((k, self.the_data[k]) for k in instance_to_keep)
            self.instances = instance_to_keep
        else:
            logger.error('Found no appropriate slices in any instance. Aborting.')
            exit(101)

    def __split_by_instance(self):
        """ Create a dictionary of instances for further processing. """
        self.the_data = {}
        if self.instances:
            for i, df in self.df.groupby(self.df['instance']):
                self.the_data[i] = df.drop('instance', axis=1)
        else:
            self.the_data['super_instance'] = self.df
            self.instances = ['super_instance']

    def __check_df(self):
        """ Check that the dataframe meets the required conditions. """

        # check all required present components
        components = set(list(self.df.columns))
        required_components = set(self.meta['components'])
        if self.instances:
            required_components.add('instance')
        missing_components = required_components - components
        if len(missing_components) > 0:
            logger.error('Some required components missing from raw data %s.',
                         str(missing_components))
            exit(101)

        # check if dataframe is time indexed
        if not isinstance(self.df.index, DatetimeIndex):
            logger.error('Raw data not time indexed.')
            exit(101)

        # check for all-nan columns
        all_nan = array([self.df[c].isnull().values.all().any() for c in required_components])
        if all_nan.any():
            logger.error('Found all-nan required column in dataframe.')
            exit(101)

        # check if instance column has any NaNs
        if 'instance' in required_components:
            if self.df['instance'].isnull().values.any():
                logger.error('Some instance IDs are NaN.')
                exit(101)

    def __check_meta(self):
        """ Check that the metadata is correctly provided. """

        # find meta
        if not self.meta:
            self.meta = self.df.attrs
            if not self.meta:
                logger.error('Error obtaining meta data for raw data.')
                exit(101)

        # check that all meta arguments are present
        missing_meta = set(self.required_meta) - set(list(self.meta.keys()))
        if len(missing_meta) > 0:
            logger.error('%s meta data needed to convert raw data.', str(missing_meta))
            exit(101)

        # check meta type format
        for m in self.required_meta:
            if not isinstance(self.meta[m], self.required_meta[m]):
                logger.error('Provided %s should be of type %s.', m,
                             str(self.required_meta[m]))
                exit(101)

        # only store required meta and optional instances
        new_meta = {}
        for m in self.required_meta:
            new_meta[m] = self.meta[m]
        if 'instances' in self.meta:
            self.instances = self.meta['instances']
        self.meta = new_meta

    def __handle_nans(self, i: object, i_slices: list):
        """ Handle missing values in the given slices. """

        has_nans = False
        for s in i_slices:
            if s.isnull().values.any():
                has_nans = True
                break

        if has_nans:
            if self.nan_filler == 'split':
                logger.warning('Instance %s has NaN values. Splitting on NaNs.', str(i))
                i_slices = self.__split_nans(i_slices, self.nan_filled_components)
            elif self.nan_filler:
                logger.warning('Instance %s has NaN values. Filling with %s.', str(i),
                               self.nan_filler)
                i_slices = self.__fill_nans(i_slices,
                                            self.nan_filler,
                                            self.nan_filled_components)
            else:
                logger.warning('Instance %s has NaN values. Leaving as is.', str(i))

        return i_slices

    @staticmethod
    def __split_nans(slices: list, nan_filled_components: str):
        """ Split slices on NaN values. """
        new_slices = []
        for d in slices:
            if nan_filled_components:
                float_columns = nan_filled_components
            else:
                float_columns = d.select_dtypes(include=[float]).columns
            d = d.copy()
            heads = list(d.columns)
            d_slices = []
            if d.notna().all(axis=1).any():
                d['nan_group'] = d[float_columns].isnull().any(axis=1).cumsum()
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
            new_slices.extend(d_slices)
        return new_slices

    @staticmethod
    def __fill_nans(slices: list, nan_filler: str, nan_filled_components: str):
        """ Fill NaNs in proved slices. """
        new_slices = []
        for s in slices:
            if nan_filled_components:
                float_columns = nan_filled_components
            else:
                float_columns = s.select_dtypes(include=[float]).columns
            s[float_columns] = s[float_columns].interpolate(method=nan_filler,
                                                            limit_direction='both')
            if not s[float_columns].isnull().values.any():
                new_slices.append(s)
            else:
                logger.warning('No non-nan values to interpolate with. Ignoring slice.')
        return new_slices

    @staticmethod
    def __split_on_timedelta(df: DataFrame, time_delta: object):
        """ Split dataframe (df) based on time_delta. """
        deltas = df.index.to_series().diff()
        mask = deltas > time_delta
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
            if start - time_delta == end:
                d_slices[d] = concat([d_slices[d], d_slices[d + 1]])
                d_slices.pop(d + 1)
            else:
                d += 1
        return d_slices

    @staticmethod
    def __drop_duplicates(df: DataFrame):
        """ Drop duplicates from dataframe (df). """
        len_check = len(df)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        len_check_dropped = len(df)
        return df, len_check - len_check_dropped
