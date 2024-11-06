"""
----------------
RawDataConverter
----------------

This module takes a given dataframe and converts it to a known datastructure for KnowIt.
See ``dataset_how_to.md`` for format details.
The resulting datastructure can be returned with the ``RawDataConverter.get_new_data`` function.
The format of the resulting data structure is defined at the top of the KnowIt.data.base_dataset.py script.

-------------
Handling NaNs
-------------
``RawDataConverter`` is instantiated with two special variables to handle possible NaNs:
    - nan_filler (str, None): Defines what method to use for NaN filling.
        - None: No NaNs will be filled.
        - 'split': Slices will be split on NaNs.
        - Any ``method`` value from ``pandas.DataFrame.interpolate``: used to interpolate NaNs in both directions.(see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)
    - nan_filled_components (str, None): Defines what components should be checked for NaNs.
        - None: All columns with float datatype will be flagged for NaN-handling.
        - A list of column headers to flag for NaN-handling.
    
---------
Take note
---------
    - Duplicate time steps within an instance are dropped after ordering chronologically.
    - Time series are split into slices based on the time_delta presented in the metadata.

"""

from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the RawDataConverter class for Knowit.'

# external imports
from pandas import DatetimeIndex, concat, DataFrame
from numpy import array
from datetime import timedelta
from pandas import Timedelta

# internal imports
from helpers.logger import get_logger

logger = get_logger()


class RawDataConverter:
    """This module takes a given dataframe and converts it to a known datastructure for KnowIt.

    This module first retrieves and checks the metadata and dataframe for correctness,
    then it splits the dataframe according to instances (if defined).
    It then compiles the datastructure on an instance by instance basis by:
        1. Sorting timesteps by time index.
        2. Dropping duplicate time steps (keeping the first of each duplicate).
        3. Splitting data into contiguous blocks (slices).
        4. Handling missing values.
        5. Packaging data into known format as defined in ``BaseDataset``.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the raw data to be processed.
    required_meta : dict
        A dictionary specifying the required metadata for processing, defined in ``BaseDataset``. Also, see heading.
    nan_filler : str | None
        A string representing how to handle missing values in the data. If None, no NaNs will be filled.
        Options include 'split', and any method value from ``pandas.DataFrame.interpolate``.
    nan_filled_components : list | None
        A list of strings specifying which components to treat for missing values. If None, all float-valued components checked for NaNs.
    meta : dict | None
        Alternative metadata to be associated with the raw data. Overrides possible metadata in the ``dataframe.attrs`` dictionary.

    Attributes
    ----------
    df : DataFrame
        The DataFrame containing the raw data to be processed.
    required_meta : dict
        A dictionary specifying the required metadata for processing, defined in ``BaseDataset``. Also, see heading.
    nan_filler : str | None
        A string representing how to handle missing values in the data. If None, no NaNs will be filled.
        Options include 'split', and any method value from ``pandas.DataFrame.interpolate``.
    nan_filled_components : list | None
        A list of strings specifying which components to treat for missing values. If None, all float-valued components checked for NaNs.
    meta : dict | None, default=None
        Alternative metadata to be associated with the raw data. Overrides possible metadata in the ``dataframe.attrs`` dictionary.

    Notes
    -----
        - If no instances are defined, all slices are placed in a single instance called 'super_instance'.
        - Duplicate time steps are dropped based on their time index, disregarding feature values.

    """
    df = None
    required_meta = None
    nan_filler = None
    nan_filled_components = None
    meta = None
    the_data = None
    instances = None

    def __init__(self, df: DataFrame, required_meta: dict,
                 nan_filler: str, nan_filled_components: list, meta: dict = None) -> None:

        self.df = df
        self.required_meta = required_meta
        self.nan_filler = nan_filler
        self.nan_filled_components = nan_filled_components
        self.meta = meta

        self._check_meta()
        self._check_df()
        self._split_by_instance()
        self._compile_data()
        self._summarize_data()

    def get_new_data(self) -> dict:
        """Return the converted data structure.

        This method returns a dictionary representing the converted data structure,
        which includes metadata and other relevant components.

        Returns
        -------
        new_data : dict
            A dictionary containing the following keys:
                - 'instances': The instances in the dataset.
                - 'base_nan_filler': The method or value used to fill NaN values in the dataset.
                - 'nan_filled_components': The components in the dataset where potential NaN values were filled.
                - 'the_data': The main data structure.

        Notes
        -----
            - The returned dictionary also contains the provided metadata (i.e. name, components, time_delta)
        """
        new_data = {}
        new_data.update(self.meta)
        new_data['instances'] = self.instances
        new_data['base_nan_filler'] = self.nan_filler
        new_data['nan_filled_components'] = self.nan_filled_components
        new_data['the_data'] = self.the_data
        return new_data

    def _summarize_data(self) -> None:
        """Displays a summary of the compiled dataset for debugging purposes."""

        logger.info('- - - - - - - - - - COMPILED DATASET - - - - - - - - - ')
        logger.info(' NAME: %s', self.meta['name'])
        logger.info(' COMPONENTS: %s', str(self.meta['components']))
        logger.info(' INSTANCES: %s', str(self.instances))

        summary = [[len(s['t']) for s in self.the_data[i]] for i in self.instances]
        num_slices = sum([len(i) for i in summary])
        num_pp = sum([sum([t for t in i]) for i in summary])

        logger.info(' TOTAL INSTANCES: %s', str(len(summary)))
        logger.info(' TOTAL SLICES: %s', str(num_slices))
        logger.info(' TOTAL PREDICTION POINTS: %s', str(num_pp))

        logger.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')

    def _compile_data(self) -> None:
        """Process and compile data for each instance and slice.

        This method performs several data processing steps for each instance in the dataset:
            - Drops duplicate entries.
            - Splits data into contiguous blocks based on a specified time delta.
            - Handles missing values.
        After processing, it compiles the data into a known format and updates the dataset.

        Steps
        -----
            1. Iterate through each instance in `self.the_data`.
            2. Drop duplicate entries (after ordering chronologically).
            3. Split data into contiguous blocks based on `self.meta['time_delta']`.
            4. Handle missing values using the appropriate method.
            5. Compile the processed data into the required format.
            6. Update `self.the_data` with the processed data.

        Notes
        -----
            - If no appropriate slices are found for an instance, log a warning and drop the instance.
            - If no appropriate slices are found for any instance, log an error and abort.
        """
        instance_to_keep = []
        for i in self.the_data:

            # drop duplicates
            i_df, num_dropped = self._drop_duplicates(self.the_data[i][self.meta['components']])
            if num_dropped > 0:
                logger.warning('Instance %s has ' + str(num_dropped) +
                               ' duplicates, dropping!', str(i))

            # split into contiguous blocks
            i_slices = self._split_on_timedelta(i_df, self.meta['time_delta'])

            # handle missing values
            i_slices = self._handle_nans(i, i_slices, self.nan_filler, self.nan_filled_components)

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

    def _split_by_instance(self) -> None:
        """Split the dataset into separate instances for further processing.

        This method creates a dictionary (`self.the_data`) where each key is an instance identifier,
        and the corresponding value is the data associated with that instance. If no instances are
        specified, the entire dataset is treated as a single 'super_instance'.
        """
        self.the_data = {}
        if self.instances:
            for i, df in self.df.groupby(self.df['instance']):
                self.the_data[i] = df.drop('instance', axis=1)
        else:
            self.the_data['super_instance'] = self.df
            self.instances = ['super_instance']

    def _check_df(self) -> None:
        """Check that the dataframe meets the required conditions.

        This method performs several checks to ensure that the dataframe (`self.df`)
        is properly structured and contains all necessary components before further
        processing. The checks include verifying the presence of required components,
        ensuring the dataframe is time-indexed, checking for all-NaN columns, and
        ensuring no NaN values in the 'instance' column if applicable.
        """

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

    def _check_meta(self) -> None:
        """Check that the metadata is correctly provided.

        This method ensures that the metadata (`self.meta`) required for processing
        the dataframe (`self.df`) is present, correctly formatted, and complete.
        The checks include verifying the existence of metadata, ensuring all required
        metadata fields are present, checking the types of metadata fields, and storing
        only the required metadata fields and optional instances.
        """

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

    @staticmethod
    def _handle_nans(i: any, i_slices: list, nan_filler: str | None,
                     nan_filled_components: list | None) -> list:
        """Handle missing values in the given slices.

        This method checks for missing values (NaNs) in the provided slices and
        handles them according to the specified `nan_filler` method. Depending on
        the `nan_filler`, it either splits the slices at NaN values, fills NaNs with
        a specified method, or leaves them as they are.

        Parameters
        ----------
        i : any
            The identifier for the current instance being processed.
        i_slices : list
            List of slices (dataframes) to be checked for missing values.
        nan_filler : str | None
            Method for handling NaNs. Possible values are 'split', a specific filler
            method, or None. If 'split', slices are split at NaN values. If a specific
            filler method, NaNs are filled using that method. If None, NaNs are left as is.
        nan_filled_components : list | None
            List of components to be filled when handling NaNs. If None, all components
            are considered.

        Returns
        -------
        i_slices : list
            List of slices with NaNs handled according to the specified method.
        """

        has_nans = False
        for s in i_slices:
            if s.isnull().values.any():
                has_nans = True
                break

        if has_nans:
            if nan_filler == 'split':
                logger.warning('Instance %s has NaN values. Splitting on NaNs.', str(i))
                i_slices = RawDataConverter._split_nans(i_slices, nan_filled_components)
            elif nan_filler:
                logger.warning('Instance %s has NaN values. Filling with %s.', str(i),
                               nan_filler)
                i_slices = RawDataConverter._fill_nans(i_slices,
                                                        nan_filler,
                                                        nan_filled_components)
            else:
                logger.warning('Instance %s has NaN values. Leaving as is.', str(i))

        return i_slices

    @staticmethod
    def _split_nans(slices: list, nan_filled_components: list) -> list:
        """Split slices on NaN values.

        This method processes a list of data slices (dataframes) and splits them
        into new slices wherever NaN values are found in the specified components.
        If no specific components are provided, it defaults to splitting on any
        float-type columns containing NaNs.

        Parameters
        ----------
        slices : list
            List of data slices (dataframes) to be processed for NaN values.
        nan_filled_components : list or None
            List of specific components (column names) to check for NaNs. If None,
            all float-type columns are considered for splitting.

        Returns
        -------
        list
            A list of new data slices with NaNs split out.

        Notes
        -----
            - The method logs a warning if no non-NaN rows are found to split on,
              indicating that the slice is ignored.
            - The 'nan_group' column is added temporarily to identify contiguous blocks
              of non-NaN values, and is removed before returning the final list of slices.
        """
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
    def _fill_nans(slices: list, nan_filler: str, nan_filled_components: list) -> list:
        """Fill NaNs in provided slices.

        This method processes a list of data slices (dataframes) and fills NaN values
        using the specified interpolation method for the given components. If no specific
        components are provided, it defaults to interpolating any float-type columns
        containing NaNs.

        Parameters
        ----------
        slices : list
            List of data slices (dataframes) to be processed for NaN values.
        nan_filler : str
            The interpolation method to be used for filling NaNs.
        nan_filled_components : list or None
            List of specific components (column names) to check for NaNs. If None,
            all float-type columns are considered for filling.

        Returns
        -------
        new_slices : list
            A list of new data slices with NaNs filled according to the specified method.

        Notes
        -----
            - The method logs a warning if no non-NaN values are available to interpolate with,
              indicating that the slice is ignored.
            - Only slices with successfully filled NaNs (or no NaNs to begin with) are included in the returned list.
        """
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
    def _split_on_timedelta(df: DataFrame, time_delta: timedelta | Timedelta) -> list:
        """Split a dataframe based on a given time delta.

        This method splits a dataframe into multiple segments where each segment is separated
        by a time difference greater than the specified `time_delta`. The segments are then
        further checked and concatenated if the difference between the end of one segment
        and the start of the next segment equals `time_delta`.

        Parameters
        ----------
        df : DataFrame
            The dataframe to be split. It is assumed to have a datetime index.
        time_delta : timedelta | Timedelta
            The time delta used to determine the points at which to split the dataframe.

        Returns
        -------
        list
            A list of dataframes, each representing a segment of the original dataframe.
        """
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
    def _drop_duplicates(df: DataFrame) -> tuple:
        """Drop duplicate entries from a dataframe based on its index.

        This method sorts the dataframe by its index and removes duplicate entries,
        keeping only the first occurrence of each index value. It also returns the
        number of duplicates that were dropped.

        Parameters
        ----------
        df : DataFrame
            The dataframe from which to drop duplicate entries.

        Returns
        -------
        DataFrame
            The dataframe with duplicates removed.
        int
            The number of duplicate entries that were dropped.
        """
        len_check = len(df)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        len_check_dropped = len(df)
        return df, len_check - len_check_dropped
