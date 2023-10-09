from pandas import DatetimeIndex, concat
from numpy import array

from helpers.logger import get_logger

logger = get_logger()


class RawDataConverter:

    def __init__(self, df, required_meta, nan_filler, nan_filled_components, meta=None):

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
        new_data = {}
        new_data.update(self.meta)
        new_data['instances'] = self.instances
        new_data['base_nan_filler'] = self.nan_filler
        new_data['nan_filled_components'] = self.nan_filled_components
        new_data['the_data'] = self.the_data
        return new_data

    def __compile_data(self):

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
        self.the_data = {}
        if self.instances:
            for i, df in self.df.groupby(self.df['instance']):
                self.the_data[i] = df.drop('instance', axis=1)
        else:
            self.the_data['super_instance'] = self.df
            self.instances = ['super_instance']

    def __check_df(self):

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

    def __handle_nans(self, i, i_slices):

        has_nans = False
        for s in i_slices:
            if s.isnull().values.any():
                has_nans = True
                break

        if has_nans:
            if self.nan_filler == 'split':
                logger.warning('Instance %s has NaN values. Splitting on NaNs.', str(i))
                i_slices = self.__split_nans(i_slices)
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
    def __split_nans(slices):
        new_slices = []
        for d in slices:
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
    def __fill_nans(slices, nan_filler, nan_filled_components):
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
    def __split_on_timedelta(df, time_delta):
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
    def __drop_duplicates(df):
        len_check = len(df)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        len_check_dropped = len(df)
        return df, len_check - len_check_dropped
