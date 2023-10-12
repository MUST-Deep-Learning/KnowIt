__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the base dataset class for Knowit.'

from pandas import DataFrame, Timedelta
from datetime import timedelta

from data.raw_data_coversion import RawDataConverter
from env.env_paths import dataset_path
from helpers.read_configs import load_from_path
from helpers.logger import get_logger
from helpers.read_configs import safe_dump

logger = get_logger()


class BaseDataset:

    def __init__(self, name, mem_light=True):
        self.__from_option(name)
        if mem_light:
            delattr(self, 'the_data')

    @classmethod
    def from_path(cls, path: str, safe_mode: bool = True,
                  base_nan_filler: str = None,
                  nan_filled_components: list = None):
        """ Init BaseDataset from path to a file containing a dataframe with raw data. """
        return cls.from_df(load_from_path(path), safe_mode, base_nan_filler,
                           nan_filled_components)

    @classmethod
    def from_df(cls, df: DataFrame, safe_mode: bool = True,
                base_nan_filler: str = None,
                nan_filled_components: list = None):
        """ Init BaseDataset from a dataframe with raw data. """
        args = RawDataConverter(df, cls.__required_base_meta(), base_nan_filler, nan_filled_components).get_new_data()
        args['data_path'] = dataset_path(args['name'])
        safe_dump(args, dataset_path(args['name']), safe_mode)
        return cls(args['name'])

    def __from_option(self, option_tag: str):
        """ Init BaseDataset from option_tag (dataset name)."""
        args = load_from_path(dataset_path(option_tag))
        for key, value in args.items():
            setattr(self, key, value)

    @staticmethod
    def __required_base_meta():

        return {'name': (str,),
                'components': (list,),
                'time_delta': (Timedelta, timedelta)}