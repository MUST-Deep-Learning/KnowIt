__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the base dataset class for Knowit.'

import pandas as pd
import numpy as np

from env.env_paths import dataset_path
from custom_data_converter import convert_df_from_path, convert_df
from helpers.read_configs import load_from_path
from helpers.logger import get_logger
from helpers.read_configs import safe_dump

logger = get_logger()


class Dataset:

    def __init__(self, name: str,
                 task: str,
                 input_components: list,
                 target_components: list,
                 time_delta: pd.Timedelta,
                 io_window: tuple,
                 the_data: dict):

        self.name = name
        self.task = task
        self.input_components = input_components
        self.target_components = target_components
        self.time_delta = time_delta
        self.io_window = io_window
        self.the_data = the_data

        self.instances = list(the_data.keys())
        self.num_instances = len(self.instances)
        self.num_slices = sum([len(the_data[i]) for i in the_data])
        self.num_timesteps = sum([sum([len(s['t']) for s in the_data[i]]) for i in the_data])
        self.num_target_timesteps = sum([sum([np.count_nonzero(~np.isnan(s['y']).any(axis=1))
                                                         for s in the_data[i]])
                                                    for i in the_data])



    @classmethod
    def from_dict(cls, args_dict: str):
        """create class from dictionary """
        return cls(**args_dict)

    @classmethod
    def from_path(cls, path: str, safe_mode=True):
        """create class from path to raw data """
        args_dict = convert_df_from_path(path, fill_nans=True)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        return cls(**args_dict)

    @classmethod
    def from_df(cls, df: pd.DataFrame, safe_mode=True):
        """create class from dataframe """
        args_dict = convert_df(df, fill_nans=True)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        return cls(**args_dict)

    @classmethod
    def from_option(cls, option: str):
        """create class from option """
        try:
            args_dict = load_from_path(dataset_path(option))
        except:
            exit(101)
        return cls(**args_dict)


# DS = Dataset.from_path('/home/tian/postdoc_work/knowit/dummy_raw_data/dummy_zero/dummy_zero.pickle')
# DS = Dataset.from_option('dummy_zero')
