__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the base dataset class for Knowit.'

import pandas as pd
import numpy as np
import datetime

from env.env_paths import dataset_path
from custom_data_converter import convert_df_from_path, convert_df
from helpers.read_configs import load_from_path
from helpers.logger import get_logger
from helpers.read_configs import safe_dump

logger = get_logger()


class BaseDataset:

    def __init__(self, value, init_method: str,
                 safe_mode: bool = True,
                 fill_nans: bool = True):

        for m in self.required_meta():
            self.__setattr__(m, None)

        self.the_data = None
        self.data_path = None
        self.instances = None
        self.num_instances = None
        self.num_slices = None
        self.num_timesteps = None
        self.num_target_timesteps = None

        if init_method == 'option':
            self.args_from_option(value)
        elif init_method == 'path':
            self.args_from_path(value, safe_mode, fill_nans)
        elif init_method == 'df':
            self.args_from_df(value, safe_mode, fill_nans)
        else:
            self.populate_meta(value)

    def args_from_path(self, path: str, safe_mode, fill_nans):
        """create meta from path to raw data """
        args_dict = convert_df_from_path(path, self.required_meta(), fill_nans=fill_nans)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        self.populate_meta(args_dict)

    def args_from_df(self, df: pd.DataFrame, safe_mode, fill_nans):
        """create meta from dataframe """
        args_dict = convert_df(df, self.required_meta(), fill_nans=fill_nans)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        self.populate_meta(args_dict)

    def args_from_option(self, option: str):
        """create meta from option """
        try:
            args_dict = load_from_path(dataset_path(option))
        except:
            exit(101)
        self.populate_meta(args_dict)

    def populate_meta(self, args):

        for m in self.required_meta():
            self.__setattr__(m, args[m])

        self.the_data = args['the_data']
        self.data_path = dataset_path(args['name'])
        self.instances = list(self.the_data.keys())
        self.num_instances = len(self.instances)
        self.num_slices = sum([len(self.the_data[i]) for i in self.the_data])
        self.num_timesteps = sum([sum([len(s['t']) for s in self.the_data[i]]) for i in self.the_data])
        self.num_target_timesteps = sum([sum([np.count_nonzero(~np.isnan(s['y']).any(axis=1))
                                              for s in self.the_data[i]])
                                         for i in self.the_data])

    @staticmethod
    def required_meta():
        return {'name': (str,), 'task': (str,), 'input_components': (list,),
                'target_components': (list,), 'time_delta': (pd.Timedelta, datetime.timedelta),
                'in_chunk': (list, tuple), 'out_chunk': (list, tuple)}

# ---------------------DEBUGGING----------------------------------------------------------------------------------------

# DS = BaseDataset('/home/tian/postdoc_work/knowit/dummy_raw_data/dummy_zero/dummy_zero.pickle', init_method='path')
# DS = BaseDataset('dummy_zero', init_method='option')
# ping = 0

# ---------------------DEBUGGING----------------------------------------------------------------------------------------
