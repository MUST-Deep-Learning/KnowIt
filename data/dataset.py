__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the base dataset class for Knowit.'

import pandas

from custom_data_converter import extract_df


class Dataset:

    def __init__(self, name: str,
                 task: str,
                 input_components: list,
                 target_components: list,
                 time_delta: pandas.Timedelta,
                 io_window: tuple,
                 raw_data: dict):

        self.name = name
        self.task = task
        self.input_components = input_components
        self.target_components = target_components
        self.time_delta = time_delta
        self.io_window = io_window
        self.raw_data = raw_data

    @classmethod
    def from_dict(cls, args_dict):
        """create class from dictionary """
        return cls(**args_dict)

    @classmethod
    def from_path(cls, path):
        """create class from yaml """
        args_dict = extract_df(path)
        return cls(**args_dict)
