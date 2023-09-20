__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the base dataset class for Knowit.'

"""
-----------
BaseDataset
-----------

The ``BaseDataset`` represents the bare minimum that Knowit requires for a proper dataset.

It receives and stores the following variables:
- name (str): The name of the new dataset option.
- task (str): The task to be performed on the dataset. (regression, classification)
- input_components (list): A list of feature names corresponding to the model inputs.
- target_components (list): A list of feature names corresponding to the model outputs.
- time_delta (pd.Timedelta or datetime.timedelta): The expected time interval between subsequent time steps.
- in_chunk (list-like): A pair of values (a, b), defining the model input chunk at each time step t. 
    The chunk is defined from t+a to t+b.
- out_chunk (list-like): A pair of values (a, b), defining the model output chunk at each time step t.
- the_data: A dictionary containing the raw data. It is structured as follows:
  - [instance] = A list of slices.
    - [slice] = A dictionary with the following three entries.
      - [x] = An array of input components (time steps x input features)
      - [y] = An array of output components (time steps x output features)
      - [t] = An array of time steps (time steps x 1)
      
The class is initialized with a ``value`` variable and an ``init_method`` variable that defines
what ``value`` represents and how to initialize the class.

- if init_method=`option` then value is assumed to be the dataset option name and 
    the class will be initialized by loading the previously stored variables for the given name.
- if init_method=`path` then value is assumed to be a path to a pd.DataFrame from which to create 
    the variables required for a dataset option.
- if init_method=`df` then value is assumed to be a pd.DataFrame from which to create 
    the variables required for a dataset option.
- if init_method=`direct` then value is assumed to provide the variables required for a 
    dataset option, directly.
    
For the details on the `path` and `df` cases see the custom_data_converter.py script.

-----------------
Instance vs Slice
-----------------

Note that we define data in terms of ``instances`` and ``slices``.

An instance is a set of time-related observations that are independent of other instances.
A slice is a contiguous block of time steps belonging to a single instance.
We make this distinction to facilitate various scenarios requiring different data splits.
For example, if several subjects are measured at the same time, and we wish to split data on subjects, 
then each subject might correspond to an instance.

This structure has some notable implications:
-   Duplicate time steps across instances are allowed, but not within an instance (across slices).
-   A single instance can contain many slices, with gaps in between.
-   Different instances can have different numbers of slices.

Note that Knowit does not require that the user make this distinction.
If no distinction is made between instances then all time steps will be regarded as
belonging to a single instance. (FYI then no duplicate time steps are allowed).

"""

from pandas import DataFrame, Timedelta
from datetime import timedelta
from numpy import count_nonzero, isnan

from env.env_paths import dataset_path
from data.custom_data_converter import convert_df_from_path, convert_df
from helpers.read_configs import load_from_path
from helpers.logger import get_logger
from helpers.read_configs import safe_dump

logger = get_logger()


class BaseDataset:

    def __init__(self, value, init_method: str,
                 safe_mode: bool = True,
                 fill_nans: bool = True):
        """
        Initialize a BaseDataset object with the specified parameters.

        Parameters:
        value (str or Any): The input data or information required to initialize the dataset.
        init_method (str): The initialization method to use, which can be one of the following:
            - 'option': Initialize from previously stored Knowit dataset option.
            - 'path': Initialize from a data file path specified in the 'value' parameter.
            - 'df': Initialize from a DataFrame provided in the 'value' parameter.
            - 'direct': Populate dataset metadata directly using 'value'.
        safe_mode (bool, optional): If True, perform initialization in safe mode (default).
        fill_nans (bool, optional): If True, fill missing values with linear interpolation
            when converting raw data to Knowit dataset option (default).

        Example usage:
        dataset = BaseDataset('/path/to/data.pkl', init_method='path')
        """

        for m in self.required_base_meta():
            self.__setattr__(m, None)

        # Additional variables to be filled dynamically
        self.the_data = None
        self.data_path = None
        self.instances = None
        self.num_target_timesteps = None

        if init_method == 'option':
            self.args_from_option(value)
        elif init_method == 'path':
            self.args_from_path(value, safe_mode, fill_nans)
        elif init_method == 'df':
            self.args_from_df(value, safe_mode, fill_nans)
        elif init_method == 'direct':
            self.populate_meta(value)
        else:
            logger.error('Unknown BaseDataset init method %s.', init_method)
            exit(101)

    def args_from_path(self, path: str, safe_mode: bool, fill_nans: bool):
        """create meta from path to dataframe """
        args_dict = convert_df_from_path(path, self.required_base_meta(), fill_nans=fill_nans)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        self.populate_meta(args_dict)

    def args_from_df(self, df: DataFrame, safe_mode: bool, fill_nans: bool):
        """create meta from dataframe """
        args_dict = convert_df(df, self.required_base_meta(), fill_nans=fill_nans)
        safe_dump(args_dict, dataset_path(args_dict['name']), safe_mode)
        self.populate_meta(args_dict)

    def args_from_option(self, option: str):
        """create meta from option string """
        args_dict = load_from_path(dataset_path(option))
        self.populate_meta(args_dict)

    def populate_meta(self, args: dict):
        """populate the meta with the given argument dictionary"""
        for m in self.required_base_meta():
            self.__setattr__(m, args[m])

        self.the_data = args['the_data']
        self.data_path = dataset_path(args['name'])
        self.instances = list(self.the_data.keys())
        self.num_target_timesteps = sum([sum([count_nonzero(~isnan(s['y']).any(axis=1))
                                              for s in self.the_data[i]])
                                         for i in self.the_data])

        # Adding these variables might be useful in the future
        # self.num_instances = len(self.instances)
        # self.num_slices = sum([len(self.the_data[i]) for i in self.the_data])
        # self.num_timesteps = sum([sum([len(s['t']) for s in self.the_data[i]]) for i in self.the_data])

    @staticmethod
    def required_base_meta():
        """ These define the metadata (and their formats) we expect the user
        to define along with the data when creating a new dataset option."""

        return {'name': (str,), 'task': (str,), 'input_components': (list,),
                'target_components': (list,), 'time_delta': (Timedelta, timedelta),
                'in_chunk': (list, tuple), 'out_chunk': (list, tuple)}


