__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the BaseDataset class for Knowit.'

"""
---------------
BaseDataset
---------------

The ``BaseDataset'' represents a dataset that only contains raw features over time in a known format.
These features do not contain duplicates, and are ordered equidistantly according to a 
specific time delta. They are seperated by ``instance'' and by ``slice''. See below. 

An instance is a potential occurrences of a phenomenon measured across time, 
assumed to be independent of other potential occurrences. For example, if two subjects 
are measured / observed at the same time. These two subjects should be stored as separate instances.
The main practical distinction is that no duplicate time steps are allowed within an instance, 
but they are allowed across instances. If no instances are defined, all slices are stored in 
one ``super instance''.

A slice is just a contiguous block of time steps with an exact time delta between each time step.
An instance, therefore, consists of a variable number of slices.

A BaseDataset contains the following variables:
 - name (str):                          A unique identifier used by Knowit to refer to the specific dataset.
 - components (list):                   A list identifying the features that are measured across time.
    It also defines the order of features in the stored data structure.
 - time_delta (object):                 The specific time delta between time steps.
 - instances (list):                    A list of instances in the dataset.
 - the_data (dict):                     The datastructure. Not always stored. 
    Can, alternatively, be loaded from disk.
 - base_nan_filler (str, None):         Identifies the method used to fill NaNs, if applicable.
 - nan_filled_components (list, None):  A list of components that were nan-filled, if applicable.


------------
the_data
------------

The data is stored as a dictionary where each key-value pair represents an instance, 
with the key being the instance name (as defined in the BaseDataset.instances variable) 
and the value being a list of slices. Each slice is a dictionary of two key-value pairs.
    - 't' = array: (n,)
    - 'd' = array: (n, f)
where n is the number of time steps in the current slice and f is the number of components 
(as define and ordered in the BaseDataset.components). The 't' array contains the time steps 
in datetime format, and the 'd' array contains the corresponding feature values.


---------
NOTE! 
---------

To instantiate a BaseDataset with BaseDataset(name), a data option corresponding to `name' needs 
to have been generated before. This can be done with the two class-methods BaseDataset.from_path and 
BaseDataset.from_df, of which the former is a wrapper for the latter. A dataframe needs to be provided 
that has the minimum required meta data (see BaseDataset.__required_base_meta) in its `attrs' attribute.
The BaseDataset module uses the RawDataConverter module to convert this dataframe into a known structure.
See that module for more details.

"""

# external imports
from pandas import DataFrame, Timedelta
from datetime import timedelta

# internal imports
from data.raw_data_coversion import RawDataConverter
from env.env_paths import dataset_path
from helpers.read_configs import load_from_path
from helpers.logger import get_logger
from helpers.read_configs import safe_dump

logger = get_logger()


class BaseDataset:

    def __init__(self, name: str, mem_light=True):
        """ Instantiate a BaseDataset object given the name of an existing dataset option. """

        logger.info('Initializing BaseClass for %s', name)

        self.name = name
        self.__populate_from_option(name)

        if mem_light:
            logger.info('the_data structure not kept in memory.')
            # apply 'memory light' mode so that the_data is not kept in memory
            delattr(self, 'the_data')
        else:
            logger.info('the_data structure is kept in memory.')


    def get_the_data(self):
        """ Return the_data structure from memory if available,
        otherwise load from disk and return. """
        if hasattr(self, 'the_data'):
            return self.the_data
        else:
            return load_from_path(dataset_path(self.name))['the_data']

    @classmethod
    def from_path(cls, path: str, safe_mode: bool = True,
                  base_nan_filler: str = None,
                  nan_filled_components: list = None):
        """ Instantiate a BaseDataset object by first creating the dataset option
        from a given path to a file containing a dataframe with raw data. """
        return cls.from_df(load_from_path(path), safe_mode, base_nan_filler,
                           nan_filled_components)

    @classmethod
    def from_df(cls, df: DataFrame, safe_mode: bool = True,
                base_nan_filler: str = None,
                nan_filled_components: list = None):
        """ Instantiate a BaseDataset object by first creating the dataset option
        from a given dataframe with raw data. """
        args = RawDataConverter(df, cls.__required_base_meta(),
                                base_nan_filler, nan_filled_components).get_new_data()
        safe_dump(args, dataset_path(args['name']), safe_mode)
        return cls(args['name'])

    def __populate_from_option(self, name: str):
        """ Populate the current object with variables from disk
        corresponding to given dataset option (name). """
        args = load_from_path(dataset_path(name))
        for key, value in args.items():
            setattr(self, key, value)

    @staticmethod
    def __required_base_meta():
        """ These are the variables (and their formats) that need to be given in the
        dataframe.attrs dictionary when creating a new dataset option. """
        return {'name': (str,),
                'components': (list,),
                'time_delta': (Timedelta, timedelta)}
