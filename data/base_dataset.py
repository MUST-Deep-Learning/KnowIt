"""
-----------
BaseDataset
-----------

The ``BaseDataset`` represents a dataset that only contains raw features over time in a known format.
These features do not contain duplicates, and are ordered equidistantly according to a 
specific time delta. They are seperated by "instance" and by "slice". See below.

An instance is a potential occurrences of a phenomenon measured across time, 
assumed to be independent of other potential occurrences. For example, if two subjects 
are measured / observed at the same time. These two subjects should be stored as separate instances.
The main practical distinction is that no duplicate time steps are allowed within an instance, 
but they are allowed across instances. If no instances are defined, all slices are stored in 
one "super instance".

A slice is just a contiguous block of time steps, within a single instance, 
with an exact time delta between each time step.
An instance, therefore, consists of a variable number of slices.

--------------------
"the_data" structure
--------------------

Along with the required metadata, the data is stored as a dictionary (i.e. "the_data")
where each key-value pair represents an instance, with the key being the instance name
(as defined in the ``BaseDataset.instances`` variable) and the value being a list of slices.
Each slice is a dictionary of two key-value pairs.
    - 't' = array: (n, )
    - 'd' = array: (n, c)
where n is the number of time steps in the current slice and c is the number of components 
(as define and ordered in the ``BaseDataset.components``). The 't' array contains the time steps
in datetime format, and the 'd' array contains the corresponding feature values.

---------------
Dataset options
---------------

BaseDataset is instantiated with a path (i.e. data_path) to an existing dictionary pickled on disk.
This path can be to the custom experiment output directory's "custom_datasets" subdirectory,
or to the default ``KnowIt.default_archs`` directory.

To construct such a dictionary the two class-methods ``BaseDataset.from_path`` and ``BaseDataset.from_df``,
of which the former is a wrapper for the latter, can be used to import external data (in Dataframe format)
and package it into the expected structure.

A dataframe needs to be provided that has the minimum required metadata (see ``BaseDataset._required_base_meta``)
in its "attrs" attribute. The BaseDataset module uses the ``RawDataConverter`` module to convert this dataframe
into a known structure. See that module for more details.
"""
from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the BaseDataset class for KnowIt.'

# external imports
from pandas import DataFrame, Timedelta
from datetime import timedelta

# internal imports
import data
from data.raw_data_coversion import RawDataConverter
from env.env_paths import custom_dataset_path
from helpers.file_dir_procs import load_from_path, safe_dump
from helpers.logger import get_logger
logger = get_logger()


class BaseDataset:
    """This is the BaseDataset class that is used to create the basic Knowit dataset.
    It is the parent class of all other dataset classes and serves to load, clean,
    compile, and store raw data (with critical metadata). It does not carry any
    concept of models or training, only information related to the data is stored.

    Parameters
    ----------
    data_path : str
        The path to the desired dataset.
    mem_light : bool, default=True
        If set to True, the dataset is not kept in memory to save resources.

    Attributes
    ----------
    data_path : str
        The path to the dataset on disk.
    name : str
        The unique identifier used by KnowIt to refer to the specific dataset.
    components : list
        A list of components that were measured across time.
        It also defines the order of features in the stored data structure.
    instances : list
        A list of instances in the dataset.
    time_delta : timedelta
        The specific time delta between time steps.
    the_data : dict | None, default=None
        The datastructure. Not always stored.
    base_nan_filler : str | None, default=None
        Identifies the method used to fill NaNs, if applicable.
    nan_filled_components : list | None, default=None
        A list of components that were nan-filled, if applicable.

    Notes
    -----
        - If mem_light is True, the data structure (the_data) is not kept in memory. This is useful for handling
          large datasets without exhausting memory resources.
        - If mem_light is False, the data structure (the_data) is kept in memory for quicker access at the cost of
          higher memory usage.
    """
    data_path = None
    name = None
    components = None
    instances = None
    time_delta = None
    the_data = None
    base_nan_filler = None
    nan_filled_components = None

    def __init__(self, data_path: str, mem_light: bool = True) -> None:
        self.data_path = data_path
        self._populate_from_path()

        if mem_light:
            logger.info('the_data structure not kept in memory.')
            # apply 'memory light' mode so that the_data is not kept in memory
            delattr(self, 'the_data')
        else:
            logger.info('the_data structure is kept in memory.')

    def _populate_from_path(self) -> None:
        """Loads data from the specified path and assigns the values to
        the corresponding attributes of the current object.
        """
        args = load_from_path(self.data_path)
        for key, value in args.items():
            setattr(self, key, value)

    def get_the_data(self) -> dict:
        """Return the `the_data` structure from memory if available, otherwise load it from disk and return it.

        This method checks if the `the_data` structure is present in the object. If it is, the method returns
        `the_data` directly from memory. If it is not, the method loads `the_data` from the specified disk
        path (`data_path`) and returns it.

        Returns
        -------
        dict[any, list]
            The `the_data` structure, either from memory or loaded from disk.

        Notes
        -----
            - This method ensures that `the_data` is accessed efficiently, either using in-memory data when available
              or loading from disk if not.
        """
        if self.the_data is not None:
            return self.the_data
        else:
            return load_from_path(self.data_path)['the_data']

    @classmethod
    def from_path(cls, path: str, safe_mode: bool,
                  base_nan_filler: str,
                  nan_filled_components: list, meta: dict | None, exp_output_dir: str) -> data.BaseDataset:
        """Instantiate a BaseDataset object by first creating the dataset option from a given path
        to a file containing a dataframe with raw data.

        This method loads data from the specified path (to a .pickle),
        then creates and returns an instance of the BaseDataset class using that data.

        Parameters
        ----------
        path : str
            The file path to the raw data file containing a dataframe.
        safe_mode : bool
            A flag indicating whether to operate in safe mode.
        base_nan_filler : str
            The method used to fill NaNs in the base dataset.
        nan_filled_components : list
            A list of components in the dataset that should have NaNs filled.
        meta: dict | None
            A dictionary containing the required metadata or None. If none, expected file at path.
        exp_output_dir : str
            The directory path for experiment output.

        Returns
        --------
        BaseDataset
            An instance of the BaseDataset class, initialized with the loaded data and specified parameters.
        """
        return cls.from_df(load_from_path(path), safe_mode, base_nan_filler,
                           nan_filled_components, meta, exp_output_dir)

    @classmethod
    def from_df(cls, df: DataFrame, safe_mode: bool,
                base_nan_filler: str | None,
                nan_filled_components: list | None, meta: dict | None, exp_output_dir: str) -> data.BaseDataset:
        """Instantiate a BaseDataset object by first creating the dataset option from a given dataframe with raw data.

        This method converts the provided dataframe into the necessary format and saves it,
        then creates and returns an instance of the BaseDataset class using that data.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing raw data to be converted.
        safe_mode : bool
            A flag indicating whether to operate in safe mode.
        base_nan_filler : str | None
            The method used to fill NaNs in the base dataset.
        nan_filled_components : list | None
            A list of components in the dataset that should have NaNs filled.
        meta: dict | None
            A dictionary containing the required metadata or None. If none, expected to be in df.attrs.
        exp_output_dir : str
            The directory path for experiment output.

        Returns
        -------
        BaseDataset
            An instance of the BaseDataset class, initialized with the processed data and specified parameters.

        Notes
        -----
        See KnowIt.raw_data_conversion for details on base_nan_filler.
        """
        args = RawDataConverter(df, cls._required_base_meta(),
                                base_nan_filler, nan_filled_components, meta).get_new_data()
        data_path = custom_dataset_path(args['name'], exp_output_dir)
        safe_dump(args, data_path, safe_mode)
        return cls(data_path)

    @staticmethod
    def _required_base_meta() -> dict:
        """Return the required metadata fields and their formats for creating a new dataset option.

        This static method defines the necessary variables and their expected types
        that need to be present in the `attrs` dictionary of a DataFrame when
        creating a new dataset option.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the required variables and
            the values are tuples containing the expected types for each variable.
                - 'name': Expected to be a string.
                - 'components': Expected to be a list.
                - 'time_delta': Expected to be either a pandas Timedelta or a Python timedelta.
        """
        return {'name': (str,),
                'components': (list,),
                'time_delta': (Timedelta, timedelta)}
