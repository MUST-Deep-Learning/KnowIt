"""
-----------
BaseDataset
-----------

The ``BaseDataset`` represents a dataset that only contains raw features
over time (i.e. components) in a known format.
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

------------
Date storage
------------

The `BaseDataset` is stored on disk as a partitioned parquet dataset (package)
along with a pickled dictionary of metadata. The stored metadata includes the name
and structure of the dataset along with other characteristics like the
time delta.

Along with the metadata, the data package itself is stored as a directory containing
a set of instances and underlying slices.
    <dataset name>
    ├── <instance=0>
    │   ├── <slice=0>
    │   │   └── file.parquet
    │   ├── ...
    │   └── <slice=B>
    │       └── file.parquet
    ├── ...
    └── <instance=A>
       ├── <slice=0>
       ├── ...
       └── <slice=C>

Each parquet file represents a dataframe for which the rows are:
    - Time indexed
    - Contiguous
    - Chronologically ordered
    - Without duplicate entries.
The columns in the dataframe are:
    - A set of columns representing the components, in the order that they are defined in the metadata.
    - A column indicating the instance of the current time step.
    - A column indicating the slice of the current time step.

---------------
Dataset options
---------------

``BaseDataset`` is instantiated with a path to the metadata (i.e. meta_path) and a path to the
data package (i.e. package_path).

To construct the above-mentioned variables, the ``BaseDataset.from_path`` and ``BaseDataset.from_df``,
of which the former is a wrapper for the latter, can be used to import external data (in Dataframe format)
and package it into the expected structure. A dataframe needs to be provided that has the minimum required
metadata in its "attrs" attribute.
The BaseDataset module uses the ``RawDataConverter`` module to convert this dataframe
into a known structure. See that module for more details.
"""
from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the BaseDataset class for KnowIt.'

# external imports
from pandas import DataFrame, Timedelta, read_parquet
from datetime import timedelta
from functools import lru_cache

# internal imports
import data
from data.raw_data_coversion import RawDataConverter
from env.env_paths import custom_dataset_meta_path, custom_dataset_package_path
from helpers.file_dir_procs import load_from_path, safe_dump, safe_dump_parquet
from helpers.logger import get_logger
logger = get_logger()


class BaseDataset:
    """This is the BaseDataset class. It is the parent class of all other dataset classes
    and serves to load, clean, compile, and store raw data (with critical metadata).
    It does not carry any concept of models or training, only information related to the data.

    Parameters
    ----------
    meta_path : str
        The path to the desired dataset metadata. The path should point to a pickle file.
    package_path : str
        The path to the desired dataset package. The path should point to a directory containing the
        partitioned parquet.

    Attributes
    ----------
    meta_path : str
        The path to the metadata.
    package_path : str
        The path to the data package.
    name : str
        The unique identifier used by KnowIt to refer to the specific dataset.
    components : list
        A list of components that were measured across time.
        It also defines the order of features in the stored data structure.
    instance_names : dict[int, Any]
        A dictionary containing the original instance names.
        The key is an integer identifier and the value is the original name for the instance at import time.
    data_structure : dict[int, dict]
        A dictionary containing the data structure.
        Each key indicates an instance,
        each value indicates a dictionary of slices (for which the values are the number of prediction points in the slice).
    time_delta : timedelta
        The specific time delta between time steps.
    base_nan_filler : str | None, default=None
        Identifies the method used to fill NaNs, if applicable, at import time.
    nan_filled_components : list | None, default=None
        A list of components that were nan-filled, if applicable, at import time.
    """
    meta_path = None
    package_path = None
    name = None
    components = None
    instance_names = None
    time_delta = None
    base_nan_filler = None
    nan_filled_components = None
    data_structure = None

    def __init__(self, meta_path: str, package_path: str) -> None:
        self.meta_path = meta_path
        self._populate_from_path()
        self.package_path = package_path

    def _populate_from_path(self) -> None:
        """Loads data from the metadata path and assigns the values to
        the corresponding attributes of the current object.
        """
        args = load_from_path(self.meta_path)
        for key, value in args.items():
            setattr(self, key, value)

    def get_extractor(self) -> DataExtractor:
        """Return a DataExtractor object corresponding to the current BaseDataset.

        This object can be used to extract particular portions of the BaseDataset from disk.

        Returns
        -------
        DataExtractor
            The DataExtractor object.

        """
        return DataExtractor(self.package_path, self.components, self.instance_names, self.data_structure)

    @classmethod
    def from_path(cls, exp_output_dir: str,
                  path: str, safe_mode: bool,
                  base_nan_filler: str | None,
                  nan_filled_components: list | None,
                  meta: dict | None) -> data.BaseDataset:
        """Instantiate a BaseDataset object by first creating the dataset from a given path
        to a file containing a dataframe with raw data.

        This method loads data from the specified path (to a .pickle),
        then creates and returns an instance of the BaseDataset class using that data.

        Parameters
        ----------
        exp_output_dir : str
            The directory path for experiment output.
        path : str
            The file path to the raw data file containing a dataframe.
        safe_mode : bool
            A flag indicating whether to operate in safe mode.
        base_nan_filler : str | None
            The method used to fill NaNs in the base dataset.
        nan_filled_components : list | None
            A list of components in the dataset that should have NaNs filled.
        meta: dict | None
            A dictionary containing the required metadata or None.
            If None, metadata should be provided in file at path.

        Returns
        --------
        BaseDataset
            An instance of the BaseDataset class, initialized with the loaded data and specified parameters.
        """
        return cls.from_df(load_from_path(path), exp_output_dir, safe_mode, base_nan_filler,
                           nan_filled_components, meta)

    @classmethod
    def from_df(cls, df: DataFrame,
                exp_output_dir: str,
                safe_mode: bool,
                base_nan_filler: str | None,
                nan_filled_components: list | None,
                meta: dict | None) -> data.BaseDataset:
        """Instantiate a BaseDataset object by first creating the dataset option from a given dataframe with raw data.

        This method converts the provided dataframe into the necessary format and saves it,
        then creates and returns an instance of the BaseDataset class using that data.

        Parameters
        ----------
        df : DataFrame
            The dataframe containing raw data to be converted.
        exp_output_dir : str
            The directory path for experiment output.
        safe_mode : bool
            A flag indicating whether to operate in safe mode.
        base_nan_filler : str | None
            The method used to fill NaNs in the base dataset.
        nan_filled_components : list | None
            A list of components in the dataset that should have NaNs filled.
        meta: dict | None
            A dictionary containing the required metadata or None.
            If None, expected to be in df.attrs.

        Returns
        -------
        BaseDataset
            An instance of the BaseDataset class, initialized with the processed data and specified parameters.

        Notes
        -----
        See ``KnowIt.raw_data_conversion`` for details on base_nan_filler.
        """

        # use RawDataConverter to ge the packaged data
        meta_data, data_package = RawDataConverter(df, cls._required_base_meta(),
                                                   base_nan_filler, nan_filled_components,
                                                   meta).get_new_data()

        # dump the metadata
        data_meta_path = custom_dataset_meta_path(meta_data['name'], exp_output_dir)
        safe_dump(meta_data, data_meta_path, safe_mode)

        # dump the data package
        data_package_path = custom_dataset_package_path(meta_data['name'], exp_output_dir)
        safe_dump_parquet(data_package_path, safe_mode, data_package, ["instance", "slice"])

        return cls(data_meta_path, data_package_path)

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


# Note: Going with pandas for DataExtractor.
# If the dataset becomes so big that even a slice cannot fit into memory,
# we will need to consider something like dask:
# from dask.dataframe import read_parquet
# df = read_parquet(self.package_path, filters=[("instance", "==", key), ("slice", "==", 0)],
#                   engine="pyarrow")
# df = df.compute()


class DataExtractor:
    """
    A class for efficiently extracting specific portions of a partitioned Parquet dataset from disk.
    Uses an LRU cache to optimize repeated reads.

    Parameters
    ----------
    package_path : str
        Path to the directory containing the partitioned Parquet dataset.
    components : list
        List of column names to be read from the dataset.
    instance_names : dict
        Dictionary mapping instance identifiers to human-readable names (for reference).
    data_structure : dict
        Dictionary describing the dataset's structure (e.g., metadata about partitions).
    engine : str, default="pyarrow"
        Parquet reading engine to use ('pyarrow' or 'fastparquet').
    cache_size : int, default=100
        Maximum number of recently read partitions to store in the cache.
    """

    def __init__(self, package_path: str, components: list,
                 instance_names: dict, data_structure: dict,
                 engine: str = "pyarrow", cache_size: int = 100) -> None:
        self.package_path = package_path
        self.components = components
        self.engine = engine
        self.cache_size = cache_size

        # Convenience attributes
        self.instance_names = instance_names
        self.data_structure = data_structure

        # Cache the read parquet function
        self._cached_read_parquet = lru_cache(maxsize=cache_size)(self._read_parquet)

    def _read_parquet(self, i, s=None):
        """
        Reads a partition of the dataset from disk based on the given instance and slice values.

        Parameters
        ----------
        i : int or str
            The instance identifier to filter the data.
        s : int, optional
            The slice identifier to further filter the data (default is None).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the requested subset of the dataset.
        """
        filters = [("instance", "==", i)]
        columns = self.components + ['slice']
        if s is not None:
            filters.append(("slice", "==", s))
            del columns[-1]
        return read_parquet(self.package_path, columns=columns,
                            filters=filters, engine=self.engine)

    def instance(self, i):
        """
        Retrieves all data for a specific instance.

        Parameters
        ----------
        i : int or str
            The instance identifier.

        Returns
        -------
        pd.DataFrame
            The dataset subset corresponding to the given instance.
        """
        return self._cached_read_parquet(i)

    def slice(self, i, s):
        """
        Retrieves data for a specific instance and slice.

        Parameters
        ----------
        i : int or str
            The instance identifier.
        s : int
            The slice identifier.

        Returns
        -------
        pd.DataFrame
            The dataset subset corresponding to the given instance and slice.
        """
        return self._cached_read_parquet(i, s)

    def time_step(self, i, s, t):
        """
        Retrieves a single time step from a given instance and slice.

        Parameters
        ----------
        i : int or str
            The instance identifier.
        s : int
            The slice identifier.
        t : int
            The time step index.

        Returns
        -------
        pd.Series
            A row of the dataset corresponding to the given instance, slice, and time step.
        """
        return self._cached_read_parquet(i, s).iloc[t]

    def time_block(self, i, s, block):
        """
        Retrieves a block of time steps from a given instance and slice.

        Parameters
        ----------
        i : int or str
            The instance identifier.
        s : int
            The slice identifier.
        block : tuple of (int, int)
            A tuple defining the start and end indices for the time step range.

        Returns
        -------
        pd.DataFrame
            A subset of the dataset containing the requested range of time steps.
        """
        return self._cached_read_parquet(i, s).iloc[block[0]:block[1]]


