__author__ = 'tiantheunissen@gmail.com'
__description__ = ('Contains functions for loading, dumping, converting, '
                   'or creating directories an files from local filesystems.')

# external imports
import os
import yaml
import bz2
import pickle
import _pickle as cPickle
import gzip
import lzma
import shutil
import csv

# internal imports
from helpers.logger import get_logger

logger = get_logger()


def proc_dir(new_dir: str, safe_mode: bool = True, overwrite: bool = False) -> None:
    """
    Ensures that a specified directory exists or is recreated based on the provided flags.

    This function checks if the specified directory exists. If it does not exist,
    the directory is created. If it exists, the behavior depends on the `overwrite`
    and `safe_mode` parameters:

    - If `safe_mode` is True and the directory exists, an error is logged, and
      the operation is aborted.
    - If `overwrite` is True and `safe_mode` is False, the existing directory is
      deleted and recreated.

    Parameters
    ----------
    new_dir : str
        The path to the directory to check or create.
    safe_mode : bool, optional
        If True, prevents overwriting an existing directory (default is True).
    overwrite : bool, optional
        If True, allows the existing directory to be overwritten (default is False).

    Raises
    ------
    SystemExit
        If there is an error creating the directory or if safe mode prevents an overwrite.

    Notes
    -----
    Consider prompting the user for confirmation if overwriting an existing directory,
    especially in non-safe mode.
    """
    dir_exists = os.path.exists(new_dir)
    if not dir_exists:
        # create new dir
        try:
            os.makedirs(new_dir)
        except:
            logger.error('Could not create new directory: %s', new_dir)
            exit(101)
    elif not overwrite:
        # do nothing, use existing directory
        # logger.warning('Directory already exists: %s. Using as is.', new_dir)
        pass
    elif safe_mode:
        # error; the dir exists and wants to be overwritten but safe_mode
        logger.error('Directory will be overwritten but safe_mode=True: %s.', new_dir)
        exit(101)
    else:
        # overwrite; dir exists, safe_mode is off, and overwrite is on.
        # TODO: Consider prompting user
        logger.warning('Recreating directory: %s.', new_dir)
        shutil.rmtree(new_dir)
        try:
            os.makedirs(new_dir)
        except:
            logger.error('Could not recreate directory: %s', new_dir)
            exit(101)


def safe_dump(data: any, path: str, safe_mode: bool) -> None:
    """
    Safely dumps data to a specified file path with an option to protect existing files.

    This function checks if a file already exists at the given path and, if `safe_mode` is enabled,
    logs an error and exits to prevent overwriting. If `safe_mode` is not enabled or the file does not exist,
    the function proceeds to write data to the specified path.

    Parameters
    ----------
    data : any
        The data to be written to the file.
    path : str
        The target file path for saving data.
    safe_mode : bool
        If True, prevents overwriting an existing file by checking if the path exists.
        If False, allows overwriting any existing file at the specified path.

    Raises
    ------
    SystemExit
        If `safe_mode` is True and a file already exists at the given path, exits with code 101.

    """
    if os.path.exists(path) and safe_mode:
        logger.error('File already exists: %s.',
                     path)
        exit(101)
    else:
        dump_at_path(data, path)


def safe_copy(path: str, new_path: str, safe_mode: bool) -> None:
    """
    Copies a file (or directory) to a new location with optional overwrite protection.

    This function attempts to copy a file from `path` to `new_path`. If `safe_mode` is enabled and
    a file already exists at `new_path`, an error is logged, and the function exits to prevent overwriting.
    Otherwise, it proceeds with the copy operation.

    Parameters
    ----------
    path : str
        The path to the source file to be copied.
    new_path : str
        The destination path where the file should be copied.
    safe_mode : bool
        If True, prevents overwriting an existing file at `new_path`.
        If False, allows overwriting any existing file at the specified path.

    Raises
    ------
    SystemExit
        If `safe_mode` is True and a file already exists at `new_path`, exits with code 101.

    """
    if os.path.exists(new_path) and safe_mode:
        logger.error('File already exists: %s.',
                     path)
        exit(101)
    else:
        if os.path.isfile(path):
            shutil.copyfile(path, new_path)
        elif os.path.isdir(path):
            shutil.copytree(path, new_path)


def yaml_to_dict(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    This function opens a specified YAML file, attempts to parse its contents, and returns
    the resulting dictionary. If an error occurs during loading, it logs the error and exits
    with code 101.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration data loaded from the YAML file.

    Raises
    ------
    SystemExit
        If an error occurs during file reading or parsing, exits with code 101.

    """
    try:
        with open(config_path, 'r') as f:
            cfg_yaml = yaml.full_load(f)
    except Exception as e:
        logger.error('Error loading config %s:\n%s' % (config_path, str(e)))
        exit(101)

    return cfg_yaml


def dump_at_path(data: any, path: str) -> None:
    """
    Dumps (and optionally compresses) data to a specified path based on the file extension.

    This function writes the provided `data` to the specified `path` in a format inferred
    from the file extension. Supported formats include `.pbz2`, `.pickle`, `.gz`, `.xz`,
    `.yaml`, and `.yml`. If an unrecognized file extension is provided, the data is stored
    as an uncompressed pickle file by default.

    Parameters
    ----------
    data : Any
        The data to be saved. Data should match the format expected by the file extension:
        - `.yaml` and `.yml` assume data is a dictionary (for YAML serialization).
        - All other extensions allow serialized Python objects.
    path : str
        The full path (including file extension) where data should be saved.

    Raises
    ------
    SystemExit
        If there is an error during saving,
        the function logs an error and exits with code 101.

    Notes
    -----
    The function supports the following file extensions:
    - `.pbz2` : BZ2 compressed pickle
    - `.pickle` : Standard pickle
    - `.gz` : GZIP compressed pickle
    - `.xz` : LZMA compressed pickle
    - `.yaml` / `.yml` : YAML format (expects a dictionary)
    - Unknown extensions will trigger a warning and default to an uncompressed pickle format.
    """

    # TODO: Can be made more robust

    logger = get_logger()
    file_ext = '.pickle'
    try:
        file_ext = '.' + path.split('.')[-1]
    except:
        logger.error('Could not determine file extension from path %s. '
                     'Aborting', path)
        exit(101)
    logger.info('Saving file %s', path)
    try:
        if file_ext == '.pbz2':
            with bz2.BZ2File(path, 'w') as f:
                cPickle.dump(data, f)
        elif file_ext == '.pickle':
            with open(path, 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
        elif file_ext == '.gz':
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        elif file_ext == '.xz':
            with lzma.open(path, 'wb') as f:
                pickle.dump(data, f)
        elif file_ext in ('.yaml', '.yml'):
            # assumes data is a dict?
            with open(path, 'w+') as handle:
                yaml.dump(data, handle, allow_unicode=True)
        else:
            logger.warning('Unknown file extension, %s, '
                           'storing as uncompressed pickle.', file_ext)
            with open(path, 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
    except:
        logger.error('Error saving file %s', path)
        exit(101)


def load_from_path(path: str) -> any:
    """
    Load data from a specified file path, with optional decompression based on file extension.

    This function infers the file format from the file extension and loads data accordingly.
    Supported formats include `.pbz2`, `.pickle` (or `.pkl`), `.gz`, `.xz`, and `.csv`.

    Args:
        path (str): Path to the file (including the extension).

    Returns:
        Any: Loaded data from the specified path.

    Raises:
        SystemExit: If the file extension is unrecognized or an error occurs during loading,
                    the function logs an error and exits with code 101.
    """
    logger = get_logger()
    file_ext = os.path.splitext(path)[1]

    try:
        if file_ext == '.pbz2':
            with bz2.BZ2File(path, 'rb') as f:
                result = pickle.load(f)
        elif file_ext in ('.pickle', '.pkl'):
            with open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.gz':
            with gzip.open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.xz':
            with lzma.open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.csv':
            result = []
            with open(path, 'r') as handle:
                reader = csv.DictReader(handle)
                result.extend(reader)  # Adds each line in reader as a dict to the result list
        else:
            logger.error('Unknown file extension: %s. Unable to load file. Aborting.', file_ext)
            exit(101)
    except Exception as e:
        logger.error('Error loading file %s: %s', path, e)
        exit(101)

    return result


def safe_dump_parquet(path: str, safe_mode: bool, data_package: any,
                      partitioned_on: list, engine: str = "pyarrow") -> None:
    """Safely save a dataset as a partitioned Parquet file.

    This function checks if the specified path already exists before saving the dataset.
    If `safe_mode` is enabled and the path exists, an error is logged, and execution stops.
    If `safe_mode` is disabled and the path exists, a warning is logged, and the existing
    directory is removed before saving the new dataset.

    Parameters
    ----------
    path : str
        The file path where the Parquet dataset will be stored.
    safe_mode : bool
        If True, prevents overwriting an existing dataset by exiting with an error.
        If False, overwrites the existing dataset after logging a warning.
    data_package : any
        The dataset to be saved as a Parquet file. Must be compatible with `.to_parquet()`.
    partitioned_on : list
        A list of column names to use for partitioning the dataset.
    engine : str, optional
        The Parquet engine to use for saving the dataset (default is "pyarrow").

    Raises
    ------
    SystemExit
        If `safe_mode` is True and the specified path already exists.

    Notes
    -----
    - If `safe_mode` is enabled and the path exists, the function exits with error code 101.
    - If `safe_mode` is disabled and the path exists, the existing directory is deleted.
    - The dataset is saved using the specified engine and partitioning scheme.
    """

    if os.path.exists(path) and safe_mode:
        logger.error('Base dataset package already exists at {}'.format(path))
        exit(101)
    elif os.path.exists(path) and not safe_mode:
        logger.warning('Base dataset package already exists at {}, overwriting.'.format(path))
        shutil.rmtree(path)
    data_package.to_parquet(path, engine=engine, partition_cols=partitioned_on)