import os
import yaml
import bz2
import pickle
import _pickle as cPickle
import gzip
import lzma

from helpers.logger import get_logger

logger = get_logger()

def safe_dump(data, path, safe_mode):
    if os.path.exists(path) and safe_mode:
        logger.error('File already exists: %s.',
                     path)
        exit(101)
    else:
        dump_at_path(data, path)


def yaml_to_dict(config_path):

    f = open(config_path, 'r')
    cfg_yaml = None
    try:
        cfg_yaml = yaml.full_load(f)
    except Exception as e:
        logger.error('Error loading config %s:\n%s' % (config_path, str(e)))
        exit(101)
    finally:
        f.close()

    cfg = dict()
    for key in cfg_yaml.keys():
        cfg[key] = cfg_yaml[key]['value']

    return cfg


def dump_at_path(data, path):
    """Dump (and possibly compress) the given data at the given path.
    File extension is inferred from path.

    Args:
        data (Variable):              data to be dumped
        path (str):                   the path (including file extension)

    """

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
        else:
            logger.warning('Unknown file extension, %s, '
                           'storing as uncompressed pickle.', file_ext)
            with open(path, 'wb') as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
    except:
        logger.error('Error saving file %s', path)
        exit(101)


def load_from_path(path):
    """Load (and possibly uncompress) the data at the given path.
    File extension is inferred from path.

    Args:
        path (str):         the path (including file extension)

    Returns:
        data (Variable):   data found at path
    """
    logger = get_logger()
    file_ext = '.pickle'
    try:
        file_ext = '.' + path.split('.')[-1]
    except:
        logger.error('Could not determine file extension from path %s. '
                     'Aborting', path)
        exit(101)
    logger.info('Loading file %s.', path)
    try:
        if file_ext == '.pbz2':
            result = bz2.BZ2File(path, 'rb')
            result = cPickle.load(result)
        elif file_ext == '.pickle' or file_ext == '.pkl':
            with open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.gz':
            with gzip.open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.xz':
            with lzma.open(path, 'rb') as handle:
                result = pickle.load(handle)
        elif file_ext == '.csv':
            with open(path, 'rb') as handle:
                result = pickle.load(handle)
        else:
            logger.error('Unknown file extension, %s, '
                         'could not load. Aborting.', file_ext)
            exit(101)
    except:
        logger.error('Error loading file %s.', path)
        exit(101)

    return result


def save_to_csv(data, path_name, data_format):
    """
    Save data in a humanly readable form to a specified file.
    """
    logger = get_logger()

    logger.info('Saving data to file: %s', path_name)
    try:
        np.savetxt(path_name, data, fmt=data_format)
    except:
        logger.error('Could not write file %s', path_name)
        exit(101)
    logger.info('Values saved as %s', path_name)


def load_from_csv(path_name):
    """
    Load humanly readable data from a given path.
    """
    logger = get_logger()

    logger.info('Loading data from file: %s', path_name)
    data_item = []
    try:
        data_item = np.loadtxt(path_name)
    except:
        logger.error('Could not read file %s', path_name)
        exit(101)
    logger.info('Values loaded from %s', path_name)
    return data_item