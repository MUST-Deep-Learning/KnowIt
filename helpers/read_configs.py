import os
import yaml
import bz2
import pickle
import _pickle as cPickle
import gzip
import lzma
import shutil
import numpy as np
import csv

from helpers.logger import get_logger

logger = get_logger()


def safe_mkdir(new_dir, safe_mode, overwrite=False):

    dir_exists = os.path.exists(new_dir)

    if not dir_exists:
        try:
            os.makedirs(new_dir)
        except:
            logger.error('Could not create dir %s', new_dir)
            exit(101)
    elif dir_exists and not safe_mode and overwrite:
        logger.warning('Automatically removing dir %s.', new_dir)
        shutil.rmtree(new_dir)
        try:
            os.makedirs(new_dir)
        except:
            logger.error('Could not create dir %s', new_dir)
            exit(101)
    elif dir_exists and safe_mode and overwrite:
        logger.warning('Dir already exists but safe_mode AND overwrite is on %s.', new_dir)
        exit(101)
    else:
        logger.warning('Dir already exists %s. Using as is.', new_dir)



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

    # cfg = dict()
    # for key in cfg_yaml.keys():
    #     if cfg_yaml[key]['value'] == 'None':
    #         cfg[key] = None
    #     else:
    #         cfg[key] = cfg_yaml[key]['value']

    return cfg_yaml


def dict_to_yaml(my_dict, dir_path, file_name):

    path = os.path.join(dir_path, file_name)
    with open(path, 'w+') as handle:
        yaml.dump(my_dict, handle, allow_unicode=True)


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


def safe_copy(destination_dir, origin_path, safe_mode):
    file_name = origin_path.split('/')[-1]
    destination_path = os.path.join(destination_dir, file_name)
    if os.path.exists(destination_path) and safe_mode:
        logger.error('File already exists: %s.',
                     destination_path)
        exit(101)
    else:
        shutil.copyfile(origin_path, destination_path)


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


def load_from_csv(path):
    """
    Load humanly readable data from a given path.
    """

    logger.info('Loading data from file: %s', path)
    data_item = []
    try:
        with open(path, 'r') as theFile:
            reader = csv.DictReader(theFile)
            for line in reader:
                data_item.append(line)
    except:
        logger.error('Could not read file %s', path)
        exit(101)
    logger.info('Values loaded from %s', path)
    return data_item