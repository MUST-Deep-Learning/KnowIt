__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Extracts arguments, necessary for importing new datasets, from the experiment dictionary.'

from helpers.logger import get_logger
logger = get_logger()

required_args = ('raw_data_path',)
optional_args = ('base_nan_filler', 'nan_filled_components')


def setup_import_args(experiment_dict, safe_mode):

    """ Extracts the relevant arguments for importing new datasets. """

    args = {}
    for a in required_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
        else:
            logger.error('%s not provided in experiment script. Cannot import dataset.', a)
            exit(101)
    for a in optional_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
    args['safe_mode'] = safe_mode
    return args
