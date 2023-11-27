__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Extracts arguments, necessary for interpreting a model, from the experiment dictionary.'

from helpers.logger import get_logger
logger = get_logger()

required_args = ('interpretation_method',)
optional_args = ('interpretation_set', 'selection', 'size')


def setup_interpret_args(experiment_dict):

    """ Extracts the relevant arguments for interpreting a model. """

    args = {}
    for a in required_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
        else:
            logger.error('%s not provided in interpret arguments. Cannot interpret a model.', a)
            exit(101)
    for a in optional_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
    if 'interpretation_set' not in args:
        args['interpretation_set'] = 'eval'
    if 'selection' not in args:
        args['selection'] = 'random'
    if 'size' not in args:
        args['size'] = 1
    return args
