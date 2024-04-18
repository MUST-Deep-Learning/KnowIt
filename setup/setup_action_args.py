__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Checks, filters, and adjusts user arguments for various actions in KnowIt.'

"""
------------------
Action arguments
------------------

The user interacts with KnowIt by sending a combination of action key words, arguments, or paths to external files.
This script contains a function ``setup_relevant_args`` that 




"""

# 'required' arguments need to be provided
# 'optional' arguments can be omitted
# 'default' arguments will be used in place of optional if not provided

# internal imports
from helpers.logger import get_logger
logger = get_logger()

arg_dict = {'importer':  {'required': ('path',),
                          'optional': ('base_nan_filler',
                                       'nan_filled_components')},
            'id':       {'required': ('experiment_name',
                                      'model_name'),
                         'optional': ()},
            'analyzer':      {'required': (),
                              'optional': ()},
            'arch':      {'required': ('task', 'name'),
                          'optional': ('arch_hps',)},
            'data':         {'required': ('name',
                                          'task',
                                          'in_components',
                                          'out_components',
                                          'in_chunk',
                                          'out_chunk',
                                          'split_portions',
                                          'batch_size'),
                             'optional': ('limit',
                                          'min_slice',
                                          'scaling_method',
                                          'scaling_tag',
                                          'split_method',
                                          'seed'),
                             'default': {'seed': 123}},
            'trainer':      {'required': ('loss_fn',
                                          'optim',
                                          'max_epochs',
                                          'learning_rate'),
                             'optional': ('lr_scheduler',
                                          'performance_metrics',
                                          'early_stopping_args',
                                          'seed',
                                          'return_final',
                                          'model_selection_mode',
                                          'mute_logger',
                                          'optional_pl_kwargs',
                                          'state'),
                             'default': {'seed': 123,
                                         'state': 'new',
                                         'mute_logger': False}},
            'tuner':         {'required': (),
                              'optional': ()},
            'interpret':    {'required': ('interpretation_method',),
                             'optional': ('interpretation_set',
                                          'selection',
                                          'size',
                                          'multiply_by_inputs'),
                             'default': {'interpretation_set': 'eval',
                                         'selection': 'random',
                                         'size': 1,
                                         'multiply_by_inputs': True,
                                         'seed': 123}},
            'predictor':      {'required': ('prediction_set',),
                             'optional': ()}
            }


def setup_relevant_args(experiment_dict):

    valid_arg_types = list(arg_dict)
    ret_args = {}
    for arg_type in experiment_dict:
        if arg_type in valid_arg_types:
            ret_args[arg_type] = setup_type_args(experiment_dict, arg_type)
        else:
            logger.warning('Unknown arg_type=%s, ignoring. Only valid arg_types: %s',
                           arg_type,
                           str(valid_arg_types))

    return ret_args


def setup_type_args(experiment_dict, arg_type):
    ret_args = {}

    # keep all required arguments
    for a in arg_dict[arg_type]['required']:
        if a in experiment_dict[arg_type].keys():
            ret_args[a] = experiment_dict[arg_type][a]
        else:
            logger.error('Argument \'%s\' not provided. Cannot compile arguments for \'%s\'.', a, arg_type)
            exit(101)

    # keep all optional arguments if available
    for a in arg_dict[arg_type]['optional']:
        if a in experiment_dict[arg_type].keys():
            ret_args[a] = experiment_dict[arg_type][a]

    # add defaults if optionals not provided and defaults available
    if 'default' in arg_dict[arg_type].keys():
        for a in arg_dict[arg_type]['default']:
            if a not in ret_args:
                ret_args[a] = arg_dict[arg_type]['default'][a]

    # warn if nonsense arguments found
    nonsense_args = set(list(experiment_dict[arg_type].keys())) - set(list(ret_args.keys()))
    if len(nonsense_args) > 0:
        logger.warning('Ignoring irrelevant arguments: %s', str(nonsense_args))

    return ret_args
