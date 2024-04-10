__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Checks, filters, and adjusts user arguments for various actions in KnowIt.'

# internal imports
from helpers.logger import get_logger
logger = get_logger()

arg_dict = {'import':  {'required': ('path',),
                             'optional': ('base_nan_filler',
                                          'nan_filled_components')},
            'analyze':      {'required': (),
                             'optional': ()},
            'data':         {'required': ('name',
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
                                          'learning_rate',
                                          'state'),
                             'optional': ('lr_scheduler',
                                          'clip_gradients',
                                          'performance_metrics',
                                          'early_stopping_args',
                                          'seed',
                                          'return_final',
                                          'model_selection_mode'),
                             'default': {'seed': 123}},
            'tune':         {'required': (),
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
            'predict':      {'required': ('prediction_set',),
                             'optional': ()}
            }


def setup_relevant_args(experiment_dict, action, safe_mode=None):
    ret_args = {}

    for a in arg_dict[action]['required']:
        if a in experiment_dict.keys():
            ret_args[a] = experiment_dict[a]
        else:
            logger.error('Argument \'%s\' not provided. Compile arguments for \'%s\'.', a, action)
            exit(101)

    for a in arg_dict[action]['optional']:
        if a in experiment_dict.keys():
            ret_args[a] = experiment_dict[a]

    if 'default' in arg_dict[action].keys():
        for a in arg_dict[action]['default']:
            if a not in ret_args:
                ret_args[a] = arg_dict[action]['default'][a]

    if safe_mode:
        ret_args['safe_mode'] = safe_mode

    return ret_args
