"""
----------------
Action arguments
----------------

The user interacts with KnowIt by sending arguments or paths to external files.
This script contains a function ``setup_relevant_args`` that checks arguments.

 - The presence of required arguments are ensured.
 - Irrelevant arguments are ignored.
 - Default values for optional arguments, that are not given, are provided here.

"""

from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Checks, filters, and adjusts user arguments for various actions in KnowIt.'

# 'required' arguments need to be provided
# 'optional' arguments can be omitted
# 'default' arguments will be used in place of optional if not provided

# internal imports
from helpers.logger import get_logger
logger = get_logger()

arg_dict = {'data_import':  {'required': ('path',),
                                  'optional': ('base_nan_filler',
                                               'nan_filled_components',
                                               'meta'),
                                  'default': {'base_nan_filler': None,
                                              'nan_filled_components': None,
                                              'meta': None}},
            'arch':      {'required': ('task', 'name'),
                          'optional': ('arch_hps',),
                          'default': {'arch_hps': {}}},
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
                                          'seed',
                                          'shuffle_train',
                                          'padding_method',
                                          'batch_sampling_mode',
                                          'succession_length',
                                          'skip_max'),
                             'default': {'seed': 123,
                                         'limit': None,
                                         'min_slice': None,
                                         'scaling_method': 'z-norm',
                                         'scaling_tag': None,
                                         'split_method': 'chronological',
                                         'shuffle_train': True,
                                         'padding_method': 'mean',
                                         'batch_sampling_mode': 0,
                                         'succession_length': 10,
                                         'skip_max': 10}},
            'trainer':      {'required': ('loss_fn',
                                          'optim',
                                          'max_epochs',
                                          'learning_rate',
                                          'task'),
                             'optional': ('lr_scheduler',
                                          'performance_metrics',
                                          'early_stopping_args',
                                          'seed',
                                          'return_final',
                                          'ckpt_mode',
                                          'logger_status',
                                          'optional_pl_kwargs'),
                             'default': {'seed': 123,
                                         'logger_status': False,
                                         'lr_scheduler': None,
                                         'performance_metrics': None,
                                         'early_stopping_args': None,
                                         'return_final': False,
                                         'ckpt_mode': 'min',
                                         'optional_pl_kwargs': {}}},
            'interpreter':    {'required': ('interpretation_method',),
                               'optional': ('interpretation_set',
                                            'selection',
                                            'size',
                                            'multiply_by_inputs',
                                            'seed',
                                            'batch_size',
                                            'rescale_inputs'),  # whether the inputs are rescaled before storage (when applicable)
                             'default': {'interpretation_set': 'valid',
                                         'selection': 'random',
                                         'size': 100,
                                         'multiply_by_inputs': True,
                                         'seed': 123,
                                         'batch_size': None,
                                         'rescale_inputs': True}},
            'predictor':      {'required': ('prediction_set',),
                               'optional': ('rescale_outputs', ), # whether the outputs and targets are rescaled before storage (when applicable)
                               'default': {'rescale_outputs': True}}
            }


def setup_relevant_args(experiment_dict: dict, required_types: tuple | None = None) -> dict:
    """
    Set up relevant arguments from `experiment_dict` based on specified types.

    This function verifies that `experiment_dict` includes required argument types, if provided.
    It then initializes and returns a dictionary of arguments only for valid types by calling
    `setup_type_args`. If an argument type in `experiment_dict` is not recognized, a warning is
    issued.

    Parameters
    ----------
    experiment_dict : dict
        Dictionary containing experimental argument configurations by type.
    required_types : tuple, optional
        Tuple of argument types that must be present in `experiment_dict`. If any required types
        are missing, an error is logged, and execution is stopped.

    Returns
    -------
    dict
        Dictionary of argument configurations for valid types in `experiment_dict`.

    Raises
    ------
    SystemExit
        If required types are specified but not all are found in `experiment_dict`.

    Warnings
    --------
    Logs a warning for any unrecognized argument types in `experiment_dict`.
    """
    if required_types:
        missing_keys = set(required_types) - set(list(experiment_dict.keys()))
        if len(missing_keys) > 0:
            logger.error('Missing arg types: %s', str(missing_keys))
            exit(101)

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


def setup_type_args(experiment_dict: dict, arg_type: str) -> dict:
    """
    Compiles relevant arguments for a given argument type from `experiment_dict`.

    This function retrieves required and optional arguments for the specified `arg_type` from
    `experiment_dict`, utilizing `arg_dict` as a reference for expected arguments. If a required
    argument is missing, an error is logged, and execution is stopped. Optional arguments are added
    if present, and default values are assigned for any optional arguments not specified. Irrelevant
    arguments are logged as warnings.

    Parameters
    ----------
    experiment_dict : dict
        Dictionary containing configurations for different argument types.
    arg_type : str
        The specific type of arguments to compile from `experiment_dict`.

    Returns
    -------
    dict
        Dictionary of relevant arguments for `arg_type`, including required arguments, optional
        arguments (if available), and defaults where applicable.

    Raises
    ------
    SystemExit
        If any required arguments are missing from `experiment_dict` for the specified `arg_type`.

    Warnings
    --------
    Logs a warning for any unrecognized arguments in `experiment_dict` that do not match `arg_dict`.
    """
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
