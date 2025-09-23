"""Functions used to import custom architectures. """

__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains functions used during the import of custom architectures.'

from helpers.logger import get_logger
import importlib
import sys
import inspect
import torch.nn as nn


from env.env_paths import custom_arch_path, arch_name
from helpers.file_dir_procs import safe_copy
logger = get_logger()


def import_custom_arch(path: str, exp_output_dir: str, safe_mode: bool) -> None:
    """
    Imports a custom architecture file, validates its extension, and copies it
    to a specified directory if it complies with given criteria.

    Parameters
    ----------
    path : str
        The file path of the custom architecture script. Must end with '.py'.
    exp_output_dir : str
        The directory where the custom architecture file will be saved if valid.
    safe_mode : bool
        Flag indicating whether to overwrite existing files in the target directory.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If `path` does not end with '.py' or if copying fails.

    Logs
    ----
    - Logs an error if the file path does not end with '.py'.
    - Logs the successful import of the custom architecture.

    Notes
    -----
    - This function assumes the existence of external functions `complies`,
      `custom_arch_path`, and `safe_copy` to verify, format the file path,
      and handle file copying with overwrite safety.
    """
    if not path.endswith(".py"):
        logger.error("Import custom arch must end with '.py'")
        exit(101)
    arch_name = path.rstrip(".py").split("/")[-1]

    if complies(path):
        new_path = custom_arch_path(arch_name, exp_output_dir)
        safe_copy(path, new_path, safe_mode)

    logger.info("Imported custom arch %s", arch_name)
    return


def complies(path: str) -> bool:
    """
    Validates a custom model architecture by importing the file and performing compliance checks.

    This function dynamically loads a Python file specified by `path` and verifies that it meets
    specific criteria for compatibility with KnowIt. The checks include inheritance, method
    definitions, attribute presence, and argument requirements.

    Parameters
    ----------
    path : str
        The path to the Python file containing the custom architecture definition.

    Returns
    -------
    bool
        True if the custom architecture file passes all compliance checks.

    Raises
    ------
    SystemExit
        If the custom architecture does not meet any of the required compliance criteria.

    Notes
    -----
    - Class Inheritance: The file must contain a class `Model` that inherits from `torch.nn.Module`.
    - Method Presence: The `Model` class must define a `forward()` method.
    - Metadata: The file must contain:
        - `available_tasks`: a list indicating the supported tasks for the model.
        - `HP_ranges_dict`: a dictionary defining the hyperparameter ranges for the model.
    - Constructor Arguments:
        - `__init__` of `Model` must accept specific arguments: `task_name`, `input_dim`, `output_dim`.
        - `HP_ranges_dict` must include all arguments in `__init__` except `task_name`, `input_dim`,
          `output_dim`, and `self`.
    - Default Values: `__init__` must define default values for any optional arguments.
    - Forward Arguments: The `forward()` method must accept exactly two arguments, one of which must
      be `self`.
    """
    module_name = arch_name(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, 'Model')

    # check if Model inherits from pytorch
    if not issubclass(cls, nn.Module):
        logger.error("Cannot import custom arch. Model must inherit from nn.Module.")
        exit(101)

    # check if Model has a forward function
    if not hasattr(cls, 'forward'):
        logger.error("Cannot import custom arch. Model must have a forward() method.")
        exit(101)

    # check that additional metadata is provided
    try:
        available_tasks = getattr(module, 'available_tasks')
    except:
        logger.error("Cannot import custom arch. Model must have available_tasks list.")
        exit(101)
    try:
        HP_ranges_dict = getattr(module, 'HP_ranges_dict')
    except:
        logger.error("Cannot import custom arch. Model must have HP_ranges_dict dictionary.")
        exit(101)

    # check init arguments
    init_args = set(inspect.getfullargspec(cls.__init__).args)
    required_args = {'task_name', 'input_dim', 'output_dim', 'self'}
    missing_args = required_args - init_args
    if len(missing_args) > 0:
        logger.error("Cannot import custom arch. __init__() missing arguments: %s", str(missing_args))
        exit(101)
    missing_ranges = init_args - set(HP_ranges_dict.keys()) - required_args
    if len(missing_ranges) > 0:
        logger.error("Cannot import custom arch. HP_ranges_dict missing arguments: %s", str(missing_ranges))
        exit(101)

    # check that default args are provided
    other_args = init_args - required_args
    signature = inspect.signature(cls.__init__)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    missing_defaults = other_args - set(defaults.keys())
    if len(missing_defaults) > 0:
        logger.error("Cannot import custom arch. Default values for %s missing.", str(missing_defaults))
        exit(101)

    # check forward arguments
    forward_args = inspect.getfullargspec(cls.forward).args
    if 'self' not in forward_args or (len(forward_args) != 2 and len(forward_args) != 3):
        logger.error("Cannot import custom arch. forward() function must receive two or three arguments "
                     "of which self is one, got %s", str(forward_args))
        exit(101)

    return True
