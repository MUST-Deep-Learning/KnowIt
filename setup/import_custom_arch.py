
from helpers.logger import get_logger
import importlib
import sys
import inspect
import torch.nn as nn


from env.env_paths import custom_arch_path
from helpers.file_dir_procs import safe_copy
logger = get_logger()


def import_custom_arch(path: str, exp_output_dir: str, safe_mode: bool):

    if not path.endswith(".py"):
        logger.error("Import custom arch must end with '.py'")
        exit(101)
    arch_name = path.rstrip(".py").split("/")[-1]

    if complies(path):
        new_path = custom_arch_path(arch_name, exp_output_dir)
        safe_copy(path, new_path, safe_mode)

    logger.info("Imported custom arch %s", arch_name)
    return



def complies(path: str):

    module_name = path.replace("/", ".").replace("\\", ".").rstrip(".py")
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
    required_args = {'x', 'self'}
    missing_args = required_args - set(forward_args)
    if len(missing_args) > 0:
        logger.error("Cannot import custom arch. forward() missing arguments: %s", str(missing_args))
        exit(101)

    return True
