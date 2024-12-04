""" Defines many different directories based on experiment and KnowIt location. """

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains functions that dynamically return KnowIt environment paths.'

# DIR = PATH TO A DIRECTORY
# PATH = PATH TO A FILE

# NOTE: PATHs don't use the safe_mode and overwrite arguments.
# The relevant code should use safe_dump and safe_copy for that.

# external imports
import os

# internal imports
from env.env_user import (default_dataset_dir, default_archs_dir)
from helpers.file_dir_procs import proc_dir
from helpers.logger import get_logger

logger = get_logger()


# ----------------------------------------------------------------------------------------------------------------------
#   EXPERIMENTS
# ----------------------------------------------------------------------------------------------------------------------

def root_exp_dir(exp_output_dir: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the root experiment output directory. """
    proc_dir(exp_output_dir, safe_mode, overwrite)
    return exp_output_dir


# ----------------------------------------------------------------------------------------------------------------------
#   DATASETS
# ----------------------------------------------------------------------------------------------------------------------


def dataset_path(name: str):
    """ Returns the dataset path for the given dataset name from the default datasets directory."""
    return os.path.join(default_dataset_dir, name + '.pickle')


def custom_dataset_dir(exp_output_dir: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the dataset directory from the given experiment path.
        Note that the directory is created if it does not exist. """
    proc_dir(os.path.join(exp_output_dir, 'custom_datasets'), safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(exp_output_dir, 'custom_datasets')


def custom_dataset_path(name: str, exp_output_dir: str):
    """ Returns the dataset path for the given dataset name from the given experiment path."""
    return os.path.join(custom_dataset_dir(exp_output_dir), name + '.pickle')

# ----------------------------------------------------------------------------------------------------------------------
#   ARCHITECTURES
# ----------------------------------------------------------------------------------------------------------------------


def arch_path(name: str):
    """ Returns the architecture path for the given architecture name from the default architectures directory."""
    return os.path.join(default_archs_dir, name + '.py')


def arch_name(path: str):
    """ Returns the architecture name for the given architecture path."""
    return path.replace("/", ".").replace("\\", ".").rstrip(".py")


def custom_arch_dir(exp_output_dir: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the architecture directory from the given experiment path.
        Note that the directory is created if it does not exist. """
    proc_dir(os.path.join(exp_output_dir, 'custom_archs'), safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(exp_output_dir, 'custom_archs')


def custom_arch_path(name: str, exp_output_dir: str):
    """ Returns the architecture path for the given architecture name from the given experiment path."""
    return os.path.join(custom_arch_dir(exp_output_dir), name + '.py')

# ----------------------------------------------------------------------------------------------------------------------
#   MODELS
# ----------------------------------------------------------------------------------------------------------------------


def custom_model_dir(exp_output_dir: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the model directory from the given experiment path.
        Note that the directory is created if it does not exist. """
    proc_dir(os.path.join(exp_output_dir, 'models'), safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(exp_output_dir, 'models')


def model_output_dir(exp_output_dir: str, name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the model output directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(custom_model_dir(exp_output_dir, safe_mode=True, overwrite=False), name),
               safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(custom_model_dir(exp_output_dir, safe_mode=True, overwrite=False), name)


def model_args_path(exp_output_dir: str, name: str):
    """ Returns the model args path for the given model name from the given experiment path."""
    return os.path.join(model_output_dir(exp_output_dir, name), 'model_args.yaml')


def model_predictions_dir(exp_output_dir: str, name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the predictions directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'predictions'),
               safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'predictions')


def ckpt_path(exp_output_dir: str, name: str):
    """ Returns the ckpt path for the given model name from the given experiment path.
        Note that if multiple are found, the first one is returned."""
    path = model_output_dir(exp_output_dir, name)
    ckpt_list = []
    for c in os.listdir(path):
        if c.endswith('.ckpt'):
            ckpt_list.append(c)
    if len(ckpt_list) > 1:
        logger.warning('Found %s checkpoints at %s. Selecting first.',
                       str(len(ckpt_list)), path)
    elif len(ckpt_list) == 0:
        logger.warning('Found zero checkpoints at %s.', path)
        exit(101)
    return os.path.join(path, ckpt_list[0])

# ----------------------------------------------------------------------------------------------------------------------
#   INTERPRETATION
# ----------------------------------------------------------------------------------------------------------------------

def model_interpretations_dir(exp_output_dir: str, name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the interpretations directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'interpretations'),
               safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'interpretations')


def interpretation_name(interpret_args: dict):
    """ Construct an interpretation name from interpretation arguments. """
    i_name = ''
    for a in ('interpretation_method', 'interpretation_set', 'selection',
              'size', 'multiply_by_inputs', 'seed', 'i_inx'):
        try:
            i_name += str(interpret_args[a]) + '-'
        except:
            pass
    i_name = i_name[:-1] + '.pickle'

    return i_name


# ----------------------------------------------------------------------------------------------------------------------
#   LEARNING PERFORMANCE DATA
# ----------------------------------------------------------------------------------------------------------------------


def learning_data_path(exp_output_dir: str, name: str):
    """ Returns the learning data path for the given model name from the given experiment path."""
    path = model_output_dir(exp_output_dir, name)
    path = os.path.join(path, 'lightning_logs')
    path = os.path.join(path, 'version_0')
    path = os.path.join(path, 'metrics.csv')
    return path

# ----------------------------------------------------------------------------------------------------------------------
#   VISUALIZATIONS
# ----------------------------------------------------------------------------------------------------------------------

def model_viz_dir(exp_output_dir: str, name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the visualization directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'visualizations'),
               safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'visualizations')

# ----------------------------------------------------------------------------------------------------------------------
#   SWEEPS
# ----------------------------------------------------------------------------------------------------------------------

def model_sweeps_dir(exp_output_dir: str, model_name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the sweeps directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(model_output_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False), 'sweeps'),
             safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_output_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False), 'sweeps')

def model_sweep_dir(exp_output_dir: str, model_name: str, sweep_name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the specific sweep directory for the given model name, and sweep name, from the given experiment path."""
    proc_dir(os.path.join(model_sweeps_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False), sweep_name),
             safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_sweeps_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False), sweep_name)

def model_run_dir(exp_output_dir: str, model_name: str, sweep_name: str, run_name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the specific run directory for the given model name, sweep name, and run name, from the given experiment path."""
    proc_dir(os.path.join(model_sweep_dir(exp_output_dir, model_name, sweep_name, safe_mode=True, overwrite=False), run_name),
             safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_sweep_dir(exp_output_dir, model_name, sweep_name, safe_mode=True, overwrite=False), run_name)

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# ----------------------------------------------------------------------------------------------------------------------
