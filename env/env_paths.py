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


def model_interpretations_dir(exp_output_dir: str, name: str, safe_mode: bool = True, overwrite: bool = False):
    """ Returns the interpretations directory for the given model name from the given experiment path."""
    proc_dir(os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'interpretations'),
               safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_output_dir(exp_output_dir, name, safe_mode=True, overwrite=False), 'interpretations')


def model_interpretations_output_dir(exp_output_dir: str, model_name: str, interpretation_name: str,
                                     safe_mode: bool = True, overwrite: bool = False):
    """ Returns the interpretation output directory for the given model and interpretation name
        from the given experiment path."""
    proc_dir(os.path.join(model_interpretations_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False),
                            interpretation_name), safe_mode=safe_mode, overwrite=overwrite)
    return os.path.join(model_interpretations_dir(exp_output_dir, model_name, safe_mode=True, overwrite=False),
                        interpretation_name)


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
#   LEARNING CURVES
# ----------------------------------------------------------------------------------------------------------------------


def learning_data_path(exp_output_dir: str, name: str):
    """ Returns the learning data path for the given model name from the given experiment path."""
    path = model_output_dir(exp_output_dir, name)
    path = os.path.join(path, 'lightning_logs')
    path = os.path.join(path, 'version_0')
    path = os.path.join(path, 'metrics.csv')
    return path


def learning_curves_path(exp_output_dir: str, name: str):
    """ Returns the learning curves path for the given model name from the given experiment path."""
    path = model_output_dir(exp_output_dir, name)
    path = os.path.join(path, 'learning_curves.png')
    return path


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# ----------------------------------------------------------------------------------------------------------------------
